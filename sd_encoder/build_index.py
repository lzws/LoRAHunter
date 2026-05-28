import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
import time
import torch
from torch.utils.data import DataLoader
from safetensors.torch import load_file
from tqdm import tqdm

import json
import torch.multiprocessing as mp



from models import TextImageEncoder, QwenVLEncoder
from encoder import LoRAEncoder
from diffusers.loaders import StableDiffusionLoraLoaderMixin
from dataset import default_lora_patterns


import re
import torch.nn.functional as F



class LoRAPoolDataset(torch.utils.data.Dataset):
    def __init__(self, metadata_file, lora_base_path, rank=0, world_size=1):
        with open(metadata_file, "r", encoding="utf-8") as f:
            datas = [json.loads(line) for line in f]

        # shard by rank
        self.datas = datas[rank::world_size]
        self.lora_base_path = lora_base_path
        self.patterns = default_lora_patterns()

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        data = self.datas[idx]
        model_file = data["model_file"]
        lora_path = f"{self.lora_base_path}/{model_file}"

        try:
            lora_dict = StableDiffusionLoraLoaderMixin.lora_state_dict(lora_path)[0]
            needed_keys = set(p["name"] + ".down.weight" for p in self.patterns) | \
                set(p["name"] + ".up.weight" for p in self.patterns)
            lora_dict = {k: v for k, v in lora_dict.items() if k in needed_keys}
            return {
                "model_file": model_file,
                "lora": lora_dict,
            }
        except Exception as e:
            print(f"[Dataset Warning] failed to load: {lora_path}, error: ")
            return None


def lora_pool_collate_fn(batch):
    batch = [x for x in batch if x is not None]

    if len(batch) == 0:
        return None

    return {
        "model_file": [x["model_file"] for x in batch],
        "lora": [x["lora"] for x in batch],
    }


def build_lora_index_worker(
    rank,
    world_size,
    lora_pool_metadata_file,
    lora_base_path,
    encoder_path,
    save_dir,
    batch_size=16,
    num_workers=4,
    dtype=torch.float,
):
    device = f"cuda:{rank}"
    torch.cuda.set_device(rank)

    dataset = LoRAPoolDataset(
        metadata_file=lora_pool_metadata_file,
        lora_base_path=lora_base_path,
        rank=rank,
        world_size=world_size,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lora_pool_collate_fn,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )

    print(f"[Rank {rank}] dataset size: {len(dataset)}")
    print(f"[Rank {rank}] loading encoder from: {encoder_path}")

    folder_path = os.path.dirname(encoder_path)
    config_path = os.path.join(folder_path, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        configs = json.load(f)

    lora_encoder = LoRAEncoder(
        L=configs["L"],
        num_encoder_layers=configs["num_encoder_layers"],
        num_probes=configs["num_probes"],
        embed_dim=configs["embed_dim"],
        encoder_intermediate_size=configs["encoder_intermediate_size"],
        block_type=configs["block_type"],
        head_mode=configs["head_mode"],
        pooling=configs["pooling"],
    )
    if configs['torch_dtype'] == 'bf16':
        dtype = torch.bfloat16
        print(f"[Rank {rank}] using bf16")
    lora_encoder.load_state_dict(load_file(encoder_path))
    lora_encoder = lora_encoder.to(device=device, dtype=dtype)
    lora_encoder.eval()

    all_paths = []
    all_txt_embs = []
    all_img_embs = []

    num_success = 0
    num_failed = 0
    start_time = time.time()

    is_dual = None  # 延迟判断，第一次成功编码后确定

    with torch.no_grad():
        pbar = tqdm(
            dataloader,
            desc=f"Rank {rank} Building Index",
            total=len(dataloader),
            position=rank,
        )

        for batch_id, batch in enumerate(pbar):
            if batch is None:
                continue

            model_files = batch["model_file"]
            loras = batch["lora"]

            valid_files = []
            valid_txt_embs = []
            valid_img_embs = []

            for mf, lora in zip(model_files, loras):
                try:
                    lora = {
                        k: v.to(device=device, dtype=dtype, non_blocking=True)
                        for k, v in lora.items()
                    }

                    emb = lora_encoder(lora)

                    if isinstance(emb, tuple):
                        txt_emb, img_emb = emb
                        if is_dual is None:
                            is_dual = True
                    else:
                        txt_emb = emb
                        img_emb = None
                        if is_dual is None:
                            is_dual = False

                    txt_emb = torch.nn.functional.normalize(txt_emb, dim=-1)

                    valid_files.append(mf)
                    valid_txt_embs.append(txt_emb.cpu())

                    if img_emb is not None:
                        img_emb = torch.nn.functional.normalize(img_emb, dim=-1)
                        valid_img_embs.append(img_emb.cpu())

                    num_success += 1

                except Exception as e:
                    print(f"[Rank {rank} Warning] encode failed: {mf}, error: {e}")
                    num_failed += 1

            if len(valid_txt_embs) > 0:
                valid_txt_embs = torch.cat(valid_txt_embs, dim=0)
                all_paths.extend(valid_files)
                all_txt_embs.append(valid_txt_embs)

                if is_dual:
                    if len(valid_img_embs) != len(valid_files):
                        print(
                            f"[Rank {rank} Warning] dual-head mismatch in batch {batch_id}: "
                            f"{len(valid_img_embs)} img embs vs {len(valid_files)} files"
                        )
                    else:
                        valid_img_embs = torch.cat(valid_img_embs, dim=0)
                        all_img_embs.append(valid_img_embs)

            elapsed = time.time() - start_time
            pbar.set_postfix({
                "success": num_success,
                "failed": num_failed,
                "elapsed": f"{elapsed/60:.1f}m",
                "dual": is_dual,
            })

    if len(all_txt_embs) == 0:
        shard_txt_embs = torch.empty(0, 768)
    else:
        shard_txt_embs = torch.cat(all_txt_embs, dim=0)

    if is_dual:
        if len(all_img_embs) == 0:
            shard_img_embs = torch.empty(0, shard_txt_embs.shape[1])
        else:
            shard_img_embs = torch.cat(all_img_embs, dim=0)
    else:
        shard_img_embs = None

    os.makedirs(save_dir, exist_ok=True)
    shard_path = os.path.join(save_dir, f"lora_index_rank{rank}.pt")

    save_obj = {
        "model_files": all_paths,
        "txt_embeddings": shard_txt_embs,
        "is_dual": bool(is_dual) if is_dual is not None else False,
    }

    if shard_img_embs is not None:
        save_obj["img_embeddings"] = shard_img_embs
    else:
        # 为了兼容你以前的单头读取逻辑，也可以额外保留 embeddings
        save_obj["embeddings"] = shard_txt_embs

    torch.save(save_obj, shard_path)

    total_time = time.time() - start_time
    print(f"[Rank {rank}] saved shard to {shard_path}")
    print(f"[Rank {rank}] success={num_success}, failed={num_failed}")
    print(f"[Rank {rank}] txt shape={shard_txt_embs.shape}")
    if shard_img_embs is not None:
        print(f"[Rank {rank}] img shape={shard_img_embs.shape}")
    print(f"[Rank {rank}] total time={total_time/60:.2f} min")


def merge_lora_index_shards(save_dir, world_size, save_path="lora_index_train.pt"):
    all_paths = []
    all_txt_embs = []
    all_img_embs = []

    merged_is_dual = None

    for rank in range(world_size):
        shard_path = os.path.join(save_dir, f"lora_index_rank{rank}.pt")
        shard = torch.load(shard_path, map_location="cpu")

        shard_is_dual = shard.get("is_dual", False)
        if merged_is_dual is None:
            merged_is_dual = shard_is_dual
        else:
            if merged_is_dual != shard_is_dual:
                raise RuntimeError(
                    f"Inconsistent shard type: rank {rank} has is_dual={shard_is_dual}, "
                    f"expected {merged_is_dual}"
                )

        all_paths.extend(shard["model_files"])
        all_txt_embs.append(shard["txt_embeddings"])

        if merged_is_dual:
            if "img_embeddings" not in shard:
                raise RuntimeError(f"Dual-head shard missing img_embeddings: {shard_path}")
            all_img_embs.append(shard["img_embeddings"])

        print(
            f"[Merge] loaded {shard_path}, "
            f"num={len(shard['model_files'])}, "
            f"is_dual={shard_is_dual}"
        )

    if len(all_txt_embs) == 0:
        raise RuntimeError("No shard embeddings found.")

    all_txt_embs = torch.cat(all_txt_embs, dim=0)

    save_obj = {
        "model_files": all_paths,
        "txt_embeddings": all_txt_embs,
        "is_dual": bool(merged_is_dual),
    }

    if merged_is_dual:
        all_img_embs = torch.cat(all_img_embs, dim=0)
        save_obj["img_embeddings"] = all_img_embs
    else:
        # 保持和旧版本兼容
        save_obj["embeddings"] = all_txt_embs

    torch.save(save_obj, save_path)

    print(f"[Merge] saved merged index to: {save_path}")
    print(f"[Merge] num items: {len(all_paths)}")
    print(f"[Merge] txt embedding shape: {all_txt_embs.shape}")
    if merged_is_dual:
        print(f"[Merge] img embedding shape: {all_img_embs.shape}")


def build_lora_index_multi_gpu(
    lora_pool_metadata_file,
    lora_base_path,
    encoder_path,
    save_path="lora_index_train.pt",
    batch_size=16,
    num_workers=4,
    dtype=torch.float,
    num_gpus=None,
    save_dir="lora_index_shards",
):
    if num_gpus is None:
        num_gpus = torch.cuda.device_count()

    assert num_gpus > 0, "No CUDA devices found"

    print(f"[Main] using {num_gpus} GPUs")
    print(f"[Main] shard save dir: {save_dir}")

    mp.spawn(
        build_lora_index_worker,
        args=(
            num_gpus,
            lora_pool_metadata_file,
            lora_base_path,
            encoder_path,
            save_dir,
            batch_size,
            num_workers,
            dtype,
        ),
        nprocs=num_gpus,
        join=True,
    )

    merge_lora_index_shards(
        save_dir=save_dir,
        world_size=num_gpus,
        save_path=save_path,
    )


def get_basename(path):
    return os.path.basename(path.strip())


def load_train_model_files(train_jsonl_path):
    train_model_files = set()
    with open(train_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            mf = data.get("model_file", None)
            if mf is not None:
                train_model_files.add(get_basename(mf))
    return train_model_files


def split_lora_index(
    index_path,
    train_jsonl_path,
    train_save_path,
    unseen_save_path,
):
    # 1. 读取训练集里的 basename
    train_model_files = load_train_model_files(train_jsonl_path)
    print(f"[Info] loaded {len(train_model_files)} unique train basenames")

    # 2. 读取总索引
    index_data = torch.load(index_path, map_location="cpu")
    model_files = index_data["model_files"]
    is_dual = index_data.get("is_dual", False)

    if "txt_embeddings" in index_data:
        txt_embeddings = index_data["txt_embeddings"]
    elif "embeddings" in index_data:
        txt_embeddings = index_data["embeddings"]
    else:
        raise KeyError("Index file must contain 'txt_embeddings' or 'embeddings'")

    img_embeddings = index_data.get("img_embeddings", None)

    # 3. 按 basename 分组
    train_indices = []
    unseen_indices = []

    for i, mf in enumerate(model_files):
        base = get_basename(mf)
        if base in train_model_files:
            train_indices.append(i)
        else:
            unseen_indices.append(i)

    print(f"[Split] train in index: {len(train_indices)}")
    print(f"[Split] unseen in index: {len(unseen_indices)}")

    # 4. 构造子索引
    def build_subset(indices):
        subset = {
            "model_files": [model_files[i] for i in indices],
            "is_dual": is_dual,
        }

        subset_txt = txt_embeddings[indices]
        subset["txt_embeddings"] = subset_txt

        if is_dual and img_embeddings is not None:
            subset["img_embeddings"] = img_embeddings[indices]
        else:
            subset["embeddings"] = subset_txt

        return subset

    train_index = build_subset(train_indices)
    unseen_index = build_subset(unseen_indices)

    # 5. 保存
    torch.save(train_index, train_save_path)
    torch.save(unseen_index, unseen_save_path)

    print(f"[Done] saved train index -> {train_save_path}")
    print(f"[Done] saved unseen index -> {unseen_save_path}")

    print(f"[Train Index] num: {len(train_index['model_files'])}, txt shape: {train_index['txt_embeddings'].shape}")
    if is_dual and "img_embeddings" in train_index:
        print(f"[Train Index] img shape: {train_index['img_embeddings'].shape}")

    print(f"[Unseen Index] num: {len(unseen_index['model_files'])}, txt shape: {unseen_index['txt_embeddings'].shape}")
    if is_dual and "img_embeddings" in unseen_index:
        print(f"[Unseen Index] img shape: {unseen_index['img_embeddings'].shape}")


def find_train_not_in_index(
    train_jsonl_path,
    index_path,
    save_txt_path="missing_in_index.txt",
    save_json_path="missing_in_index.json",
):
    train_basenames, train_records = load_train_basenames(train_jsonl_path)
    index_basenames, index_records = load_index_basenames(index_path)

    missing_basenames = sorted(train_basenames - index_basenames)

    print(f"[Info] train unique basenames: {len(train_basenames)}")
    print(f"[Info] index unique basenames: {len(index_basenames)}")
    print(f"[Info] missing basenames (in train but not in index): {len(missing_basenames)}")

    # 找回这些 basename 对应的原始训练样本
    missing_records = [r for r in train_records if r["basename"] in set(missing_basenames)]

    # 保存 txt
    with open(save_txt_path, "w", encoding="utf-8") as f:
        for name in missing_basenames:
            f.write(name + "\n")

    # 保存 json
    with open(save_json_path, "w", encoding="utf-8") as f:
        json.dump(missing_records, f, ensure_ascii=False, indent=2)

    print(f"[Done] saved missing basename list to: {save_txt_path}")
    print(f"[Done] saved missing record list to: {save_json_path}")

    # 打印前几个例子
    print("\n[Examples: first 20 missing basenames]")
    for x in missing_basenames[:20]:
        print(x)

    return missing_basenames, missing_records

    

if __name__ == "__main__":
    build_lora_index_multi_gpu(
        lora_pool_metadata_file="/shark/zhiwen/LoRAHunter/SD_adapter_metadata/sd_lora_1/exist_file_adapters.jsonl",
        lora_base_path="/shark/zhiwen/LoRAHunter/sd_lora/sd_lora_1",
        encoder_path="/shark/zhiwen/LoRAHunter/sd_encoder/models/lora_encoder/qwenemb/train_sd_lora_dataset_20k/contrastive_loss/0/lora_encoder-196.safetensors",
        save_path="qwenemb_contrastive_all_0-196.pt",
        batch_size=2,
        num_workers=2,
        dtype=torch.bfloat16,
        num_gpus=4,
        save_dir="lora_index_shards_clip",
    )

    # split_lora_index(
    #     index_path="lora_index/qwenemb_cosinsmoothl1_all_5-199.pt",
    #     train_jsonl_path="/shark/zhiwen/LoRAHunter/sd_encoder/train_sd_lora_dataset_20k.jsonl",
    #     train_save_path="lora_index/qwenemb_cosinsmoothl1_all_5-199_train.pt",
    #     unseen_save_path="lora_index/qwenemb_cosinsmoothl1_all_5-199_unseen.pt",
    # )

    # find_train_not_in_index(
    #     train_jsonl_path="train_sd_lora_dataset.jsonl",
    #     index_path="lora_index.pt",
    #     save_txt_path="missing_in_index.txt",
    #     save_json_path="missing_in_index.json",
    # )
# nohup python build_index.py > zlog/build_index.log 2>&1 &
# export TMPDIR=/shark/zhiwen/LoRAHunter/cache
