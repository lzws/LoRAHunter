import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import time
import torch
from torch.utils.data import DataLoader
from safetensors.torch import load_file
from tqdm import tqdm

import json
import torch
import torch.multiprocessing as mp



from models import TextImageEncoder
from encoder import LoRAEncoder
from diffusers.loaders import StableDiffusionLoraLoaderMixin
from dataset import default_lora_patterns


import re
import torch.nn.functional as F
from rank_bm25 import BM25Okapi


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

    lora_encoder = LoRAEncoder(L=1,num_encoder_layers=8,num_probes=8)
    lora_encoder.load_state_dict(load_file(encoder_path))
    lora_encoder = lora_encoder.to(device=device, dtype=dtype)
    lora_encoder.eval()

    all_paths = []
    all_embs = []

    num_success = 0
    num_failed = 0
    start_time = time.time()

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
            valid_embs = []

            for mf, lora in zip(model_files, loras):
                try:
                    lora = {
                        k: v.to(device=device, dtype=dtype, non_blocking=True)
                        for k, v in lora.items()
                    }
                    emb = lora_encoder(lora)   # [1, D]
                    emb = torch.nn.functional.normalize(emb, dim=-1)

                    valid_files.append(mf)
                    valid_embs.append(emb.cpu())
                    num_success += 1
                except Exception as e:
                    print(f"[Rank {rank} Warning] encode failed: {mf}, error: {e}")
                    num_failed += 1

            if len(valid_embs) > 0:
                valid_embs = torch.cat(valid_embs, dim=0)
                all_paths.extend(valid_files)
                all_embs.append(valid_embs)

            elapsed = time.time() - start_time
            pbar.set_postfix({
                "success": num_success,
                "failed": num_failed,
                "elapsed": f"{elapsed/60:.1f}m",
            })

    if len(all_embs) == 0:
        shard_embs = torch.empty(0, 768)
    else:
        shard_embs = torch.cat(all_embs, dim=0)

    os.makedirs(save_dir, exist_ok=True)
    shard_path = os.path.join(save_dir, f"lora_index_rank{rank}.pt")

    torch.save({
        "model_files": all_paths,
        "embeddings": shard_embs,
    }, shard_path)

    total_time = time.time() - start_time
    print(f"[Rank {rank}] saved shard to {shard_path}")
    print(f"[Rank {rank}] success={num_success}, failed={num_failed}")
    print(f"[Rank {rank}] shape={shard_embs.shape}")
    print(f"[Rank {rank}] total time={total_time/60:.2f} min")


def merge_lora_index_shards(save_dir, world_size, save_path="lora_index_train.pt"):
    all_paths = []
    all_embs = []

    for rank in range(world_size):
        shard_path = os.path.join(save_dir, f"lora_index_rank{rank}.pt")
        shard = torch.load(shard_path, map_location="cpu")

        all_paths.extend(shard["model_files"])
        all_embs.append(shard["embeddings"])

        print(f"[Merge] loaded {shard_path}, num={len(shard['model_files'])}")

    if len(all_embs) == 0:
        raise RuntimeError("No shard embeddings found.")

    all_embs = torch.cat(all_embs, dim=0)

    torch.save({
        "model_files": all_paths,
        "embeddings": all_embs,
    }, save_path)

    print(f"[Merge] saved merged index to: {save_path}")
    print(f"[Merge] num items: {len(all_paths)}")
    print(f"[Merge] embedding shape: {all_embs.shape}")



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






class LoRARetriever:
    def __init__(
        self,
        index_path,
        metadata_path,
        device="cuda",
        alpha=0.7,
    ):
        """
        alpha:
            hybrid 模式下 dense 的权重
            bm25 权重 = 1 - alpha
        """
        self.device = device
        self.alpha = alpha

        # dense encoder
        self.clip_encoder = TextImageEncoder().to(device)
        self.clip_encoder.eval()

        # load dense index
        index_data = torch.load(index_path, map_location="cpu")
        self.model_files = index_data["model_files"]
        self.lora_embs = index_data["embeddings"].float()

        print(f"[Index] loaded dense index: {self.lora_embs.shape}")

        # normalize once
        self.lora_embs = F.normalize(self.lora_embs, dim=-1).to(device)

        # load metadata
        with open(metadata_path, "r", encoding="utf-8") as f:
            all_datas = [json.loads(line) for line in f]

        metadata_map = {}
        for data in all_datas:
            model_file = data.get("model_file", "")
            metadata_map[model_file] = data

        # build bm25 corpus aligned with dense index order
        self.documents = []
        self.tokenized_corpus = []

        missing_count = 0
        for model_file in self.model_files:
            data = metadata_map.get(model_file, None)
            if data is None:
                doc = ""
                missing_count += 1
            else:
                title = data.get("title", "")
                description = data.get("llm_description", "")
                tags = data.get("tags", [])
                if isinstance(tags, list):
                    tag_str = " ".join(tags)
                else:
                    tag_str = str(tags)

                doc = f"{title}. {description}. {tag_str}".strip()

            self.documents.append(doc)
            self.tokenized_corpus.append(self._tokenize(doc))

        self.bm25 = BM25Okapi(self.tokenized_corpus)

        print(f"[Index] built BM25 corpus: {len(self.documents)} docs")
        print(f"[Index] metadata missing for {missing_count} docs")

    def _tokenize(self, text):
        if not text:
            return []
        return re.findall(r"\w+", text.lower())

    def _minmax_norm(self, scores: torch.Tensor):
        min_v = scores.min()
        max_v = scores.max()
        if (max_v - min_v).abs() < 1e-12:
            return torch.zeros_like(scores)
        return (scores - min_v) / (max_v - min_v)

    @torch.no_grad()
    def retrieve_dense_scores(self, query_text):
        text_emb = self.clip_encoder.encoding_text([query_text])   # [1, D]
        text_emb = F.normalize(text_emb, dim=-1)
        scores = (text_emb @ self.lora_embs.t())[0]   # [N]
        return scores.detach().cpu()

    def retrieve_bm25_scores(self, query_text):
        tokenized_query = self._tokenize(query_text)
        scores = self.bm25.get_scores(tokenized_query)   # numpy [N]
        return torch.tensor(scores, dtype=torch.float)

    @torch.no_grad()
    def retrieve(self, query_text, top_k=5, mode="dense"):
        """
        mode:
            - dense
            - bm25
            - hybrid
        """
        mode = mode.lower()
        assert mode in ["dense", "bm25", "hybrid"], f"Unsupported mode: {mode}"

        if mode == "dense":
            dense_scores = self.retrieve_dense_scores(query_text)
            top_scores, top_indices = torch.topk(dense_scores, k=top_k)

            results = []
            for score, idx in zip(top_scores.tolist(), top_indices.tolist()):
                results.append({
                    "model_file": self.model_files[idx],
                    "score": score,
                    "dense_score": score,
                    "bm25_score": None,
                    "mode": "dense",
                })
            return results

        elif mode == "bm25":
            bm25_scores = self.retrieve_bm25_scores(query_text)
            top_scores, top_indices = torch.topk(bm25_scores, k=top_k)

            results = []
            for score, idx in zip(top_scores.tolist(), top_indices.tolist()):
                results.append({
                    "model_file": self.model_files[idx],
                    "score": score,
                    "dense_score": None,
                    "bm25_score": score,
                    "mode": "bm25",
                })
            return results

        else:  # hybrid
            dense_scores = self.retrieve_dense_scores(query_text)
            bm25_scores = self.retrieve_bm25_scores(query_text)

            dense_scores_norm = self._minmax_norm(dense_scores)
            bm25_scores_norm = self._minmax_norm(bm25_scores)

            final_scores = self.alpha * dense_scores_norm + (1.0 - self.alpha) * bm25_scores_norm

            top_scores, top_indices = torch.topk(final_scores, k=top_k)

            results = []
            for score, idx in zip(top_scores.tolist(), top_indices.tolist()):
                results.append({
                    "model_file": self.model_files[idx],
                    "score": score,
                    "dense_score": dense_scores[idx].item(),
                    "bm25_score": bm25_scores[idx].item(),
                    "dense_score_norm": dense_scores_norm[idx].item(),
                    "bm25_score_norm": bm25_scores_norm[idx].item(),
                    "mode": "hybrid",
                })
            return results


def call_lora(
    index_path="lora_index_train.pt",
    metadata_path="lora_pool_metadata.jsonl",
    test_data_path="retrieval_testdata_100.jsonl",
    output_path="test_data/call_lora_res.jsonl",
    device="cuda",
    mode="dense",   # dense / bm25 / hybrid
    top_k=10,
    alpha=0.3,
):
    retriever = LoRARetriever(
        index_path=index_path,
        metadata_path=metadata_path,
        device=device,
        alpha=alpha,
    )

    with open(test_data_path, "r", encoding="utf-8") as f:
        test_datas = [json.loads(line) for line in f]

    with open(output_path, "w", encoding="utf-8") as fout:
        for i, data in enumerate(test_datas):
            print(f"[{i}]")
            extract_concept = data["extract_concept"]
            data["retrieval_results"] = {}

            for ec in extract_concept:
                keyword = ec["keyword"]
                retrieval_description = ec["retrieval_description"]

                query = f"a '{keyword}' adapter. " + retrieval_description
                print(f"  query: {query}")

                topk = retriever.retrieve(
                    query_text=query,
                    top_k=top_k,
                    mode=mode,
                )
                data["retrieval_results"][keyword] = topk

            fout.write(json.dumps(data, ensure_ascii=False) + "\n")

    print(f"[Done] saved results to {output_path}")




if __name__ == "__main__":
    # build_lora_index_multi_gpu(
    #     lora_pool_metadata_file="/shark/zhiwen/LoRAHunter/SD_adapter_metadata/sd_lora_1/exist_file_adapters.jsonl",
    #     lora_base_path="/shark/zhiwen/LoRAHunter/sd_lora/sd_lora_1",
    #     encoder_path="/shark/zhiwen/LoRAHunter/sd_encoder/models/lora_encoder/clipemb/train_sd_lora_dataset_2/7/lora_encoder-108.safetensors",
    #     save_path="lora_index_clipemb-7_all.pt",
    #     batch_size=16,
    #     num_workers=8,
    #     dtype=torch.float,
    #     num_gpus=1,
    #     save_dir="lora_index_shards",
    # )

    call_lora(
        index_path="lora_index_clipemb-7_all.pt",
        metadata_path="../SD_adapter_metadata/sd_lora_1/exist_file_adapters.jsonl",
        test_data_path="../test_data/retrieval_testdata_250_extract.jsonl",
        output_path="test_data/250_clipemb-7_all_res.jsonl",
        device="cuda",
        mode="hybrid",   # dense / bm25 / hybrid
        top_k=40,
        alpha=0.4,
    )

# nohup python call_lora.py > build_lora_index.log 2>&1 &