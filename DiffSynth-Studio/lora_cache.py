import os
import json
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from diffusers.loaders import StableDiffusionLoraLoaderMixin
from dataaset2 import default_lora_patterns
from diffsynth.utils.lora import GeneralLoRALoader
from diffsynth.core import load_state_dict


def build_needed_keys(patterns):
    return (
        set(p["name"] + ".lora_A.weight" for p in patterns)
        | set(p["name"] + ".lora_B.weight" for p in patterns)
    )


def process_worker(rank, world_size, metadata_path, base_path, save_root):
    patterns = default_lora_patterns()
    needed_keys = build_needed_keys(patterns)

    with open(metadata_path, "r", encoding="utf-8") as f:
        datas = [json.loads(line) for line in f]

    sub_datas = datas[rank::world_size]
    print(f"[rank {rank}] total={len(sub_datas)}")
    lora_loader = GeneralLoRALoader()

    for item in tqdm(sub_datas, desc=f"rank{rank}"):
        model_file = item["model_file"]
        # model_id = item["adapter_id"]
        # model_id = model_file.split("/")[-1].split(".")[0]

        full_model_path = os.path.join(base_path, model_file)

        save_path = os.path.join(save_root, model_file.replace(".safetensors", ".pt"))

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        if os.path.exists(save_path):
            continue

        try:
            lora = load_state_dict(full_model_path, torch_dtype=torch.bfloat16, device='cpu')
            lora = lora_loader.convert_state_dict(lora)
            lora = {k: v.cpu() for k, v in lora.items() if k in needed_keys}
            torch.save(lora, save_path)
        except Exception as e:
            print(f"[rank {rank}] failed: {full_model_path} | {e}")


def main():
    metadata_path = "/shark/zhiwen/LoRAHunter/DiffSynth-Studio/rank_dataset/train_rank_lora_all.jsonl"
    # metadata_path = "/shark/zhiwen/LoRAHunter/rank_encoder/train_gcl_topk_2.jsonl"
    # base_path = "/shark/zhiwen/LoRAHunter/sd_lora/sd_lora_1"
    # base_path = "/shark/zhiwen/LoRAHunter/sd_lora/lzwecnu"
    base_path = "/shark/zhiwen/LoRAHunter/Qwen_LoRA"
    save_root = "/shark/zhiwen/LoRAHunter/Qwen_LoRA_cache"

    world_size = min(16, os.cpu_count())
    mp.spawn(
        process_worker,
        args=(world_size, metadata_path, base_path, save_root),
        nprocs=world_size,
        join=True,
    )

def test():
    file_path = "/shark/zhiwen/LoRAHunter/sd_lora/lora_cache/civitai-lora-16k-20k/85/85779.pt"

    lora = torch.load(file_path)

if __name__ == "__main__":
    main()
    # test()

# nohup python lora_cache.py > zlog/lora_cache.log 2>&1 &