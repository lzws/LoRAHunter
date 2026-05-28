import os,json
import torch
from diffsynth.core import load_state_dict
from diffsynth.utils.lora import GeneralLoRALoader
from dataaset2 import default_lora_patterns
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

base_path = '/shark/zhiwen/LoRAHunter/Qwen_LoRA'
cache_path = '/shark/zhiwen/LoRAHunter/Qwen_LoRA_cache'
lora_patterns = default_lora_patterns()

def main():
    metadata_path = '/shark/zhiwen/LoRAHunter/DiffSynth-Studio/rank_dataset/train_rank_lora_all.jsonl'

    with open(metadata_path, 'r') as f:
        datas = [json.loads(line) for line in f.readlines()]

    broken_loras = []

    for data in tqdm(datas):
        model_file = data['model_file']
        full_model_path = f'{base_path}/{model_file}'
        cache_model_path = f'{cache_path}/{model_file.replace(".safetensors", ".pt")}'

        start = time.time()
        if os.path.exists(cache_model_path):
            lora = torch.load(cache_model_path, map_location="cpu", weights_only=True)
        else:
            lora = load_state_dict(full_model_path, torch_dtype=torch.bfloat16, device='cpu')
            lora = lora_loader.convert_state_dict(lora)
        end = time.time()

        print(f"{model_file} loaded in {end - start:.4f} seconds")

        lora_cnt = 0
        miss_cnt = 0

        start = time.time()
        for lora_pattern in lora_patterns:
            name, layer_type = lora_pattern["name"], lora_pattern["type"]
            name_key = name.replace(".", "___")
            type_key = layer_type.replace(".", "___")

            A_key = name + ".lora_A.weight"
            B_key = name + ".lora_B.weight"


            if A_key in lora and B_key in lora:
                lora_A = lora[A_key]
                lora_B = lora[B_key]
                lora_cnt += 1
                
            else:
                # missing layer -> zero token
                miss_cnt += 1

        end = time.time()
        print(f"read layers in {end - start:.4f} seconds")

        if miss_cnt > 0:
            print(f"{model_file} is missing {miss_cnt} layers")

        if miss_cnt >= 600:
            print(f"{model_file} is missing all layers")
            broken_loras.append(model_file)
            nn = 0
            for k, v in lora.items():
                print(k)
                nn += 1
                if nn > 5:
                    break

        with open('broken_loras.txt', 'a') as f:
            f.write(model_file + '\n')
    
metadata_path = '/shark/zhiwen/LoRAHunter/DiffSynth-Studio/rank_dataset/train_rank_lora_all.jsonl'

with open(metadata_path, 'r', encoding='utf-8') as f:
    datas = [json.loads(line) for line in f if line.strip()]

def check_one(data):
    model_file = data['model_file']
    full_model_path = f'{base_path}/{model_file}'
    cache_model_path = f'{cache_path}/{model_file.replace(".safetensors", ".pt")}'

    try:
        if os.path.exists(cache_model_path):
            lora = torch.load(cache_model_path, map_location="cpu", weights_only=True)
        else:
            lora = load_state_dict(full_model_path, torch_dtype=torch.bfloat16, device='cpu')
            lora = lora_loader.convert_state_dict(lora)

        miss_cnt = 0
        for p in lora_patterns:
            name = p["name"]
            A_key = name + ".lora_A.weight"
            B_key = name + ".lora_B.weight"
            if not (A_key in lora and B_key in lora):
                miss_cnt += 1

        return model_file if miss_cnt >= 600 else None

    except Exception:
        return model_file

broken_loras = []
with ThreadPoolExecutor(max_workers=16) as executor:
    futures = [executor.submit(check_one, data) for data in datas]
    for future in tqdm(as_completed(futures), total=len(futures)):
        result = future.result()
        if result is not None:
            broken_loras.append(result)

with open('broken_loras.txt', 'w', encoding='utf-8') as f:
    for x in broken_loras:
        f.write(x + '\n')
