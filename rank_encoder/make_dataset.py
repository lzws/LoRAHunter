import os
import json
from tqdm import tqdm
from models import QwenVLEncoder
import torch
from compute_reward.CARLoS_prompt import prompts_by_category
import pandas as pd
import torch.nn.functional as F
import random

lora_metadata_path = "/shark/zhiwen/LoRAHunter/SD_adapter_metadata/sd_lora_1/exist_file_adapters.jsonl"

'''
每个prompt挑选
top20 + random5 + semi-hard5 (semi-hard 就是从 rank 20~200 里随机抽几个)

然后每个lora生成5张图片，底模生成图片，算出分数
'''


def get_train_prompt():
    '''
    1. 从coco里面挑选1w条prompt
    2. CARLos 中所有的prompt 560条
    '''
    all_datas = []
    global_id = 0
    coco_path = "/shark/zhiwen/benchmark/EraseBenchmark/dataset/dataset/coco_30k.csv"
    df = pd.read_csv(coco_path)
    df_sampled = df.sample(n=10000, random_state=42)
    for i, row in df_sampled.iterrows():
        image_id = row['image_id']
        prompt = row['prompt']
        all_datas.append({
            "iid":global_id,
            "source": "coco",
            "source_id": image_id,
            "prompt": prompt
        })
        global_id += 1
    
    for category in prompts_by_category:
        for sub_category, prompts in prompts_by_category[category].items():
            for p in prompts:
                all_datas.append({
                    "iid":global_id,
                    "source": "CARLos",
                    "source_id": category,
                    "prompt": p
                })
                global_id += 1
    

    diffusion_db_path = "/shark/zhiwen/lora_diff_length/diffusiondb_5000.csv"
    df = pd.read_csv(diffusion_db_path)
    # df_sampled = df.sample(n=5000, random_state=42)
    for i, row in df.iterrows():
        prompt = row['prompt']
        all_datas.append({
            "iid":global_id,
            "source": "diffusiondb",
            "source_id": i,
            "prompt": prompt
        })
        global_id += 1
    
    diffusion_db_path = "diffusiondb_5000_2.csv"
    df = pd.read_csv(diffusion_db_path)
    # df_sampled = df.sample(n=5000, random_state=42)
    for i, row in df.iterrows():
        prompt = row['prompt']
        all_datas.append({
            "iid":global_id,
            "source": "diffusiondb",
            "source_id": i,
            "prompt": prompt
        })
        global_id += 1

    print(f"[Done] collected {len(all_datas)} prompts")
    with open("train_prompts_2.jsonl", "w", encoding="utf-8") as f:
        for data in all_datas:
            json_str = json.dumps(data, ensure_ascii=False)
            f.write(json_str + '\n')
        



# 1、先给整个 lora 库构造索引，用 qwenvl-embedding，编码 llm-description

def main(device="cuda"):
    # load encoder
    encoder = QwenVLEncoder(device=device).to(device)
    # load lora metadata
    with open(lora_metadata_path, 'r', encoding='utf-8') as f:
        datas = [json.loads(line) for line in f.readlines()]

    all_embs = []
    all_adapter_ids = []
    all_model_files = []

    output_path = "lora_text_index_qwen.pt"
    batch_size = 32
    # 2. 开始分批处理
    for i in tqdm(range(0, len(datas), batch_size), desc="Encoding LoRA texts"):
        # 切片获取当前批次
        batch_datas = datas[i : i + batch_size]
        
        batch_texts = []
        batch_ids = []
        batch_model_files = []

        for data in batch_datas:
            model_id = data['adapter_id']
            llm_description = data['llm_description']
            title = data['title']
            tags = data['tags']
            model_file = data['model_file']

            query_text = (
                f"Convert Stable Diffusion finetuned adapter description into an embedding for search: "
                f"Title: {title}; Description: {llm_description}; Tags: {tags};"
            )
            batch_texts.append(query_text)
            # batch_ids.append(model_id)
            batch_model_files.append(model_file)

        # 4. 批量推理 [batch_size, 2048]
        with torch.no_grad():
            batch_embs = encoder.encoding_text(batch_texts)
        batch_embs = batch_embs.detach().cpu()

        all_embs.append(batch_embs)
        all_model_files.extend(batch_model_files)
    
    all_embs = torch.cat(all_embs, dim=0)
    save_obj = {
        "embeddings": all_embs,
        "model_files": all_model_files
    }
    torch.save(save_obj, output_path)
    print(f"[Done] saved index to {output_path}")
    print(f"embeddings shape: {all_embs.shape}")
    print(f"num adapter_ids: {len(all_model_files)}")


def build_prompt_embedding_index(
    batch_size=32,
    device="cuda:0",
):
    """
    读取 prompt jsonl，批量编码，保存到一个 pt 文件中
    """
    prompts_path = "train_dataset/train_prompts_2.jsonl"
    output_path = "train_dataset/train_prompt_index_qwen_2.pt"

    encoder = QwenVLEncoder(device=device).to(device)

    datas = []
    with open(prompts_path, "r", encoding="utf-8") as f:
        for line in f:
            datas.append(json.loads(line))

    all_embs = []
    all_prompts = []
    all_ids = []

    for i in tqdm(range(0, len(datas), batch_size), desc="Encoding prompts"):
        batch_datas = datas[i:i + batch_size]
        batch_prompts = []
        batch_ids = []

        for row in batch_datas:
            prompt = row["prompt"]
            iid = row.get("iid", i)

            batch_prompts.append(prompt)
            batch_ids.append(iid)

        with torch.no_grad():
            batch_embs = encoder.encoding_text(batch_prompts)   # [B, D]
            batch_embs = F.normalize(batch_embs, dim=-1)

        all_embs.append(batch_embs.cpu())
        all_prompts.extend(batch_prompts)
        all_ids.extend(batch_ids)

    all_embs = torch.cat(all_embs, dim=0)   # [N, D]

    id_to_index = {str(x): i for i, x in enumerate(all_ids)}

    torch.save(
        {
            "ids": all_ids,
            "prompts": all_prompts,
            "embeddings": all_embs,
            "id_to_index": id_to_index,
        },
        output_path,
    )

    print(f"[Done] saved prompt index to {output_path}")
    print(f"embeddings shape: {all_embs.shape}")


    
def load_prompt_index(prompt_index_path, device="cpu"):
    data = torch.load(prompt_index_path, map_location="cpu")
    return {
        "ids": data["ids"],
        "prompts": data["prompts"],
        "embeddings": F.normalize(data["embeddings"], dim=-1).to(device),
        "id_to_index": data["id_to_index"],
    }


def load_lora_index(lora_index_path, device="cpu"):
    data = torch.load(lora_index_path, map_location="cpu")

    model_files = data.get("model_files", None)
    adapter_ids = data.get("adapter_ids", None)

    if model_files is None:
        model_files = adapter_ids

    return {
        "model_files": model_files,
        "adapter_ids": adapter_ids,
        "embeddings": F.normalize(data["embeddings"], dim=-1).to(device),
    }


def sample_candidates_for_one_prompt(
    prompt_emb,                  # [D]
    lora_embs,                   # [N_lora, D]
    model_files,
    top_k=20,
    semi_hard_k=5,
    random_k=5,
    semi_hard_range=(20, 100),
):
    sims = torch.matmul(lora_embs, prompt_emb)   # [N_lora]
    sims_cpu = sims.cpu()

    ranked_idxs = torch.argsort(sims_cpu, descending=True).tolist()

    # top
    top_idxs = ranked_idxs[:top_k]

    # semi-hard
    sh_start, sh_end = semi_hard_range
    semi_pool = ranked_idxs[sh_start:sh_end]
    semi_pool = [idx for idx in semi_pool if idx not in top_idxs]
    semi_hard_idxs = random.sample(semi_pool, k=min(semi_hard_k, len(semi_pool)))

    # random
    selected = set(top_idxs + semi_hard_idxs)
    all_pool = list(range(len(model_files)))
    random_pool = [idx for idx in all_pool if idx not in selected]
    random_idxs = random.sample(random_pool, k=min(random_k, len(random_pool)))

    rank_map = {idx: rank + 1 for rank, idx in enumerate(ranked_idxs)}

    def build_items(idxs, source):
        items = []
        for idx in idxs:
            items.append({
                "model_file": model_files[idx],
                "retrieval_score": float(sims_cpu[idx].item()),
                "rank": rank_map[idx],
                "source": source,
            })
        return items

    return (
        build_items(top_idxs, "top")
        + build_items(semi_hard_idxs, "semi_hard")
        + build_items(random_idxs, "random")
    )


def augment_jsonl_with_lora_candidates(
    input_jsonl_path,
    output_jsonl_path,
    prompt_index_path,
    lora_index_path,
    device="cuda:0",
    top_k=20,
    semi_hard_k=5,
    random_k=5,
    semi_hard_range=(20, 100),
    candidate_field="candidate_loras",
):
    prompt_index = load_prompt_index(prompt_index_path, device=device)
    lora_index = load_lora_index(lora_index_path, device=device)

    prompt_embs = prompt_index["embeddings"]          # [N_prompt, D]
    id_to_index = prompt_index["id_to_index"]

    lora_embs = lora_index["embeddings"]              # [N_lora, D]
    model_files = lora_index["model_files"]

    with open(input_jsonl_path, "r", encoding="utf-8") as fin, \
         open(output_jsonl_path, "w", encoding="utf-8") as fout:

        for line_idx, line in enumerate(tqdm(fin, desc="Augmenting prompts")):
            row = json.loads(line)

            # 优先按 iid 找 embedding
            iid = row.get("iid", None)

            if iid is not None and str(iid) in id_to_index:
                emb_idx = id_to_index[str(iid)]
            else:
                # fallback：按顺序取
                emb_idx = line_idx
                if emb_idx >= prompt_embs.shape[0]:
                    print(f"[Warning] line_idx={line_idx} out of range, skip")
                    continue

            prompt_emb = prompt_embs[emb_idx]

            candidates = sample_candidates_for_one_prompt(
                prompt_emb=prompt_emb,
                lora_embs=lora_embs,
                model_files=model_files,
                top_k=top_k,
                semi_hard_k=semi_hard_k,
                random_k=random_k,
                semi_hard_range=semi_hard_range,
            )

            # 保持原格式，只新增字段
            row[candidate_field] = candidates

            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[Done] saved augmented jsonl to {output_jsonl_path}")
    


def generate_images(num=0,start=0,end=100):
    root_lora_dir = "/shark/zhiwen/LoRAHunter/sd_lora/sd_lora_1"
    metadata_path = "/shark/zhiwen/LoRAHunter/rank_encoder/train_dataset/train_prompts_candidates.jsonl"
    with open(metadata_path, "r", encoding="utf-8") as f:
        datas = [json.loads(line) for line in f.readlines()]
    seeds = [3428,746347,7855463,64573,8563,3274385]

    device = f"cuda:{num}"
    bs = 5
    print(f"[Info] device: {device}, start: {start}, end: {end}")

    # load sd 1.5 pipe
    pipe = StableDiffusionPipeline.from_pretrained(
        "/shark/zhiwen/LoRAHunter/DiffSynth-Studio/models/AI-ModelScope/stable-diffusion-v1-5", 
        torch_dtype=torch.bfloat16
    )
    pipe.safety_checker = None
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)

    generators = [torch.Generator(device=device).manual_seed(seed) for seed in seeds[:bs]]

    for data in datas[start:end]:
        iid = data["iid"]
        prompt = data["prompt"]
        candidates = data["candidate_loras"]

        img_save_path_sd = f"images/{iid}/sdv15"
        os.makedirs(img_save_path_sd, exist_ok=True)

        # 生成底模图片
        images = pipe(
            prompt, 
            negative_prompt="", 
            num_inference_steps=35,
            num_images_per_prompt=bs,
            generator=generators,
            guidance_scale=7.0
        ).images
        for seed, image in zip(seeds[:bs], images):
            image.save(f"{img_save_path_sd}/{seed}.jpg")
        
        try:
            for candidate in candidates:
                model_file = candidate["model_file"]
                model_id = model_file.split("/")[-1].replace(".safetensors", "")

                img_save_path_lora = f"images/{iid}/{model_id}"
                os.makedirs(img_save_path_lora, exist_ok=True)

                pipe.load_lora_weights(f"{root_lora_dir}/{model_file}")
                images = pipe(
                    prompt, 
                    negative_prompt="", 
                    num_inference_steps=35,
                    num_images_per_prompt=bs,
                    generator=generators,
                    guidance_scale=7.0,
                ).images
                for seed, image in zip(seeds[:bs], images):
                    image.save(f"{img_save_path_lora}/{seed}.jpg")
                
                pipe.unload_lora_weights()

        except Exception as e:
            print("++"*10)
            print(f"prompt: {iid}, model_id: {model_id}, Error: {e}")
            print("++"*10)
            with open('error_log_3.jsonl', 'a', encoding='utf-8') as f:
            # json.dumps 将字典转为 JSON 字符串
                f.write(json.dumps({'model_id':model_id,'msg':str(e)}, ensure_ascii=False) + '\n')
        


def sample_diffusiondb():
    
    data_path = '/shark/zhiwen/LoRA-fusion/dataset/prompt/metadata.parquet'
    data = pd.read_parquet(data_path,engine='pyarrow')

    subset = data[['prompt', 'seed']].sample(n=5000)
    subset.to_csv('diffusiondb_5000_2.csv',index=False)


if __name__ == "__main__":
    # get_train_prompt()
    # build_prompt_embedding_index()

    augment_jsonl_with_lora_candidates(
        input_jsonl_path="diffusion_db2.jsonl",
        output_jsonl_path="train_dataset/diffusion_db2_candidates.jsonl",
        prompt_index_path="train_dataset/train_prompt_index_qwen_2.pt",
        lora_index_path="train_dataset/lora_text_index_qwen.pt",
        device="cuda:0",
        top_k=20,
        semi_hard_k=5,
        random_k=5,
        semi_hard_range=(20, 150),
        candidate_field="candidate_loras",
    )


