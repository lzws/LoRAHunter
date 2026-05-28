import os,json
from modules.models import TextImageEncoder, QwenVLEncoder
from tqdm import tqdm
import torch
import torch.nn.functional as F
import random

# 0. 从源文件找到lora的title
def find_metainfo():
    metadata_path = "/shark/zhiwen/LoRAHunter/source_files/LoRA_QWEN_IMAGE_20_B_json.jsonl"
    with open(metadata_path, 'r', encoding='utf-8') as f:
        datas = [json.loads(line) for line in f.readlines()]
    meta_map = {data['model_id']: data for data in datas}

    input_file = "/shark/zhiwen/LoRAHunter/source_files/Available_LoRA_all.jsonl"
    with open(input_file, 'r', encoding='utf-8') as f:
        input_datas = [json.loads(line) for line in f.readlines()]
    
    for data in input_datas:
        model_id = data['model_id']
        meta_info = meta_map[model_id]
        data['title'] = meta_info['title']
        data['tags'] = meta_info['tags']
    
    with open(input_file, "w", encoding="utf-8") as f:
        for item in input_datas:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"done {len(input_datas)} data")




# 1. 建立所有lora的文本索引
def build_lora_database():
    device = "cuda:0"
    lora_metadata_path = "/shark/zhiwen/LoRAHunter/source_files/Available_LoRA_all.jsonl"
    # 1、先给整个 lora 库构造索引，用 qwenvl-embedding，编码 llm-description
    encoder = QwenVLEncoder(device=device).to(device)
    # load lora metadata
    with open(lora_metadata_path, 'r', encoding='utf-8') as f:
        datas = [json.loads(line) for line in f.readlines()]

    all_embs = []
    all_adapter_ids = []
    all_model_files = []

    output_path = "qwenlora_desc_index_qwen.pt"
    batch_size = 32
    # 2. 开始分批处理
    for i in tqdm(range(0, len(datas), batch_size), desc="Encoding LoRA texts"):
        # 切片获取当前批次
        batch_datas = datas[i : i + batch_size]
        
        batch_texts = []
        batch_ids = []
        batch_model_files = []

        for data in batch_datas:
            model_id = data['model_id']
            llm_description = data['llm_description']
            title = data['title']
            tags = data['tags']
            model_file = data['model_file']

            query_text = (
                f"Convert Stable Diffusion finetuned adapter description into an embedding for search: "
                f"Title: {title}; Description: {llm_description}; Tag: {tags};"
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


# 2. 给训练集的prompt找候选lora
    
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




if __name__ == "__main__":
    augment_jsonl_with_lora_candidates(
        input_jsonl_path="rank_dataset/train_prompts_2.jsonl",
        output_jsonl_path="rank_dataset/train_prompt2_candidates.jsonl",
        prompt_index_path="rank_dataset/train_prompt_index_qwen_2.pt",
        lora_index_path="rank_dataset/qwenlora_desc_index_qwen.pt",
        device="cuda:0",
        top_k=10,
        semi_hard_k=5,
        random_k=0,
        semi_hard_range=(10, 50),
        candidate_field="candidate_loras",
    )