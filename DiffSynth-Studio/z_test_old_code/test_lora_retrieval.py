
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import os,json
import torch
from train import CombineEncoder
from diffsynth.core import load_state_dict
from DiffimageEmb import EmbSaver
from encoder import TextImageEncoder
from LoRAEncoder import LoRAEncoder

def build_lora_database(
    datas,
    combine_encoder,
    clip_encoder,
    base_path,
    emb_saver,
    device="cuda",
    dtype=torch.float,
    lora_text_key="short_description",
    tag_key="t_tag",
):
    """
    构建 LoRA 数据库 embedding，并保留 model_id 和 tag。
    """
    model_ids = []
    lora_tags = []
    lora_embs = []

    # combine_encoder.eval()
    clip_encoder.eval()

    with torch.no_grad():
        for data in tqdm(datas, desc="Building LoRA database"):
            try:
                model_id = data["model_id"]
                tag = data[tag_key]
                lora_path = f'{base_path}/{data["model_file"]}'
                text = data[lora_text_key]

                if isinstance(combine_encoder, CombineEncoder):
                    diff_vec = emb_saver.load_vec(model_id).to(device=device, dtype=dtype)   # [1, 768]
                    txt_emb = clip_encoder.encoding_text(text).to(device=device, dtype=dtype) # [1, 768]
                    lora_emb = combine_encoder(lora_path, txt_emb, diff_vec)  # [1, 768]
                elif isinstance(combine_encoder, LoRAEncoder):
                    lora = load_state_dict(lora_path, torch_dtype=dtype, device=device)
                    lora_emb = combine_encoder(lora)
                else:
                    diff_vec = emb_saver.load_vec(model_id).to(device=device, dtype=dtype)
                    txt_emb = clip_encoder.encoding_text(text).to(device=device, dtype=dtype)
                    lora_emb = txt_emb

                lora_emb = lora_emb.squeeze(0).detach().cpu()

                model_ids.append(model_id)
                lora_tags.append(tag)
                lora_embs.append(lora_emb)

            except Exception as e:
                print(f"Skip model {data.get('model_id', 'unknown')}, error: {e}")

    if len(lora_embs) == 0:
        raise ValueError("No valid LoRA embeddings were generated.")

    lora_embs = torch.stack(lora_embs, dim=0)
    lora_embs = F.normalize(lora_embs, p=2, dim=1)

    model_id_to_index = {mid: i for i, mid in enumerate(model_ids)}
    return lora_embs, model_ids, lora_tags, model_id_to_index


def flatten_prompt_queries(datas, prompt_key="prompts", tag_key="t_tag"):
    """
    展开 prompt query，同时保留 gt tag。
    """
    queries = []
    for data in datas:
        model_id = data["model_id"]
        tag = data[tag_key]
        prompts = data[prompt_key]

        for prompt in prompts[-1:]:
            queries.append({
                "model_id": model_id,
                "tag": tag,
                "prompt": prompt
            })
    return queries


def evaluate_text_to_lora_retrieval_with_tag(
    queries,
    lora_embs,
    model_ids,
    lora_tags,
    model_id_to_index,
    clip_encoder,
    device="cuda",
    dtype=torch.float,
    topk_list=(1, 5, 10),
):
    """
    同时评估：
    1. model_id 级检索
    2. tag 级检索
    """
    clip_encoder.eval()

    # model-level
    hits = {k: 0 for k in topk_list}
    ranks = []

    # tag-level
    tag_hits = {k: 0 for k in topk_list}
    tag_ranks = []
    top1_tag_correct = 0

    results = []

    with torch.no_grad():
        for query in tqdm(queries, desc="Evaluating retrieval"):
            try:
                gt_model_id = query["model_id"]
                gt_tag = query["tag"]
                prompt = query["prompt"]

                if gt_model_id not in model_id_to_index:
                    continue

                text_emb = clip_encoder.encoding_text(prompt).to(device=device, dtype=dtype)
                text_emb = text_emb.squeeze(0).detach().cpu()
                text_emb = F.normalize(text_emb, p=2, dim=0)

                sims = torch.matmul(lora_embs, text_emb)   # [N]
                sorted_indices = torch.argsort(sims, descending=True)

                # ===== model-level rank =====
                gt_index = model_id_to_index[gt_model_id]
                rank = (sorted_indices == gt_index).nonzero(as_tuple=True)[0].item() + 1
                ranks.append(rank)

                for k in topk_list:
                    if rank <= k:
                        hits[k] += 1

                # ===== tag-level rank =====
                sorted_tags = [lora_tags[i] for i in sorted_indices.tolist()]

                first_tag_rank = None
                for idx, pred_tag in enumerate(sorted_tags):
                    if pred_tag == gt_tag:
                        first_tag_rank = idx + 1
                        break

                if first_tag_rank is not None:
                    tag_ranks.append(first_tag_rank)

                for k in topk_list:
                    topk_tags = sorted_tags[:k]
                    if gt_tag in topk_tags:
                        tag_hits[k] += 1

                if len(sorted_tags) > 0 and sorted_tags[0] == gt_tag:
                    top1_tag_correct += 1

                # 保存详细结果
                topk_max = max(topk_list)
                topk_indices = sorted_indices[:topk_max].tolist()
                topk_model_ids = [model_ids[i] for i in topk_indices]
                topk_tags = [lora_tags[i] for i in topk_indices]
                topk_scores = [float(sims[i]) for i in topk_indices]

                results.append({
                    "prompt": prompt,
                    "gt_model_id": gt_model_id,
                    "gt_tag": gt_tag,
                    "rank": rank,
                    "tag_rank": first_tag_rank,
                    "topk_model_ids": topk_model_ids,
                    "topk_tags": topk_tags,
                    "topk_scores": topk_scores,
                })

            except Exception as e:
                print(f"Skip query, error: {e}")

    num_queries = len(ranks)
    metrics = {}

    # ===== model-level metrics =====
    for k in topk_list:
        metrics[f"Recall@{k}"] = hits[k] / num_queries if num_queries > 0 else 0.0

    metrics["MeanRank"] = float(np.mean(ranks)) if num_queries > 0 else None
    metrics["MedianRank"] = float(np.median(ranks)) if num_queries > 0 else None
    metrics["MRR"] = float(np.mean([1.0 / r for r in ranks])) if num_queries > 0 else None

    # ===== tag-level metrics =====
    for k in topk_list:
        metrics[f"TagRecall@{k}"] = tag_hits[k] / num_queries if num_queries > 0 else 0.0

    metrics["TagTop1Accuracy"] = top1_tag_correct / num_queries if num_queries > 0 else 0.0
    metrics["MeanTagRank"] = float(np.mean(tag_ranks)) if len(tag_ranks) > 0 else None
    metrics["MedianTagRank"] = float(np.median(tag_ranks)) if len(tag_ranks) > 0 else None

    metrics["NumQueries"] = num_queries
    metrics["NumLoras"] = len(model_ids)

    return metrics, results


def evaluate_lora_retrieval_with_tag(
    dataset_path,
    combine_encoder,
    clip_encoder,
    base_path,
    emb_saver,
    device="cuda",
    dtype=torch.float,
    lora_text_key="short_description",
    prompt_key="prompts",
    tag_key="t_tag",
    topk_list=(1, 5, 10),
    save_result_path=None,
):
    with open(dataset_path, "r") as f:
        datas = [json.loads(line) for line in f.readlines()]

    lora_embs, model_ids, lora_tags, model_id_to_index = build_lora_database(
        datas=datas,
        combine_encoder=combine_encoder,
        clip_encoder=clip_encoder,
        base_path=base_path,
        emb_saver=emb_saver,
        device=device,
        dtype=dtype,
        lora_text_key=lora_text_key,
        tag_key=tag_key,
    )

    queries = flatten_prompt_queries(
        datas,
        prompt_key=prompt_key,
        tag_key=tag_key,
    )

    metrics, results = evaluate_text_to_lora_retrieval_with_tag(
        queries=queries,
        lora_embs=lora_embs,
        model_ids=model_ids,
        lora_tags=lora_tags,
        model_id_to_index=model_id_to_index,
        clip_encoder=clip_encoder,
        device=device,
        dtype=dtype,
        topk_list=topk_list,
    )

    print("Retrieval Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    if save_result_path is not None:
        with open(save_result_path, "w") as f:
            json.dump(
                {
                    "metrics": metrics,
                    "results": results,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"Saved results to: {save_result_path}")

    return metrics, results


def build_lora_database_for_inference(
    datas,
    combine_encoder,
    clip_encoder,
    base_path,
    emb_saver,
    device="cuda",
    dtype=torch.float,
    lora_text_key="short_description",
    tag_key="t_tag",
):
    """
    构建用于检索推理的 LoRA 数据库。

    Returns:
        db: dict
            {
                "lora_embs": Tensor [N, D], 归一化后
                "model_ids": list[str],
                "model_files": list[str],
                "tags": list[str],
            }
    """
    model_ids = []
    model_files = []
    tags = []
    lora_embs = []

    # combine_encoder.eval()
    clip_encoder.eval()

    with torch.no_grad():
        for data in tqdm(datas, desc="Building LoRA database"):
            try:
                model_id = data["model_id"]
                model_file = data["model_file"]
                tag = data.get(tag_key, None)
                text = data[lora_text_key]
                lora_path = f'{base_path}/{model_file}'

                if isinstance(combine_encoder, CombineEncoder):
                    diff_vec = emb_saver.load_vec(model_id).to(device=device, dtype=dtype)   # [1, 768]
                    txt_emb = clip_encoder.encoding_text(text).to(device=device, dtype=dtype) # [1, 768]
                    lora_emb = combine_encoder(lora_path, txt_emb, diff_vec)  # [1, 768]
                elif isinstance(combine_encoder, LoRAEncoder):
                    lora = load_state_dict(lora_path, torch_dtype=dtype, device=device)
                    lora_emb = combine_encoder(lora)
                else:
                    diff_vec = emb_saver.load_vec(model_id).to(device=device, dtype=dtype)
                    txt_emb = clip_encoder.encoding_text(text).to(device=device, dtype=dtype)
                    lora_emb = txt_emb

                lora_emb = lora_emb.squeeze(0).detach().cpu()

                model_ids.append(model_id)
                model_files.append(model_file)
                tags.append(tag)
                lora_embs.append(lora_emb)

            except Exception as e:
                print(f"Skip model {data.get('model_id', 'unknown')}, error: {e}")

    if len(lora_embs) == 0:
        raise ValueError("No valid LoRA embeddings were generated.")

    lora_embs = torch.stack(lora_embs, dim=0)   # [N, D]
    lora_embs = F.normalize(lora_embs, p=2, dim=1)

    db = {
        "lora_embs": lora_embs,
        "model_ids": model_ids,
        "model_files": model_files,
        "tags": tags,
    }
    return db

def retrieve_lora_by_prompt(
    prompt,
    db,
    clip_encoder,
    device="cuda",
    dtype=torch.float,
    topk=5,
):
    """
    输入单条 prompt，返回 topk 检索结果。

    Args:
        prompt: str
        db: build_lora_database_for_inference 返回的字典
        topk: int

    Returns:
        results: list[dict]
            [
                {
                    "rank": 1,
                    "model_id": ...,
                    "model_file": ...,
                    "tag": ...,
                    "score": ...
                },
                ...
            ]
    """
    clip_encoder.eval()

    lora_embs = db["lora_embs"]         # [N, D], cpu
    model_ids = db["model_ids"]
    model_files = db["model_files"]
    tags = db["tags"]

    with torch.no_grad():
        text_emb = clip_encoder.encoding_text(prompt).to(device=device, dtype=dtype)  # [1, D]
        text_emb = text_emb.squeeze(0).detach().cpu()                                  # [D]
        text_emb = F.normalize(text_emb, p=2, dim=0)

        sims = torch.matmul(lora_embs, text_emb)   # [N]
        sorted_indices = torch.argsort(sims, descending=True)

        topk = min(topk, len(model_ids))
        topk_indices = sorted_indices[:topk].tolist()

        results = []
        for rank, idx in enumerate(topk_indices, start=1):
            results.append({
                "rank": rank,
                "model_id": model_ids[idx],
                "model_file": model_files[idx],
                "tag": tags[idx],
                "score": float(sims[idx]),
            })

    return results


def get_or_build_lora_database(
    db_path,
    datas,
    combine_encoder,
    clip_encoder,
    base_path,
    emb_saver,
    device="cuda",
    dtype=torch.float,
    lora_text_key="short_description",
    tag_key="t_tag",
    force_rebuild=False,
):
    """
    如果 db_path 存在且不强制重建，则直接加载；
    否则构建并保存。
    """
    if os.path.exists(db_path) and not force_rebuild:
        db = load_lora_database(db_path)
    else:
        print("LoRA database cache not found, start building...")
        db = build_lora_database_for_inference(
            datas=datas,
            combine_encoder=combine_encoder,
            clip_encoder=clip_encoder,
            base_path=base_path,
            emb_saver=emb_saver,
            device=device,
            dtype=dtype,
            lora_text_key=lora_text_key,
            tag_key=tag_key,
        )
        save_lora_database(db, db_path)

    return db

def save_lora_database(db, save_path):
    """
    保存检索数据库
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
    torch.save(db, save_path)
    print(f"LoRA database saved to: {save_path}")


def load_lora_database(save_path):
    """
    加载检索数据库
    """
    db = torch.load(save_path, map_location="cpu")
    print(f"LoRA database loaded from: {save_path}")
    return db

if __name__ == "__main__":

    dataset_path = '/shark/zhiwen/LoRAHunter/train_mini_datas.jsonl'
    base_path='/shark/zhiwen/LoRAHunter/Qwen_LoRA'

    dtype = torch.float
    device = "cuda:0"

    # encoder_path = '/shark/zhiwen/LoRAHunter/DiffSynth-Studio/models/combine_encoder/train_mini_datas/2/combine_encoder-29.safetensors'
    encoder_path = '/shark/zhiwen/LoRAHunter/DiffSynth-Studio/models/combine_encoder_cls/train_mini_datas/0/combine_encoder-29.safetensors'
    # encoder_path = "/shark/zhiwen/LoRAHunter/DiffSynth-Studio/models/lora_encoder/train_mini_datas/1/lora_encoder-29.safetensors"
    # encoder_path = "/shark/zhiwen/LoRAHunter/DiffSynth-Studio/models/lora_encoder_prompt/train_mini_datas/1/lora_encoder-29.safetensors"
    c_encoder_dict = load_state_dict(encoder_path, torch_dtype=dtype, device=device)

    only_loraencoder = False
    if 'combine_encoder' in encoder_path:
        if only_loraencoder:
            combine_encoder = LoRAEncoder(L=1).to(dtype=dtype)
            c_encoder_dict  = {k.replace('lora_encoder.', ''):v for k,v in c_encoder_dict.items() if 'lora_encoder' in k}
        else:
            combine_encoder = CombineEncoder(L=1).to(dtype=dtype)
    elif 'lora_encoder' in encoder_path:
        combine_encoder = LoRAEncoder(L=1).to(dtype=dtype)
    else:
        combine_encoder = LoRAEncoder(L=1).to(dtype=dtype)
    
    
    
    combine_encoder.load_state_dict(c_encoder_dict)
    combine_encoder = combine_encoder.to(device)

    combine_encoder = None

    clip_encoder = TextImageEncoder().to(dtype=dtype, device=device)
    emb_saver = EmbSaver()

    # metrics, results = evaluate_lora_retrieval_with_tag(
    #     dataset_path=dataset_path,
    #     combine_encoder=combine_encoder,
    #     clip_encoder=clip_encoder,
    #     base_path=base_path,
    #     emb_saver=EmbSaver(),
    #     device=device,
    #     dtype=dtype,
    #     lora_text_key="short_description",
    #     prompt_key="prompts",
    #     tag_key="t_tag",
    #     topk_list=(1, 5, 10),
    #     save_result_path="retrieval_with_tag_results_lora_encoder_prompt.json",
    # )


    # 检索单个prompt

    with open(dataset_path, "r") as f:
        datas = [json.loads(line) for line in f.readlines()]

    db_path = "zcache/lora_encoder_txt.pt"

    db = get_or_build_lora_database(
        db_path=db_path,
        datas=datas,
        combine_encoder=combine_encoder,
        clip_encoder=clip_encoder,
        base_path=base_path,
        emb_saver=EmbSaver(),
        device=device,
        dtype=dtype,
        lora_text_key="short_description",
        tag_key="t_tag",
        force_rebuild=False,
    )

    prompt = "Neon Lights, Futuristic City"
    prompt = "watercolor art, sketch art"
    prompt = "watercolor art, Traditional Chinese Clothing"

    results = retrieve_lora_by_prompt(
        prompt=prompt,
        db=db,
        clip_encoder=clip_encoder,
        device=device,
        dtype=dtype,
        topk=10,
    )

    for item in results:
        print(item)


