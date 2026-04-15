import os,json
import torch
from train import CombineEncoder
from diffsynth.core import load_state_dict
from DiffimageEmb import EmbSaver
from encoder import TextImageEncoder
from LoRAEncoder import LoRAEncoder


import random
from collections import defaultdict

import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm


def sample_datas_by_tag(datas, max_per_tag=None, seed=42):
    """
    按 tag 采样，每个 tag 最多保留 max_per_tag 个样本。
    如果 max_per_tag=None，则返回原始 datas。
    """
    if max_per_tag is None:
        return datas

    random.seed(seed)
    tag2datas = defaultdict(list)
    for data in datas:
        tag2datas[data['t_tag']].append(data)

    sampled_datas = []
    for tag, items in tag2datas.items():
        if len(items) <= max_per_tag:
            sampled_datas.extend(items)
        else:
            sampled_datas.extend(random.sample(items, max_per_tag))

    return sampled_datas




def encode_lora_embeddings(
    datas,
    combine_encoder,
    clip_encoder,
    base_path,
    emb_saver,
    device="cuda",
    dtype=torch.float,
    max_samples=None,
    max_per_tag=None,
):
    if max_per_tag is not None:
        datas = sample_datas_by_tag(datas, max_per_tag=max_per_tag)

    if max_samples is not None:
        datas = datas[:max_samples]

    all_embs = []
    all_tags = []
    all_ids = []

    # combine_encoder.eval()
    # clip_encoder.eval()

    with torch.no_grad():
        for data in tqdm(datas, desc="Encoding LoRA embeddings"):
            try:
                text = data['short_description']
                model_id = data['model_id']
                tag = data['t_tag']
                lora_path = f'{base_path}/{data["model_file"]}'

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
                    lora_emb = txt_emb + diff_vec


                lora_emb = lora_emb.squeeze(0).detach().cpu()             # [768]


                all_embs.append(lora_emb)
                all_tags.append(tag)
                all_ids.append(model_id)

            except Exception as e:
                print(f"Skip sample {data.get('model_id', 'unknown')}, error: {e}")

    if len(all_embs) == 0:
        raise ValueError("No valid embeddings were generated.")

    all_embs = torch.stack(all_embs, dim=0)
    return all_embs, all_tags, all_ids



def compute_cosine_similarity_matrix(all_embs):
    """
    输入:
        all_embs: torch.Tensor [N, D]
    返回:
        sim_matrix: np.ndarray [N, N]
    """
    all_embs = F.normalize(all_embs, p=2, dim=1)
    sim_matrix = all_embs @ all_embs.T
    return sim_matrix.numpy()


def plot_similarity_heatmap(
    sim_matrix,
    tags,
    model_ids=None,
    save_path="lora_emb_similarity_heatmap.png",
    title="LoRA Embedding Cosine Similarity Heatmap",
    figsize=(14, 12),
    show_boundary=True,
    label_type="none",   # "none", "tag", "model_id"
    label_fontsize=4,
    cmap="Reds",
):
    """
    按 tag 排序后绘制相似度热力图。

    Args:
        sim_matrix: [N, N]
        tags: list[str]
        model_ids: list[str] or None
        label_type:
            - "none": 不显示标签
            - "tag": 显示 tag
            - "model_id": 显示 model_id
    """
    sorted_indices = np.argsort(tags)
    sorted_sim_matrix = sim_matrix[sorted_indices][:, sorted_indices]
    sorted_tags = [tags[i] for i in sorted_indices]
    sorted_model_ids = [model_ids[i] for i in sorted_indices] if model_ids is not None else None

    plt.figure(figsize=figsize)
    plt.imshow(sorted_sim_matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    plt.colorbar(label="Cosine Similarity")

    if show_boundary and len(sorted_tags) > 0:
        boundaries = []
        last_tag = sorted_tags[0]
        for i, tag in enumerate(sorted_tags):
            if tag != last_tag:
                boundaries.append(i)
                last_tag = tag

        for b in boundaries:
            plt.axhline(b - 0.5, color='black', linewidth=0.5)
            plt.axvline(b - 0.5, color='black', linewidth=0.5)

    if label_type == "tag":
        positions = np.arange(len(sorted_tags))
        plt.xticks(positions, sorted_tags, rotation=90, fontsize=label_fontsize)
        plt.yticks(positions, sorted_tags, fontsize=label_fontsize)

    elif label_type == "model_id":
        if sorted_model_ids is None:
            raise ValueError("label_type='model_id' requires model_ids.")
        positions = np.arange(len(sorted_model_ids))
        plt.xticks(positions, sorted_model_ids, rotation=90, fontsize=label_fontsize)
        plt.yticks(positions, sorted_model_ids, fontsize=label_fontsize)

    else:
        plt.xticks([])
        plt.yticks([])

    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)


def evaluate_lora_encoder_heatmap(
    dataset_path,
    combine_encoder,
    clip_encoder,
    base_path,
    emb_saver,
    device="cuda",
    dtype=torch.float,
    max_samples=None,
    max_per_tag=None,
    heatmap_path="lora_emb_similarity_heatmap.png",
    label_type="none",   # "none", "tag", "model_id"
    label_fontsize=4,
):
    with open(dataset_path, 'r') as f:
        datas = [json.loads(line) for line in f.readlines()]

    all_embs, all_tags, all_ids = encode_lora_embeddings(
        datas=datas,
        combine_encoder=combine_encoder,
        clip_encoder=clip_encoder,
        base_path=base_path,
        emb_saver=emb_saver,
        device=device,
        dtype=dtype,
        max_samples=max_samples,
        max_per_tag=max_per_tag,
    )

    print("Embedding shape:", all_embs.shape)

    sim_matrix = compute_cosine_similarity_matrix(all_embs)

    plot_similarity_heatmap(
        sim_matrix=sim_matrix,
        tags=all_tags,
        model_ids=all_ids,
        save_path=heatmap_path,
        figsize=(14, 12),
        label_type=label_type,
        label_fontsize=label_fontsize,
        cmap="Reds",
    )

    return sim_matrix, all_tags, all_ids

def main_0():
    dataset_path = '/shark/zhiwen/LoRAHunter/train_mini_datas.jsonl'
    base_path='/shark/zhiwen/LoRAHunter/Qwen_LoRA'

    with open(dataset_path, 'r') as f:
        datas = [json.loads(line) for line in f.readlines()]

    # load lora encoder
    dtype=torch.float
    device="cuda"
    encoder_path = '/shark/zhiwen/LoRAHunter/DiffSynth-Studio/models/combine_encoder/train_mini_datas/2/combine_encoder-29.safetensors'
    combine_encoder = CombineEncoder(L=1).to(dtype=dtype)
    c_encoder_dict = load_state_dict(encoder_path, torch_dtype=dtype, device=device)
    combine_encoder.load_state_dict(c_encoder_dict)
    combine_encoder = combine_encoder.to(device)

    # load clip 
    clip_encoder = TextImageEncoder().to(dtype=dtype, device=device)

    for data in datas[:1]:
        text = data['short_description']
        model_id = data['model_id']
        diff_vec = EmbSaver().load_vec(model_id) #[1, 768]
        diff_vec = diff_vec.to(device)
        tag = data['t_tag']
        lora_path = f'{base_path}/{data["model_file"]}'

        txt_emb = clip_encoder.encoding_text(text) # [1, 768]
        lora_emb = combine_encoder(lora_path, txt_emb, diff_vec) #[1, 768]
        print(lora_emb.shape)



def main():
    dataset_path = '/shark/zhiwen/LoRAHunter/train_mini_datas.jsonl'
    base_path='/shark/zhiwen/LoRAHunter/Qwen_LoRA'

    # with open(dataset_path, 'r') as f:
    #     datas = [json.loads(line) for line in f.readlines()]

    # load lora encoder
    dtype=torch.float
    device="cuda:4"
    encoder_path = '/shark/zhiwen/LoRAHunter/DiffSynth-Studio/models/combine_encoder/train_mini_datas/2/combine_encoder-29.safetensors'
    # encoder_path = "/shark/zhiwen/LoRAHunter/DiffSynth-Studio/models/lora_encoder/train_mini_datas/1/lora_encoder-29.safetensors"
    # encoder_path = '/shark/zhiwen/LoRAHunter/DiffSynth-Studio/models/lora_encoder_prompt/train_mini_datas/1/lora_encoder-29.safetensors'
    
    c_encoder_dict = load_state_dict(encoder_path, torch_dtype=dtype, device=device)

    only_loraencoder = True
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

    # combine_encoder = None

    emb_saver = EmbSaver()

    # load clip 
    clip_encoder = TextImageEncoder().to(dtype=dtype, device=device)

    sim_matrix, tags, ids = evaluate_lora_encoder_heatmap(
        dataset_path=dataset_path,
        combine_encoder=combine_encoder,
        clip_encoder=clip_encoder,
        base_path=base_path,
        emb_saver=emb_saver,
        device=device,
        dtype=dtype,
        max_samples=200,   # 可选
        max_per_tag=20,    # 可选
        heatmap_path="zheatmap/lora_emb_similarity_heatmap_com_loraencoder.png",
        label_type="tag",
        label_fontsize=4,
    )



if __name__ == "__main__":
    main()