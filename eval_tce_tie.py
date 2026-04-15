import os
from collections import defaultdict
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from transformers import CLIPProcessor, CLIPModel
from torchvision import models, transforms


# ==========================================================
# 1. 基础工具
# ==========================================================
def load_image(image_path: str) -> Image.Image:
    ori_image_name = image_path.split("/")[-1]
    if 'sdv15' not in image_path:
        new_name = 'rerank_'+ori_image_name
        image_path = image_path.replace(ori_image_name, new_name)
    return Image.open(image_path).convert("RGB")


def entropy_from_probs(probs: np.ndarray, eps: float = 1e-12) -> float:
    probs = probs.astype(np.float64)
    probs = probs[probs > eps]
    if len(probs) == 0:
        return 0.0
    return float(-(probs * np.log(probs + eps)).sum())


def truncate_distribution_topk(probs: np.ndarray, topk: int = None) -> np.ndarray:
    probs = probs.astype(np.float64)

    if topk is None or topk >= len(probs):
        s = probs.sum()
        return probs / s if s > 0 else probs

    idx = np.argsort(probs)[::-1][:topk]
    truncated = np.zeros_like(probs)
    truncated[idx] = probs[idx]

    s = truncated.sum()
    if s > 0:
        truncated = truncated / s
    return truncated


def truncate_distribution_mass(probs: np.ndarray, mass: float = 0.95) -> np.ndarray:
    probs = probs.astype(np.float64)

    if mass >= 1.0:
        s = probs.sum()
        return probs / s if s > 0 else probs

    sorted_idx = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_idx]
    cumsum = np.cumsum(sorted_probs)
    keep_num = np.searchsorted(cumsum, mass, side="left") + 1

    keep_idx = sorted_idx[:keep_num]
    truncated = np.zeros_like(probs)
    truncated[keep_idx] = probs[keep_idx]

    s = truncated.sum()
    if s > 0:
        truncated = truncated / s
    return truncated


def group_images_by_prompt_id(image_dir: str) -> Dict[str, List[str]]:
    """
    按文件名中的 prompt_id 分组
    文件名格式假设为：
        {prompt_id}_{img_id}.png
    例如：
        0_0.png
        0_1.png
        1_0.png
    """
    groups = defaultdict(list)

    for fname in os.listdir(image_dir):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp")):
            continue
        fname = fname.replace("rerank_", "")
        stem = os.path.splitext(fname)[0]
        parts = stem.split("_")
        if len(parts) < 2:
            continue

        prompt_id = parts[0]
        if int(prompt_id) < 75:
            continue
        full_path = os.path.join(image_dir, fname)
        groups[prompt_id].append(full_path)

    # 排序，保证稳定
    for k in groups:
        groups[k] = sorted(groups[k])

    return dict(groups)


# ==========================================================
# 2. 特征提取器
# ==========================================================
class CLIPFeatureExtractor:
    def __init__(
        self,
        model_name: str = "/shark/zhiwen/LoRAHunter/DiffSynth-Studio/models/AI-ModelScope/clip-vit-large-patch14",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = "cuda"
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def extract_features(self, image_paths: List[str], batch_size: int = 32) -> np.ndarray:
        all_feats = []

        for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting CLIP features"):
            batch_paths = image_paths[i:i + batch_size]
            images = [load_image(p) for p in batch_paths]

            inputs = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)
            feats = self.model.get_image_features(**inputs)

            # 兼容不同返回类型
            if hasattr(feats, "pooler_output"):
                feats = feats.pooler_output
            elif hasattr(feats, "last_hidden_state"):
                feats = feats.last_hidden_state[:, 0]
            elif isinstance(feats, tuple):
                feats = feats[0]

            feats = F.normalize(feats, dim=-1)

            all_feats.append(feats.cpu().numpy())

        return np.concatenate(all_feats, axis=0)


class InceptionFeatureExtractor:
    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = "cuda"

        model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT, aux_logits=True)
        model.fc = torch.nn.Identity()
        model.eval()
        self.model = model.to(self.device)

        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    @torch.no_grad()
    def extract_features(self, image_paths: List[str], batch_size: int = 32) -> np.ndarray:
        all_feats = []

        for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting Inception features"):
            batch_paths = image_paths[i:i + batch_size]
            images = [self.transform(load_image(p)) for p in batch_paths]
            images = torch.stack(images, dim=0).to(self.device)

            feats = self.model(images)
            feats = F.normalize(feats, dim=-1)

            all_feats.append(feats.cpu().numpy())

        return np.concatenate(all_feats, axis=0)


# ==========================================================
# 3. 对“单个 prompt 的多张图”计算 truncated entropy
# ==========================================================
def compute_group_distribution_from_similarity(features: np.ndarray, temperature: float = 0.1) -> np.ndarray:
    """
    给定一个 prompt 下的 N 张图特征，构造一个基于“独特性”的分布。

    思路：
    1. 算 cosine similarity matrix
    2. 对每张图，计算它和其它图的平均相似度
    3. 独特性 = 1 - avg_similarity（做一个平移保证为正）
    4. 用 softmax / 归一化得到概率分布

    如果某张图和其它图都很像，它的独特性较低；
    如果某张图更独特，它会分到更高概率。
    """
    if features.shape[0] == 1:
        return np.array([1.0], dtype=np.float64)

    feats = torch.from_numpy(features).float()   # [N, D]
    feats = F.normalize(feats, dim=-1)

    sim = feats @ feats.t()   # [N, N]

    n = sim.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool)
    sim_others = sim[mask].view(n, n - 1)   # 每张图与其它图的相似度

    avg_sim = sim_others.mean(dim=1)        # [N]

    # 独特性：相似度越低，独特性越高
    uniqueness = 1.0 - avg_sim

    # 为避免负数/极小值导致问题，做 softmax
    probs = torch.softmax(uniqueness / temperature, dim=0)
    return probs.cpu().numpy().astype(np.float64)


def compute_truncated_entropy_for_group(
    features: np.ndarray,
    truncate_mode: str = "topk",
    truncate_topk: int = None,
    truncate_mass: float = 0.95,
    temperature: float = 0.1,
) -> Dict[str, float]:
    """
    对一个 prompt 的多张图，基于 feature similarity 构造分布，再算截断熵。
    """
    probs = compute_group_distribution_from_similarity(features, temperature=temperature)
    raw_entropy = entropy_from_probs(probs)

    if truncate_mode == "topk":
        trunc_probs = truncate_distribution_topk(probs, truncate_topk)
    elif truncate_mode == "mass":
        trunc_probs = truncate_distribution_mass(probs, truncate_mass)
    else:
        raise ValueError(f"Unknown truncate_mode: {truncate_mode}")

    trunc_entropy = entropy_from_probs(trunc_probs)

    return {
        "raw_entropy": raw_entropy,
        "entropy": trunc_entropy,
        "num_images": int(features.shape[0]),
    }


# ==========================================================
# 4. 按 prompt 分组计算 TCE / TIE
# ==========================================================
def compute_grouped_entropy_metric(
    grouped_image_paths: Dict[str, List[str]],
    extractor,
    batch_size: int = 32,
    truncate_mode: str = "topk",
    truncate_topk: int = None,
    truncate_mass: float = 0.95,
    temperature: float = 0.1,
    metric_name: str = "TCE",
) -> Dict:
    """
    对每个 prompt 单独算 entropy，再求平均。
    """
    # 先把所有图合并提特征，避免重复跑模型
    all_image_paths = []
    image_to_prompt = []
    for prompt_id, paths in grouped_image_paths.items():
        for p in paths:
            all_image_paths.append(p)
            image_to_prompt.append(prompt_id)

    all_features = extractor.extract_features(all_image_paths, batch_size=batch_size)

    # 按 prompt 收集 feature
    prompt_to_features = defaultdict(list)
    for feat, prompt_id in zip(all_features, image_to_prompt):
        prompt_to_features[prompt_id].append(feat)

    per_prompt_results = {}
    all_scores = []
    all_raw_scores = []

    for prompt_id, feats in prompt_to_features.items():
        feats = np.stack(feats, axis=0)

        res = compute_truncated_entropy_for_group(
            features=feats,
            truncate_mode=truncate_mode,
            truncate_topk=truncate_topk,
            truncate_mass=truncate_mass,
            temperature=temperature,
        )

        per_prompt_results[prompt_id] = res
        all_scores.append(res["entropy"])
        all_raw_scores.append(res["raw_entropy"])

    summary = {
        "metric": metric_name,
        "num_prompts": len(per_prompt_results),
        "mean_entropy": float(np.mean(all_scores)) if len(all_scores) > 0 else 0.0,
        "std_entropy": float(np.std(all_scores)) if len(all_scores) > 0 else 0.0,
        "mean_raw_entropy": float(np.mean(all_raw_scores)) if len(all_raw_scores) > 0 else 0.0,
        "per_prompt": per_prompt_results,
    }
    return summary


def compute_grouped_tce_tie(
    image_dir: str,
    clip_model_name: str = "/shark/zhiwen/LoRAHunter/DiffSynth-Studio/models/AI-ModelScope/clip-vit-large-patch14",
    batch_size: int = 32,
    truncate_mode: str = "topk",
    truncate_topk: int = None,
    truncate_mass: float = 0.95,
    temperature: float = 0.1,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict:
    """
    按 prompt 分组，分别计算：
    - TCE: CLIP feature space truncated entropy
    - TIE: Inception feature space truncated entropy
    """
    grouped = group_images_by_prompt_id(image_dir)

    clip_extractor = CLIPFeatureExtractor(
        model_name=clip_model_name,
        device=device,
    )
    inception_extractor = InceptionFeatureExtractor(
        device=device,
    )

    tce = compute_grouped_entropy_metric(
        grouped_image_paths=grouped,
        extractor=clip_extractor,
        batch_size=batch_size,
        truncate_mode=truncate_mode,
        truncate_topk=truncate_topk,
        truncate_mass=truncate_mass,
        temperature=temperature,
        metric_name="TCE",
    )

    tie = compute_grouped_entropy_metric(
        grouped_image_paths=grouped,
        extractor=inception_extractor,
        batch_size=batch_size,
        truncate_mode=truncate_mode,
        truncate_topk=truncate_topk,
        truncate_mass=truncate_mass,
        temperature=temperature,
        metric_name="TIE",
    )

    return {
        "TCE": tce,
        "TIE": tie,
    }


# ==========================================================
# 5. 示例
# ==========================================================
if __name__ == "__main__":
    image_dir = "/shark/zhiwen/LoRAHunter/sd_encoder/outputs/250_clipemb-7_all_res_combinations_reank"
    image_dir = "/shark/zhiwen/LoRAHunter/sd_encoder/outputs/sdv15_250_clipemb-7_all_res_combinations_reank"

    results = compute_grouped_tce_tie(
        image_dir=image_dir,
        clip_model_name="/shark/zhiwen/LoRAHunter/DiffSynth-Studio/models/AI-ModelScope/clip-vit-large-patch14",
        batch_size=32,
        truncate_mode="topk",     # "topk" or "mass"
        truncate_topk=3,          # 每个 prompt 只有 5 张图，这里设 3 比较合理
        truncate_mass=0.95,
        temperature=0.1,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    print("\n===== Grouped TCE/TIE Results =====")
    print("TCE mean:", results["TCE"]["mean_entropy"])
    print("TCE std :", results["TCE"]["std_entropy"])
    print("TIE mean:", results["TIE"]["mean_entropy"])
    print("TIE std :", results["TIE"]["std_entropy"])

    # 看某个 prompt 的结果
    if "0" in results["TCE"]["per_prompt"]:
        print("\nPrompt 0 TCE:", results["TCE"]["per_prompt"]["0"])
    if "0" in results["TIE"]["per_prompt"]:
        print("Prompt 0 TIE:", results["TIE"]["per_prompt"]["0"])
