import os
from typing import List, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from sklearn.cluster import KMeans

from transformers import CLIPProcessor, CLIPModel
from torchvision import models, transforms


# ==========================================================
# 1. 基础工具
# ==========================================================
def load_image(image_path: str) -> Image.Image:
    return Image.open(image_path).convert("RGB")


def list_image_paths(image_dir: str, exts=(".png", ".jpg", ".jpeg", ".webp", ".bmp")) -> List[str]:
    image_paths = []
    for fname in os.listdir(image_dir):
        if fname.lower().endswith(exts):
            if 'rerank' in fname:
                p_id = int(fname.split('_')[1])
            else :
                p_id = int(fname.split('_')[0])
            # if p_id < 75:
            #     continue
            image_paths.append(os.path.join(image_dir, fname))
    image_paths.sort()
    return image_paths


def entropy_from_probs(probs: np.ndarray, eps: float = 1e-12) -> float:
    probs = probs.astype(np.float64)
    probs = probs[probs > eps]
    if len(probs) == 0:
        return 0.0
    return float(-(probs * np.log(probs + eps)).sum())


def truncate_distribution_topk(probs: np.ndarray, topk: Optional[int] = None) -> np.ndarray:
    """
    只保留最大的 top-k 概率项，再重新归一化
    """
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
    """
    保留累计概率达到 mass 的最大若干项，再重新归一化
    """
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


def cluster_histogram(labels: np.ndarray, num_clusters: int) -> np.ndarray:
    hist = np.bincount(labels, minlength=num_clusters).astype(np.float64)
    if hist.sum() > 0:
        hist /= hist.sum()
    return hist


# ==========================================================
# 2. CLIP 特征提取
# ==========================================================
class CLIPFeatureExtractor:
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str = "cuda",
    ):
        self.device = 'cuda'
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to('cuda')
        self.model.eval()

    @torch.no_grad()
    def extract_features(self, image_paths: List[str], batch_size: int = 32) -> np.ndarray:
        all_feats = []

        for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting CLIP features"):
            batch_paths = image_paths[i:i + batch_size]
            images = [load_image(p) for p in batch_paths]

            inputs = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)
            feats = self.model.get_image_features(**inputs)

            if hasattr(feats, "pooler_output"):
                feats = feats.pooler_output
            elif hasattr(feats, "last_hidden_state"):
                feats = feats.last_hidden_state[:, 0]
            elif isinstance(feats, tuple):
                feats = feats[0]

            feats = F.normalize(feats, dim=-1)

            all_feats.append(feats.cpu().numpy())

        return np.concatenate(all_feats, axis=0)


# ==========================================================
# 3. Inception 特征提取（兼容 torchvision 新版本）
# ==========================================================
class InceptionFeatureExtractor:
    def __init__(
        self,
        device: str = "cuda",
    ):
        self.device = "cuda"

        model = models.inception_v3(
            weights=models.Inception_V3_Weights.DEFAULT,
            aux_logits=True,
        ).to("cuda")
        model.fc = torch.nn.Identity()
        model.eval()
        self.model = model.to("cuda")

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

            # 兼容不同 torchvision 返回类型
            if isinstance(feats, tuple):
                feats = feats[0]
            elif hasattr(feats, "logits"):
                feats = feats.logits

            feats = F.normalize(feats, dim=-1)
            all_feats.append(feats.cpu().numpy())

        return np.concatenate(all_feats, axis=0)


# ==========================================================
# 4. 核心：从特征计算 truncated entropy
# ==========================================================
def compute_truncated_entropy_from_features(
    features: np.ndarray,
    num_clusters: int = 20,
    truncate_mode: str = "topk",   # "topk" or "mass"
    truncate_topk: int = 10,
    truncate_mass: float = 0.95,
    random_state: int = 42,
) -> Dict[str, float]:
    """
    流程：
    1. KMeans 聚类
    2. 统计 cluster histogram
    3. 截断
    4. 算 entropy
    """
    assert features.ndim == 2, "features should be [N, D]"

    n = features.shape[0]
    if n == 0:
        return {
            "entropy": 0.0,
            "raw_entropy": 0.0,
            "num_clusters": 0,
            "num_samples": 0,
        }

    num_clusters = min(num_clusters, n)

    kmeans = KMeans(
        n_clusters=num_clusters,
        random_state=random_state,
        n_init=10,
    )
    labels = kmeans.fit_predict(features)

    probs = cluster_histogram(labels, num_clusters)
    raw_entropy = entropy_from_probs(probs)

    if truncate_mode == "topk":
        trunc_probs = truncate_distribution_topk(probs, topk=truncate_topk)
    elif truncate_mode == "mass":
        trunc_probs = truncate_distribution_mass(probs, mass=truncate_mass)
    else:
        raise ValueError(f"Unknown truncate_mode: {truncate_mode}")

    trunc_entropy = entropy_from_probs(trunc_probs)

    return {
        "entropy": trunc_entropy,
        "raw_entropy": raw_entropy,
        "num_clusters": num_clusters,
        "num_samples": n,
    }


# ==========================================================
# 5. 分别计算 TCE / TIE
# ==========================================================
def compute_tce_from_folder(
    image_dir: str,
    clip_model_name: str = "openai/clip-vit-base-patch32",
    batch_size: int = 32,
    num_clusters: int = 20,
    truncate_mode: str = "topk",
    truncate_topk: int = 10,
    truncate_mass: float = 0.95,
    device: str = "cuda",
) -> Dict[str, float]:
    image_paths = list_image_paths(image_dir)

    extractor = CLIPFeatureExtractor(
        model_name=clip_model_name,
        device=device,
    )
    features = extractor.extract_features(image_paths, batch_size=batch_size)

    result = compute_truncated_entropy_from_features(
        features=features,
        num_clusters=num_clusters,
        truncate_mode=truncate_mode,
        truncate_topk=truncate_topk,
        truncate_mass=truncate_mass,
    )
    result["metric"] = "TCE"
    result["image_dir"] = image_dir
    return result


def compute_tie_from_folder(
    image_dir: str,
    batch_size: int = 32,
    num_clusters: int = 20,
    truncate_mode: str = "topk",
    truncate_topk: int = 10,
    truncate_mass: float = 0.95,
    device: str = "cuda",
) -> Dict[str, float]:
    image_paths = list_image_paths(image_dir)

    extractor = InceptionFeatureExtractor(device=device)
    features = extractor.extract_features(image_paths, batch_size=batch_size)

    result = compute_truncated_entropy_from_features(
        features=features,
        num_clusters=num_clusters,
        truncate_mode=truncate_mode,
        truncate_topk=truncate_topk,
        truncate_mass=truncate_mass,
    )
    result["metric"] = "TIE"
    result["image_dir"] = image_dir
    return result


def compute_tce_tie_from_folder(
    image_dir: str,
    clip_model_name: str = "openai/clip-vit-base-patch32",
    batch_size: int = 32,
    num_clusters: int = 20,
    truncate_mode: str = "topk",
    truncate_topk: int = 10,
    truncate_mass: float = 0.95,
    device: str = "cuda",
) -> Dict[str, Dict[str, float]]:
    tce = compute_tce_from_folder(
        image_dir=image_dir,
        clip_model_name=clip_model_name,
        batch_size=batch_size,
        num_clusters=num_clusters,
        truncate_mode=truncate_mode,
        truncate_topk=truncate_topk,
        truncate_mass=truncate_mass,
        device=device,
    )

    tie = compute_tie_from_folder(
        image_dir=image_dir,
        batch_size=batch_size,
        num_clusters=num_clusters,
        truncate_mode=truncate_mode,
        truncate_topk=truncate_topk,
        truncate_mass=truncate_mass,
        device=device,
    )

    return {
        "TCE": tce,
        "TIE": tie,
    }


# ==========================================================
# 6. Example
# ==========================================================
if __name__ == "__main__":
    image_dir = "/shark/zhiwen/LoRAHunter/sd_encoder/outputs/250_clipemb-7_all_res_combinations_reank"
    image_dir = "/shark/zhiwen/LoRAHunter/sd_encoder/outputs/sdv15_250_clipemb-7_all_res_combinations_reank"
    image_dir = "/shark/zhiwen/LoRAHunter/sd_encoder/outputs2/sdv15_retrieval_testdata_500/realistic"
    # image_dir = "/shark/zhiwen/LoRAHunter/sd_encoder/outputs2/stylus_testdata_500_totalpool_stylus/realistic"
    # image_dir = "/shark/zhiwen/LoRAHunter/sd_encoder/outputs2/hunter_data_500_qwenemb_4-28_reank_beam/realistic"
    results = compute_tce_tie_from_folder(
        image_dir=image_dir,
        clip_model_name="/shark/zhiwen/LoRAHunter/DiffSynth-Studio/models/AI-ModelScope/clip-vit-large-patch14",
        batch_size=32,
        num_clusters=30,
        truncate_mode="topk",   # "topk" or "mass"
        truncate_topk=15,
        truncate_mass=0.95,
        device="cuda",
    )

    print("\n===== Folder-level TCE / TIE =====")
    print("TCE:", results["TCE"])
    print("TIE:", results["TIE"])
