import os
import json
from typing import List, Dict, Any

import torch
import torch.nn.functional as F
import numpy as np

from PIL import Image
from tqdm import tqdm
from torchvision import models, transforms
from transformers import AutoProcessor, AutoModel
from sklearn.cluster import KMeans


class DiversityEvaluator:
    def __init__(
        self,
        device="cuda",
        clip_model_name="/shark/zhiwen/LoRAHunter/rank_encoder/models/muse/openai-clip-vit-large-patch14",
        tce_num_prototypes=50,
        tce_top_k=10,
        tie_top_k=20,
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.tce_num_prototypes = tce_num_prototypes
        self.tce_top_k = tce_top_k
        self.tie_top_k = tie_top_k

        # -------------------------
        # CLIP image encoder
        # -------------------------
        self.clip_processor = AutoProcessor.from_pretrained(clip_model_name)
        self.clip_model = AutoModel.from_pretrained(clip_model_name).to(self.device)
        self.clip_model.eval()

        # -------------------------
        # InceptionV3
        # -------------------------
        self.inception_model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
        self.inception_model.eval().to(self.device)

        self.inception_transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        # TCE prototypes
        self.tce_prototypes = None

    # =====================================================
    # utils
    # =====================================================
    def load_jsonl(self, path: str) -> List[Dict[str, Any]]:
        datas = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                datas.append(json.loads(line))
        return datas

    def find_image_path(self, image_dir: str, iid, sample_idx: int):
        exts = ["png", "jpg", "jpeg", "webp", "bmp"]
        for ext in exts:
            p = os.path.join(image_dir, f"{iid}_{sample_idx}.{ext}")
            p2 = os.path.join(image_dir, f"rerank_{iid}_{sample_idx}.{ext}")
            if os.path.exists(p):
                return p
            if os.path.exists(p2):
                return p2
        return None

    def batch_load_images(self, image_paths: List[str]):
        images = []
        valid_paths = []
        for p in image_paths:
            try:
                img = Image.open(p).convert("RGB")
                images.append(img)
                valid_paths.append(p)
            except Exception as e:
                print(f"[Warn] failed to open image: {p}, err={e}")
        return images, valid_paths

    def _mean_or_none(self, xs):
        xs = [x for x in xs if x is not None]
        if len(xs) == 0:
            return None
        return sum(xs) / len(xs)

    def _truncate_probs(self, probs: torch.Tensor, top_k: int):
        top_k = min(top_k, probs.shape[-1])
        values, indices = torch.topk(probs, k=top_k, dim=-1)

        truncated = torch.zeros_like(probs)
        truncated.scatter_(dim=-1, index=indices, src=values)
        truncated = truncated / truncated.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return truncated

    def _entropy_from_probs(self, probs: torch.Tensor, eps=1e-12):
        probs = probs.clamp_min(eps)
        ent = -(probs * probs.log()).sum(dim=-1)
        return ent

    # =====================================================
    # CLIP image features
    # =====================================================
    @torch.no_grad()
    def get_clip_image_features_batch(self, images: List[Image.Image], batch_size=16):
        all_feats = []

        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            inputs = self.clip_processor(images=batch_images, return_tensors="pt").to(self.device)

            image_features = self.clip_model.get_image_features(pixel_values=inputs["pixel_values"])
            image_features = F.normalize(image_features, dim=-1)
            all_feats.append(image_features.detach().cpu())

        return torch.cat(all_feats, dim=0)

    # =====================================================
    # Inception probs
    # =====================================================
    @torch.no_grad()
    def get_inception_probs_batch(self, images: List[Image.Image], batch_size=16):
        all_probs = []

        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            batch_tensor = torch.stack([self.inception_transform(img) for img in batch_images], dim=0).to(self.device)

            logits = self.inception_model(batch_tensor)
            probs = F.softmax(logits, dim=-1)

            all_probs.append(probs.detach().cpu())

        return torch.cat(all_probs, dim=0)

    # =====================================================
    # TCE prototypes
    # =====================================================
    def fit_tce_prototypes(
        self,
        image_dir: str,
        jsonl_path: str,
        iid_key="iid",
        num_images_per_prompt=10,
        clip_batch_size=32,
        max_images_for_kmeans=5000,
    ):
        datas = self.load_jsonl(jsonl_path)

        all_paths = []
        for data in datas:
            iid = data[iid_key]
            for sample_idx in range(1, num_images_per_prompt + 1):
                p = self.find_image_path(image_dir, iid, sample_idx)
                if p is not None:
                    all_paths.append(p)

        if len(all_paths) == 0:
            raise ValueError("No images found for fitting TCE prototypes.")

        if len(all_paths) > max_images_for_kmeans:
            all_paths = all_paths[:max_images_for_kmeans]

        all_feats = []
        for i in tqdm(range(0, len(all_paths), clip_batch_size), desc="Fitting TCE prototypes"):
            batch_paths = all_paths[i:i + clip_batch_size]
            images, _ = self.batch_load_images(batch_paths)
            if len(images) == 0:
                continue
            feats = self.get_clip_image_features_batch(images, batch_size=clip_batch_size)
            all_feats.append(feats)

        if len(all_feats) == 0:
            raise ValueError("Failed to extract CLIP features for TCE prototypes.")

        all_feats = torch.cat(all_feats, dim=0).numpy()

        k = min(self.tce_num_prototypes, len(all_feats))
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(all_feats)

        centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
        centers = F.normalize(centers, dim=-1)

        self.tce_prototypes = centers
        print(f"[TCE] fitted prototypes: {self.tce_prototypes.shape}")

    # =====================================================
    # diversity metrics
    # =====================================================
    @torch.no_grad()
    def compute_tce(self, clip_image_feats: torch.Tensor):
        """
        Truncated CLIP Entropy
        """
        if self.tce_prototypes is None:
            raise ValueError("TCE prototypes not fitted.")

        feats = F.normalize(clip_image_feats.float(), dim=-1)
        protos = self.tce_prototypes.to(feats.device)

        sims = feats @ protos.T
        probs = F.softmax(sims, dim=-1)
        probs = self._truncate_probs(probs, top_k=self.tce_top_k)

        ent = self._entropy_from_probs(probs)
        return ent.mean().item()

    @torch.no_grad()
    def compute_tie(self, inception_probs: torch.Tensor):
        """
        Truncated Inception Entropy
        """
        probs = inception_probs.float()
        probs = self._truncate_probs(probs, top_k=self.tie_top_k)
        ent = self._entropy_from_probs(probs)
        return ent.mean().item()

    @torch.no_grad()
    def compute_i2i(self, clip_image_feats: torch.Tensor):
        """
        I2I: average pairwise CLIP image-image cosine similarity
        越低越好
        """
        feats = F.normalize(clip_image_feats.float(), dim=-1)
        n = feats.shape[0]

        if n <= 1:
            return None

        sim_matrix = feats @ feats.T  # [N, N]

        vals = []
        for i in range(n):
            for j in range(i + 1, n):
                vals.append(sim_matrix[i, j].item())

        if len(vals) == 0:
            return None

        return sum(vals) / len(vals)

    # =====================================================
    # main
    # =====================================================
    def evaluate_from_jsonl(
        self,
        image_dir: str,
        jsonl_path: str,
        output_jsonl: str,
        iid_key="iid",
        prompt_key="prompt",
        num_images_per_prompt=10,
        clip_batch_size=16,
        inception_batch_size=16,
        fit_tce_first=True,
        num_prompts=500,
    ):
        datas = self.load_jsonl(jsonl_path)
        datas = datas[:num_prompts]

        if fit_tce_first:
            self.fit_tce_prototypes(
                image_dir=image_dir,
                jsonl_path=jsonl_path,
                iid_key=iid_key,
                num_images_per_prompt=num_images_per_prompt,
                clip_batch_size=clip_batch_size,
            )

        global_tce = []
        global_tie = []
        global_i2i = []

        with open(output_jsonl, "w", encoding="utf-8") as fout:
            for data in tqdm(datas, desc="Evaluating diversity"):
                iid = data[iid_key]
                prompt = data.get(prompt_key, "")

                image_paths = []
                for sample_idx in range(0, num_images_per_prompt):
                    p = self.find_image_path(image_dir, iid, sample_idx)
                    if p is not None:
                        image_paths.append(p)
                    else:
                        print(f"[Warn] missing image: {iid}_{sample_idx}")

                if len(image_paths) == 0:
                    result = {
                        "iid": iid,
                        "prompt": prompt,
                        "num_valid_images": 0,
                        "tce": None,
                        "tie": None,
                        "i2i": None,
                        "images": [],
                    }
                    fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                    continue

                images, valid_paths = self.batch_load_images(image_paths)
                if len(images) == 0:
                    result = {
                        "iid": iid,
                        "prompt": prompt,
                        "num_valid_images": 0,
                        "tce": None,
                        "tie": None,
                        "i2i": None,
                        "images": [],
                    }
                    fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                    continue

                try:
                    clip_feats = self.get_clip_image_features_batch(images, batch_size=clip_batch_size)
                    tce_score = self.compute_tce(clip_feats)
                    i2i_score = self.compute_i2i(clip_feats)
                except Exception as e:
                    print(f"[Warn] TCE/I2I failed for iid={iid}, err={e}")
                    tce_score = None
                    i2i_score = None

                try:
                    inception_probs = self.get_inception_probs_batch(images, batch_size=inception_batch_size)
                    tie_score = self.compute_tie(inception_probs)
                except Exception as e:
                    print(f"[Warn] TIE failed for iid={iid}, err={e}")
                    tie_score = None

                if tce_score is not None:
                    global_tce.append(tce_score)
                if tie_score is not None:
                    global_tie.append(tie_score)
                if i2i_score is not None:
                    global_i2i.append(i2i_score)

                result = {
                    "iid": iid,
                    "prompt": prompt,
                    "num_valid_images": len(valid_paths),
                    "tce": tce_score,
                    "tie": tie_score,
                    "i2i": i2i_score,
                    "images": valid_paths,
                }

                fout.write(json.dumps(result, ensure_ascii=False) + "\n")

        summary = {
            "num_prompts": len(datas),
            "global_mean_tce": self._mean_or_none(global_tce),
            "global_mean_tie": self._mean_or_none(global_tie),
            "global_mean_i2i": self._mean_or_none(global_i2i),
        }

        summary_path = output_jsonl.replace(".jsonl", "_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"[Done] saved per-prompt diversity results to: {output_jsonl}")
        print(f"[Done] saved diversity summary to: {summary_path}")
        print(json.dumps(summary, ensure_ascii=False, indent=2))


def evaluate_diversity(
    image_dir,
    jsonl_path,
    output_jsonl,
    device="cuda",
    iid_key="iid",
    prompt_key="prompt",
    num_images_per_prompt=10,
    clip_batch_size=16,
    inception_batch_size=16,
    tce_num_prototypes=50,
    tce_top_k=10,
    tie_top_k=10,
    num_prompts=500,
):
    evaluator = DiversityEvaluator(
        device=device,
        tce_num_prototypes=tce_num_prototypes,
        tce_top_k=tce_top_k,
        tie_top_k=tie_top_k,
    )
    evaluator.evaluate_from_jsonl(
        image_dir=image_dir,
        jsonl_path=jsonl_path,
        output_jsonl=output_jsonl,
        iid_key=iid_key,
        prompt_key=prompt_key,
        num_images_per_prompt=num_images_per_prompt,
        clip_batch_size=clip_batch_size,
        inception_batch_size=inception_batch_size,
        fit_tce_first=True,
        num_prompts=num_prompts,
    )

if __name__ == "__main__":
    # base_path = "/shark/zhiwen/LoRAHunter/sd_encoder/outputs2"
    base_path = "/shark/zhiwen/LoRAHunter/DiffSynth-Studio/outputs2"
    methods = ["sdv15_retrieval_testdata_500","stylus_testdata_500_totalpool_stylus","hunter_coco_part_500_qwenemb"] 
    methods = ['QwenImage','stylus_500_calllora_composer',]
    methods = ['QwenImage_diffusiondb200','diffuisondb_200_loraverse_qwenlora_com','diffusiondb_test_200_concept_qwenlora_gcl-trainfilter-110_set','diffusiondb_test_200_concept_qwenlora_gclemb016_set']
    for method in methods:
        image_dir = f"{base_path}/{method}"
        jsonl_path = "/shark/zhiwen/LoRAHunter/DiffSynth-Studio/test_data/diffusiondb_test_200.jsonl"

        output_jsonl = f"db200_out2_1/{method}_diversity.jsonl"

        evaluate_diversity(
            device="cuda:7",
            image_dir=image_dir,
            jsonl_path=jsonl_path,
            output_jsonl=output_jsonl,
            num_prompts=500,
        )