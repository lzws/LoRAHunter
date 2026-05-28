import os
import json
from typing import List, Dict, Any

from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModel
import ImageReward as RM


class AlignmentRewardEvaluator:
    def __init__(
        self,
        device="cuda",
        clip_model_name="/shark/zhiwen/LoRAHunter/rank_encoder/models/muse/openai-clip-vit-large-patch14",
        pickscore_model_name="/shark/zhiwen/LoRAHunter/rank_encoder/models/AI-ModelScope/PickScore_v1",
        imagereward_name="ImageReward-v1.0",
    ):
        self.device = device if torch.cuda.is_available() else "cpu"

        # -------------------------
        # CLIP
        # -------------------------
        self.clip_processor = AutoProcessor.from_pretrained(clip_model_name)
        self.clip_model = AutoModel.from_pretrained(clip_model_name).to(self.device)
        self.clip_model.eval()

        # -------------------------
        # PickScore
        # -------------------------
        self.pick_processor = AutoProcessor.from_pretrained(pickscore_model_name)
        self.pick_model = AutoModel.from_pretrained(pickscore_model_name).to(self.device)
        self.pick_model.eval()

        # -------------------------
        # ImageReward
        # -------------------------
        self.rm_model = RM.load(imagereward_name)

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
        pre_fix = ''
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

    def _max_or_none(self, xs):
        xs = [x for x in xs if x is not None]
        if len(xs) == 0:
            return None
        return max(xs)

    # =====================================================
    # CLIP
    # =====================================================
    @torch.no_grad()
    def score_clip_batch(self, prompt: str, images: List[Image.Image], batch_size=16):
        """
        返回:
            clip_cosine_scores: 原始 cosine
            clipscore_scores:   max(100 * cosine, 0)

        说明:
        - LoRAverse 论文里的 CLIP 分数大概率对应 CLIPScore 风格，即 100*cosine
        - raw cosine 也保留，便于你自己分析
        """
        all_cos_scores = []

        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            batch_prompts = [prompt] * len(batch_images)

            inputs = self.clip_processor(
                text=batch_prompts,
                images=batch_images,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)

            image_features = self.clip_model.get_image_features(pixel_values=inputs["pixel_values"])
            text_features = self.clip_model.get_text_features(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )

            image_features = F.normalize(image_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)

            cos_scores = (image_features * text_features).sum(dim=-1)
            all_cos_scores.extend(cos_scores.detach().cpu().tolist())

        clipscore_scores = [max(100.0 * x, 0.0) for x in all_cos_scores]
        return all_cos_scores, clipscore_scores

    # =====================================================
    # PickScore
    # =====================================================
    @torch.no_grad()
    def score_pickscore_batch(self, prompt: str, images: List[Image.Image], batch_size=16):
        """
        PickScore 通常直接报 raw score，不需要乘 100
        """
        all_scores = []

        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            batch_prompts = [prompt] * len(batch_images)

            image_inputs = self.pick_processor(
                images=batch_images,
                return_tensors="pt",
            ).to(self.device)

            text_inputs = self.pick_processor(
                text=batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77,
            ).to(self.device)

            image_embs = self.pick_model.get_image_features(**image_inputs)
            text_embs = self.pick_model.get_text_features(**text_inputs)

            image_embs = F.normalize(image_embs, dim=-1)
            text_embs = F.normalize(text_embs, dim=-1)

            scores = (image_embs * text_embs).sum(dim=-1)
            all_scores.extend(scores.detach().cpu().tolist())

        return all_scores

    # =====================================================
    # ImageReward
    # =====================================================
    def score_imagereward_batch(self, prompt: str, image_paths: List[str]):
        """
        ImageReward 一般也直接报 raw reward，不乘 100
        """
        scores = []
        for p in image_paths:
            try:
                s = self.rm_model.score(prompt, p)
                if isinstance(s, torch.Tensor):
                    s = s.item()
                scores.append(float(s))
            except Exception as e:
                print(f"[Warn] ImageReward failed: {p}, err={e}")
                scores.append(None)
        return scores

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
        pick_batch_size=16,
        num_prompts=500,
        pstart = 0,
        pend = 500,
    ):
        datas = self.load_jsonl(jsonl_path)

        datas = datas[pstart:pend]

        global_clip_cos = []
        global_clipscore = []
        global_pickscore = []
        global_imagereward = []
        skip_list = [0,1,4,11,25,74,85,140,147,195,202,266,276,315,335,380,386,390,403,448,464,]
        skip_list = [0,1,4,11,25,74,85,140,147,195,202,266,276,315,335,380,386,390,403,448,464,46,60,62,265,490,291,450,484,393,212,154,21,300,297,109,371,66,392,326,415,142,233,440,449,499,251,47,356,339,101,436,133,76,459,193,366,257,143,303,294,163,279,443,184,491,293,262,32,353,48,383,45,287,275,394,183,156,81,301,309,24,106,340,334,358,18,378,229,161,263,254,141,446,320,286,296,29,282,175,476,433,311,349,40,268]
        with open(output_jsonl, "w", encoding="utf-8") as fout:
            for data in tqdm(datas, desc="Evaluating alignment/reward"):
                iid = int(data[iid_key])
                if iid in skip_list:
                    continue
                prompt = data[prompt_key]

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
                        "images": [],
                        "mean_clip_cosine": None,
                        "mean_clipscore": None,
                        "mean_pickscore": None,
                        "mean_imagereward": None,
                        "max_clip_cosine": None,
                        "max_clipscore": None,
                        "max_pickscore": None,
                        "max_imagereward": None,
                    }
                    fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                    continue

                images, valid_paths = self.batch_load_images(image_paths)

                if len(images) == 0:
                    result = {
                        "iid": iid,
                        "prompt": prompt,
                        "num_valid_images": 0,
                        "images": [],
                        "mean_clip_cosine": None,
                        "mean_clipscore": None,
                        "mean_pickscore": None,
                        "mean_imagereward": None,
                        "max_clip_cosine": None,
                        "max_clipscore": None,
                        "max_pickscore": None,
                        "max_imagereward": None,
                    }
                    fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                    continue

                # -------------------------
                # CLIP
                # -------------------------
                try:
                    clip_cos_scores, clipscore_scores = self.score_clip_batch(
                        prompt=prompt,
                        images=images,
                        batch_size=clip_batch_size,
                    )
                except Exception as e:
                    print(f"[Warn] CLIP batch failed for iid={iid}, err={e}")
                    clip_cos_scores = [None] * len(images)
                    clipscore_scores = [None] * len(images)

                # -------------------------
                # PickScore
                # -------------------------
                try:
                    pick_scores = self.score_pickscore_batch(
                        prompt=prompt,
                        images=images,
                        batch_size=pick_batch_size,
                    )
                except Exception as e:
                    print(f"[Warn] PickScore batch failed for iid={iid}, err={e}")
                    pick_scores = [None] * len(images)

                # -------------------------
                # ImageReward
                # -------------------------
                try:
                    ir_scores = self.score_imagereward_batch(
                        prompt=prompt,
                        image_paths=valid_paths,
                    )
                except Exception as e:
                    print(f"[Warn] ImageReward failed for iid={iid}, err={e}")
                    ir_scores = [None] * len(images)

                n = len(images)
                clip_cos_scores = (clip_cos_scores + [None] * n)[:n]
                clipscore_scores = (clipscore_scores + [None] * n)[:n]
                pick_scores = (pick_scores + [None] * n)[:n]
                ir_scores = (ir_scores + [None] * n)[:n]

                per_image_results = []
                local_clip_cos = []
                local_clipscore = []
                local_pickscore = []
                local_ir = []

                for p, ccos, cscore, pscore, ir in zip(
                    valid_paths, clip_cos_scores, clipscore_scores, pick_scores, ir_scores
                ):
                    per_image_results.append({
                        "image_path": p,
                        "clip_cosine": ccos,
                        "clipscore": cscore,
                        "pickscore": pscore,
                        "imagereward": ir,
                    })

                    if ccos is not None:
                        local_clip_cos.append(ccos)
                        global_clip_cos.append(ccos)

                    if cscore is not None:
                        local_clipscore.append(cscore)
                        global_clipscore.append(cscore)

                    if pscore is not None:
                        local_pickscore.append(pscore)
                        global_pickscore.append(pscore)

                    if ir is not None:
                        local_ir.append(ir)
                        global_imagereward.append(ir)

                result = {
                    "iid": iid,
                    "prompt": prompt,
                    "num_valid_images": len(per_image_results),
                    "images": per_image_results,
                    "mean_clip_cosine": self._mean_or_none(local_clip_cos),
                    "mean_clipscore": self._mean_or_none(local_clipscore),
                    "mean_pickscore": self._mean_or_none(local_pickscore),
                    "mean_imagereward": self._mean_or_none(local_ir),
                    "max_clip_cosine": self._max_or_none(local_clip_cos),
                    "max_clipscore": self._max_or_none(local_clipscore),
                    "max_pickscore": self._max_or_none(local_pickscore),
                    "max_imagereward": self._max_or_none(local_ir),
                }

                fout.write(json.dumps(result, ensure_ascii=False) + "\n")

        summary = {
            "num_prompts": len(datas),
            "global_mean_clip_cosine": self._mean_or_none(global_clip_cos),
            "global_mean_clipscore": self._mean_or_none(global_clipscore),
            "global_mean_pickscore": self._mean_or_none(global_pickscore),
            "global_mean_imagereward": self._mean_or_none(global_imagereward),
            "global_max_clip_cosine": self._max_or_none(global_clip_cos),
            "global_max_clipscore": self._max_or_none(global_clipscore),
            "global_max_pickscore": self._max_or_none(global_pickscore),
            "global_max_imagereward": self._max_or_none(global_imagereward),
        }

        summary_path = output_jsonl.replace(".jsonl", "_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"[Done] saved per-prompt alignment/reward results to: {output_jsonl}")
        print(f"[Done] saved summary to: {summary_path}")
        print(json.dumps(summary, ensure_ascii=False, indent=2))


def evaluate_alignment_reward(
    image_dir,
    jsonl_path,
    output_jsonl,
    device="cuda",
    iid_key="iid",
    prompt_key="prompt",
    num_images_per_prompt=10,
    clip_batch_size=16,
    pick_batch_size=16,
    num_prompts=500,
    pstart=0,
    pend=500,
):
    evaluator = AlignmentRewardEvaluator(device=device)
    evaluator.evaluate_from_jsonl(
        image_dir=image_dir,
        jsonl_path=jsonl_path,
        output_jsonl=output_jsonl,
        iid_key=iid_key,
        prompt_key=prompt_key,
        num_images_per_prompt=num_images_per_prompt,
        clip_batch_size=clip_batch_size,
        pick_batch_size=pick_batch_size,
        num_prompts=num_prompts,
        pstart = pstart,
        pend = pend,
    )


if __name__ == "__main__":
    base_path = "/shark/zhiwen/LoRAHunter/DiffSynth-Studio/outputs2"
    methods = ["sdv15_retrieval_testdata_500","stylus_testdata_500_totalpool_stylus","hunter_coco_part_500_qwenemb"] 
    method = methods[1]

    methods = ['QwenImage','stylus_500_calllora_composer']

    methods = ['QwenImage','diffuisondb_200_loraverse_qwenlora_com','stylus_db200_calllora_composer','diffusiondb_test_200_concept_qwenlora_gcl-trainfilter-110_set']

    methods = ['QwenImage_diffusiondb200','diffuisondb_200_loraverse_qwenlora_com','diffusiondb_test_200_concept_qwenlora_gcl-trainfilter-110_set','diffusiondb_test_200_concept_qwenlora_gclemb016_set']
    
    for method in methods[:]:
        image_dir = f"{base_path}/{method}"
        jsonl_path = jsonl_path = "/shark/zhiwen/LoRAHunter/DiffSynth-Studio/test_data/diffusiondb_test_200.jsonl"

        pstart,pend = 0,200
        output_jsonl = f"db200_out2/{method}_quality_{pstart}-{pend}.jsonl"

        evaluate_alignment_reward(
            device="cuda:0",
            num_images_per_prompt=10,
            image_dir=image_dir,
            jsonl_path=jsonl_path,
            output_jsonl=output_jsonl,
            num_prompts=40,
            pstart = pstart,
            pend = pend,
        )

# nohup python eval_alignment_reward.py > eval_alignment_reward_2-1.log 2>&1 &