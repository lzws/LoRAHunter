import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3,4,5,6,7'
import time
import torch
from torch.utils.data import DataLoader
from safetensors.torch import load_file
from tqdm import tqdm

import json
import torch
import torch.multiprocessing as mp



from models import TextImageEncoder, QwenVLEncoder
from encoder import LoRAEncoder
from diffusers.loaders import StableDiffusionLoraLoaderMixin
from dataset import default_lora_patterns


import re
import torch.nn.functional as F
from rank_bm25 import BM25Okapi


class LoRARetriever:
    def __init__(
        self,
        index_path,
        metadata_path,
        device="cuda",
        alpha=0.7,
        dense_text_ratio=1.0,
        task="clipemb",
        use_subject_contrast=False,
        negative_weight=0.3,
        general_prompts=None,
        general_prompt_emb_path=None,
        img_prompt_batch_size=32,
        purity_weight_dense=0,
        purity_weight_hybrid=0,
    ):
        self.device = device
        self.alpha = alpha
        self.dense_text_ratio = dense_text_ratio
        self.task = task
        self.use_subject_contrast = use_subject_contrast
        self.negative_weight = negative_weight

        self.general_prompts = general_prompts if general_prompts is not None else []
        self.general_prompt_emb_path = general_prompt_emb_path
        self.img_prompt_batch_size = img_prompt_batch_size
        self.general_prompt_embs = None

        self.purity_weight_dense = purity_weight_dense
        self.purity_weight_hybrid = purity_weight_hybrid

        # query encoder
        if task == "clipemb":
            self.clip_encoder = TextImageEncoder(device=device).to(device)
            self.clip_encoder.eval()
        else:
            self.clip_encoder = QwenVLEncoder(device=device)
            self.clip_encoder.eval()

        # load dense index
        index_data = torch.load(index_path, map_location="cpu")
        self.model_files = index_data["model_files"]
        self.is_dual = index_data.get("is_dual", False)

        dtype = torch.float

        # text embeddings
        if "txt_embeddings" in index_data:
            self.lora_txt_embs = index_data["txt_embeddings"].to(dtype)
        elif "embeddings" in index_data:
            self.lora_txt_embs = index_data["embeddings"].to(dtype)
        else:
            raise KeyError("Index file must contain 'txt_embeddings' or 'embeddings'")

        # img embeddings
        self.lora_img_embs = None
        if self.is_dual:
            if "img_embeddings" not in index_data:
                raise KeyError("Dual index must contain 'img_embeddings'")
            self.lora_img_embs = index_data["img_embeddings"].to(dtype)

        print(f"[Index] loaded dense index")
        print(f"[Index] num files: {len(self.model_files)}")
        print(f"[Index] txt embeddings shape: {self.lora_txt_embs.shape}")
        print(f"[Index] is_dual: {self.is_dual}")
        if self.lora_img_embs is not None:
            print(f"[Index] img embeddings shape: {self.lora_img_embs.shape}")

        # normalize once and move to device
        self.lora_txt_embs = F.normalize(self.lora_txt_embs, dim=-1).to(device=device)

        if self.lora_img_embs is not None:
            self.lora_img_embs = F.normalize(self.lora_img_embs, dim=-1).to(device=device)

        # load metadata
        with open(metadata_path, "r", encoding="utf-8") as f:
            all_datas = [json.loads(line) for line in f]

        self.metadata_map = {}
        for data in all_datas:
            model_file = data.get("model_file", "")
            self.metadata_map[model_file] = data

        # build bm25 corpus aligned with dense index order
        self.documents = []
        self.tokenized_corpus = []

        missing_count = 0
        for model_file in self.model_files:
            data = self.metadata_map.get(model_file, None)
            if data is None:
                doc = ""
                missing_count += 1
            else:
                title = data.get("title", "")
                description = data.get("llm_description", "")
                tags = data.get("tags", [])
                if isinstance(tags, list):
                    tag_str = " ".join(tags)
                else:
                    tag_str = str(tags)
                # if not isinstance(description, str):
                #     description = title

                doc = f"{title} . {description}. {tag_str}".strip()

            self.documents.append(doc)
            self.tokenized_corpus.append(self._tokenize(doc))

        self.bm25 = BM25Okapi(self.tokenized_corpus)

        print(f"[Index] built BM25 corpus: {len(self.documents)} docs")
        print(f"[Index] metadata missing for {missing_count} docs")
        print(f"[Index] use_subject_contrast={self.use_subject_contrast}, negative_weight={self.negative_weight}")
        print(f"[Index] purity_weight_dense={self.purity_weight_dense}")
        print(f"[Index] purity_weight_hybrid={self.purity_weight_hybrid}")

        print(f"[Index] num general_prompts={len(self.general_prompts)}")
        print(f"[Index] img_prompt_batch_size={self.img_prompt_batch_size}")
        print(f"[Index] general_prompt_emb_path={self.general_prompt_emb_path}")

        # load precomputed Enc(p_i)
        if self.general_prompt_emb_path is not None:
            prompt_data = torch.load(self.general_prompt_emb_path, map_location="cpu")

            saved_prompts = prompt_data["prompts"]
            prompt_embs = prompt_data["prompt_embeddings"]

            if len(self.general_prompts) > 0:
                if len(saved_prompts) != len(self.general_prompts):
                    raise ValueError(
                        f"Loaded prompt count {len(saved_prompts)} != provided general_prompts count {len(self.general_prompts)}"
                    )
                for i, (a, b) in enumerate(zip(saved_prompts, self.general_prompts)):
                    if a != b:
                        raise ValueError(
                            f"Prompt mismatch at index {i}\nloaded: {a}\ncurrent: {b}"
                        )
            else:
                self.general_prompts = saved_prompts

            self.general_prompt_embs = F.normalize(prompt_embs, dim=-1).to(self.device)
            print(f"[Prompt] loaded general_prompt_embs: {self.general_prompt_embs.shape}")

    # ======================================================
    # basic utils
    # ======================================================
    def _tokenize(self, text):
        if not text:
            return []
        return re.findall(r"\w+", text.lower())

    def _minmax_norm(self, scores: torch.Tensor):
        min_v = scores.min()
        max_v = scores.max()
        if (max_v - min_v).abs() < 1e-12:
            return torch.zeros_like(scores)
        return (scores - min_v) / (max_v - min_v)

    # ======================================================
    # subject-contrast query templates
    # ======================================================
    def _build_subject_positive_query(self, query_text: str) -> str:
        return (
            f"{query_text}. "
            f"Focus on the main subject itself, the actual object/entity/animal/character itself, "
            f"as a standalone main subject."
        )

    def _build_subject_negative_query(self, query_text: str) -> str:
        return (
            f"{query_text}. "
            f"Pattern, print, decoration, clothing element, accessory, ornament, "
            f"background, scene, themed design, style reference, or contextual element "
            f"rather than the main subject itself."
        )

    # ======================================================
    # purity scoring utils
    # ======================================================
    def _split_query_terms(self, query_text: str):
        tokens = re.findall(r"\w+", query_text.lower())
        stopwords = {
            "a", "an", "the", "of", "in", "on", "at", "with", "for", "to",
            "and", "or", "style", "high", "quality"
        }
        tokens = [t for t in tokens if t not in stopwords]
        return tokens

    def _count_term_occurrence(self, text: str, term: str):
        if not text:
            return 0
        words = re.findall(r"\w+", text.lower())
        return sum(1 for w in words if w == term.lower())

    def _contains_phrase_loose(self, text: str, phrase: str):
        if not text or not phrase:
            return False
        text = text.lower()
        phrase = phrase.lower().strip()
        return phrase in text

    def _estimate_other_entity_density(self, text: str, query_terms):
        if not text:
            return 0.0

        words = re.findall(r"\w+", text.lower())
        if len(words) == 0:
            return 0.0

        remain = [w for w in words if w not in query_terms]

        stopwords = {
            "a", "an", "the", "of", "in", "on", "at", "with", "for", "to",
            "and", "or", "is", "are", "this", "that", "it", "its",
            "lora", "image", "images", "style", "creates", "create",
            "generate", "generates", "generated", "depicts", "depicting", "showing",
            "focus", "focused", "main", "subject", "actual", "standalone",
            "high", "quality", "details", "representation", "various", "scenarios"
        }
        remain = [w for w in remain if w not in stopwords]

        uniq = set(remain)
        return min(len(uniq) / 50.0, 1.0)

    def compute_subject_purity_score(self, query_text: str, model_file: str):
        data = self.metadata_map.get(model_file, {})

        title = str(data.get("title", "") or "")
        description = str(data.get("llm_description", "") or "")
        tags = data.get("tags", [])
        if isinstance(tags, list):
            tag_text = " ".join([str(x) for x in tags])
            tag_list = [str(x).lower() for x in tags]
        else:
            tag_text = str(tags)
            tag_list = re.findall(r"\w+", tag_text.lower())

        title_l = title.lower()
        desc_l = description.lower()
        tag_text_l = tag_text.lower()

        query_text_l = query_text.lower().strip()
        query_terms = self._split_query_terms(query_text_l)

        if len(query_terms) == 0:
            return 0.0

        score = 0.0

        # A. phrase match
        if self._contains_phrase_loose(title_l, query_text_l):
            score += 0.35
        elif self._contains_phrase_loose(desc_l[:200], query_text_l):
            score += 0.20

        # B. term coverage
        title_hit = 0
        desc_hit = 0
        tag_hit = 0
        early_desc_hit = 0
        total_occ = 0

        desc_early = desc_l[:300]

        for t in query_terms:
            c_title = self._count_term_occurrence(title_l, t)
            c_desc = self._count_term_occurrence(desc_l, t)
            c_tag = sum(1 for x in tag_list if x == t)
            c_early = self._count_term_occurrence(desc_early, t)

            if c_title > 0:
                title_hit += 1
            if c_desc > 0:
                desc_hit += 1
            if c_tag > 0:
                tag_hit += 1
            if c_early > 0:
                early_desc_hit += 1

            total_occ += c_title + c_desc + c_tag

        coverage_title = title_hit / len(query_terms)
        coverage_desc = desc_hit / len(query_terms)
        coverage_tag = tag_hit / len(query_terms)
        coverage_early = early_desc_hit / len(query_terms)

        score += 0.25 * coverage_title
        score += 0.15 * coverage_tag
        score += 0.15 * coverage_early
        score += 0.10 * min(total_occ / (2.0 * len(query_terms)), 1.0)

        # C. weak appearance penalty
        if coverage_title == 0 and coverage_tag == 0 and coverage_early == 0:
            score -= 0.20

        # D. topic dispersion penalty
        full_text = f"{title} {description} {tag_text}"
        other_entity_density = self._estimate_other_entity_density(full_text, set(query_terms))
        score -= 0.18 * other_entity_density

        # E. weak generic contamination penalty
        generic_noise_words = {
            "scene", "background", "pattern", "print", "decoration",
            "ornament", "toy", "miniature", "diorama", "crash",
            "accident", "wreck", "debris", "smoke", "flame", "rescue",
            "emergency"
        }

        noise_hits = sum(1 for w in generic_noise_words if w in desc_l or w in title_l or w in tag_text_l)
        score -= min(noise_hits * 0.03, 0.18)

        score = max(min(score, 1.0), -1.0)
        return score

    def apply_subject_purity_adjustment(self, query_text: str, scores: torch.Tensor, weight: float = 0.15):
        purity_scores = []
        for mf in self.model_files:
            p = self.compute_subject_purity_score(query_text, mf)
            purity_scores.append(p)

        purity_scores = torch.tensor(purity_scores, dtype=scores.dtype)
        adjusted_scores = scores + weight * purity_scores
        return adjusted_scores, purity_scores

    # ======================================================
    # image route
    # ======================================================
    @torch.no_grad()
    def build_img_query_embedding(self, query_text):
        if self.general_prompts is None or len(self.general_prompts) == 0:
            base_query_emb, _ = self.encode_query_routes(query_text)
            return base_query_emb

        if self.general_prompt_embs is None:
            raise ValueError(
                "general_prompt_embs is None. "
                "Please provide general_prompt_emb_path or precompute prompt embeddings first."
            )

        if len(self.general_prompts) != self.general_prompt_embs.shape[0]:
            raise ValueError(
                f"Prompt count mismatch: len(general_prompts)={len(self.general_prompts)} "
                f"but general_prompt_embs.shape[0]={self.general_prompt_embs.shape[0]}"
            )

        qp_texts = [f"{query_text}. {p}" for p in self.general_prompts]

        qp_embs = []
        bs = self.img_prompt_batch_size

        for i in range(0, len(qp_texts), bs):
            batch_texts = qp_texts[i:i + bs]
            batch_embs = self.clip_encoder.encoding_text(batch_texts)
            batch_embs = F.normalize(batch_embs, dim=-1).to(self.device)
            qp_embs.append(batch_embs)

        qp_embs = torch.cat(qp_embs, dim=0)
        delta = qp_embs - self.general_prompt_embs
        img_query_emb = delta.mean(dim=0, keepdim=True)
        img_query_emb = F.normalize(img_query_emb, dim=-1)

        return img_query_emb

    @torch.no_grad()
    def encode_query_routes(self, query_text):
        query_txt_emb = self.clip_encoder.encoding_text([query_text])
        query_txt_emb = F.normalize(query_txt_emb, dim=-1).to(device=self.device)

        query_img_emb = None
        if self.is_dual and self.lora_img_embs is not None:
            if query_txt_emb.shape[-1] == self.lora_img_embs.shape[-1]:
                query_img_emb = query_txt_emb
            else:
                print(
                    f"[Warn] query_txt_emb dim={query_txt_emb.shape[-1]} "
                    f"!= lora_img_embs dim={self.lora_img_embs.shape[-1]}, "
                    f"fallback to text route only."
                )

        return query_txt_emb, query_img_emb

    # ======================================================
    # dense scoring
    # ======================================================
    @torch.no_grad()
    def _compute_text_scores(self, query_text):
        if self.use_subject_contrast:
            pos_query = self._build_subject_positive_query(query_text)
            neg_query = self._build_subject_negative_query(query_text)

            pos_txt_emb, _ = self.encode_query_routes(pos_query)
            neg_txt_emb, _ = self.encode_query_routes(neg_query)

            pos_txt_emb = pos_txt_emb.to(self.lora_txt_embs.dtype)
            neg_txt_emb = neg_txt_emb.to(self.lora_txt_embs.dtype)

            pos_scores = (pos_txt_emb @ self.lora_txt_embs.t())[0]
            neg_scores = (neg_txt_emb @ self.lora_txt_embs.t())[0]

            text_scores = pos_scores - self.negative_weight * neg_scores

            return {
                "text_scores": text_scores,
                "text_pos_scores": pos_scores,
                "text_neg_scores": neg_scores,
            }

        else:
            query_txt_emb, _ = self.encode_query_routes(query_text)
            query_txt_emb = query_txt_emb.to(self.lora_txt_embs.dtype)
            text_scores = (query_txt_emb @ self.lora_txt_embs.t())[0]

            return {
                "text_scores": text_scores,
                "text_pos_scores": text_scores,
                "text_neg_scores": None,
            }

    @torch.no_grad()
    def retrieve_dense_scores(self, query_text):
        text_out = self._compute_text_scores(query_text)
        text_scores = text_out["text_scores"]
        text_pos_scores = text_out["text_pos_scores"]
        text_neg_scores = text_out["text_neg_scores"]

        query_txt_emb, query_img_emb = self.encode_query_routes(query_text)

        img_scores = None
        dense_scores = text_scores

        if self.is_dual and (self.lora_img_embs is not None) and (query_img_emb is not None):
            img_query_emb = self.build_img_query_embedding(query_text)
            query_img_emb = query_img_emb.to(self.lora_img_embs.dtype)
            img_scores = (query_img_emb @ self.lora_img_embs.t())[0]

            dense_scores = (
                self.dense_text_ratio * text_scores
                + (1.0 - self.dense_text_ratio) * img_scores
            )

        return {
            "dense_scores": dense_scores.detach().cpu(),
            "text_scores": text_scores.detach().cpu(),
            "img_scores": None if img_scores is None else img_scores.detach().cpu(),
            "text_pos_scores": text_pos_scores.detach().cpu(),
            "text_neg_scores": None if text_neg_scores is None else text_neg_scores.detach().cpu(),
        }

    def retrieve_bm25_scores(self, query_text):
        tokenized_query = self._tokenize(query_text)
        scores = self.bm25.get_scores(tokenized_query)
        return torch.tensor(scores, dtype=torch.float)

    # ======================================================
    # retrieve
    # ======================================================
    @torch.no_grad()
    def retrieve(self, query_text, top_k=5, mode="dense"):
        mode = mode.lower()
        assert mode in ["dense", "bm25", "hybrid"], f"Unsupported mode: {mode}"

        if mode == "dense":
            dense_out = self.retrieve_dense_scores(query_text)
            dense_scores = dense_out["dense_scores"]
            text_scores = dense_out["text_scores"]
            img_scores = dense_out["img_scores"]
            text_pos_scores = dense_out["text_pos_scores"]
            text_neg_scores = dense_out["text_neg_scores"]

            adjusted_dense_scores, purity_scores = self.apply_subject_purity_adjustment(
                query_text=query_text,
                scores=dense_scores,
                weight=self.purity_weight_dense,
            )

            top_scores, top_indices = torch.topk(adjusted_dense_scores, k=top_k)

            results = []
            for score, idx in zip(top_scores.tolist(), top_indices.tolist()):
                results.append({
                    "model_file": self.model_files[idx],
                    "score": score,
                    "raw_dense_score": dense_scores[idx].item(),
                    "dense_score": score,
                    "text_dense_score": text_scores[idx].item(),
                    "text_pos_score": text_pos_scores[idx].item(),
                    "text_neg_score": None if text_neg_scores is None else text_neg_scores[idx].item(),
                    "img_dense_score": None if img_scores is None else img_scores[idx].item(),
                    "purity_score": purity_scores[idx].item(),
                    "bm25_score": None,
                    "mode": "dense",
                    "is_dual_index": self.is_dual,
                    "dense_text_ratio": self.dense_text_ratio,
                    # "use_subject_contrast": self.use_subject_contrast,
                    "negative_weight": self.negative_weight,
                    "purity_weight_dense": self.purity_weight_dense,
                })
            return results

        elif mode == "bm25":
            bm25_scores = self.retrieve_bm25_scores(query_text)
            top_scores, top_indices = torch.topk(bm25_scores, k=top_k)

            results = []
            for score, idx in zip(top_scores.tolist(), top_indices.tolist()):
                results.append({
                    "model_file": self.model_files[idx],
                    "score": score,
                    "dense_score": None,
                    "text_dense_score": None,
                    "text_pos_score": None,
                    "text_neg_score": None,
                    "img_dense_score": None,
                    "purity_score": None,
                    "bm25_score": score,
                    "mode": "bm25",
                    "is_dual_index": self.is_dual,
                })
            return results

        else:  # hybrid
            dense_out = self.retrieve_dense_scores(query_text)
            dense_scores = dense_out["dense_scores"]
            text_scores = dense_out["text_scores"]
            img_scores = dense_out["img_scores"]
            text_pos_scores = dense_out["text_pos_scores"]
            text_neg_scores = dense_out["text_neg_scores"]

            bm25_scores = self.retrieve_bm25_scores(query_text)

            dense_scores_norm = self._minmax_norm(dense_scores)
            bm25_scores_norm = self._minmax_norm(bm25_scores)

            base_final_scores = self.alpha * dense_scores_norm + (1.0 - self.alpha) * bm25_scores_norm

            if self.purity_weight_hybrid > 0:
                adjusted_final_scores, purity_scores = self.apply_subject_purity_adjustment(
                    query_text=query_text,
                    scores=base_final_scores,
                    weight=self.purity_weight_hybrid,
                )
                base_final_scores = adjusted_final_scores

            top_scores, top_indices = torch.topk(base_final_scores, k=top_k)

            results = []
            for score, idx in zip(top_scores.tolist(), top_indices.tolist()):
                results.append({
                    "model_file": self.model_files[idx],
                    "score": score,
                    "raw_hybrid_score": base_final_scores[idx].item(),
                    "dense_score": dense_scores[idx].item(),
                    "text_dense_score": text_scores[idx].item(),
                    "text_pos_score": text_pos_scores[idx].item(),
                    "text_neg_score": None if text_neg_scores is None else text_neg_scores[idx].item(),
                    "img_dense_score": None if img_scores is None else img_scores[idx].item(),
                    "bm25_score": bm25_scores[idx].item(),
                    "dense_score_norm": dense_scores_norm[idx].item(),
                    "bm25_score_norm": bm25_scores_norm[idx].item(),
                    # "purity_score": purity_scores[idx].item(),
                    # "mode": "hybrid",
                    # "is_dual_index": self.is_dual,
                    # "dense_text_ratio": self.dense_text_ratio,
                    # "alpha": self.alpha,
                    # "use_subject_contrast": self.use_subject_contrast,
                    # "negative_weight": self.negative_weight,
                    # "purity_weight_hybrid": self.purity_weight_hybrid,
                })
            return results


def call_lora(
    index_path="lora_index_train.pt",
    metadata_path="lora_pool_metadata.jsonl",
    test_data_path="retrieval_testdata_100.jsonl",
    output_path="test_data/call_lora_res.jsonl",
    device="cuda",
    mode="dense",   # dense / bm25 / hybrid
    top_k=10,
    alpha=0.7,
    dense_text_ratio=1.0,
    task="clipemb",
    use_subject_contrast=False,
    negative_weight=0.4,
    general_prompts=None,
    general_prompt_emb_path="/shark/zhiwen/LoRAHunter/carlos_difftxt_qwenemb.pt",
    img_prompt_batch_size=32,
    purity_weight_dense=0,
    purity_weight_hybrid=0,
):
    retriever = LoRARetriever(
        index_path=index_path,
        metadata_path=metadata_path,
        device=device,
        alpha=alpha,
        dense_text_ratio=dense_text_ratio,
        task=task,
        use_subject_contrast=use_subject_contrast,
        negative_weight=negative_weight,
        general_prompt_emb_path=general_prompt_emb_path,
        img_prompt_batch_size=img_prompt_batch_size,
        purity_weight_hybrid=purity_weight_hybrid,
    )

    with open(test_data_path, "r", encoding="utf-8") as f:
        test_datas = [json.loads(line) for line in f]

    with open(output_path, "w", encoding="utf-8") as fout:
        for i, data in enumerate(test_datas):
            print(f"[{i}]")
            extract_concept = data["extract_concept"]
            data["retrieval_results"] = {}

            for ec in extract_concept:
                keyword = ec["keyword"]
                retrieval_description = ec["retrieval_description"]

                # query = f"{keyword} " 
                retrieval_description = retrieval_description.replace("This LoRA generates","").replace("This adapter generates","").replace("adapter","").replace("This","").replace("This adapter should generate","")
                query = f"{keyword}, {retrieval_description} "

                print(f"  query: {query}")

                topk = retriever.retrieve(
                    query_text=query,
                    top_k=top_k,
                    mode=mode,
                )
                data["retrieval_results"][keyword] = topk

            fout.write(json.dumps(data, ensure_ascii=False) + "\n")

    print(f"[Done] saved results to {output_path}")



def call_lora_one(
    query=" ",
    index_path="/shark/zhiwen/LoRAHunter/sd_encoder/lora_index_qwenemb_rwamse_all.pt",
    metadata_path="../SD_adapter_metadata/sd_lora_1/exist_file_adapters.jsonl",
    device="cuda:4",
    mode="hybrid",   # dense / bm25 / hybrid
    top_k=10,
    alpha=0.8,
    dense_text_ratio=0.8,
    task="qwenemb",
    use_subject_contrast=True,
    negative_weight=0.3,
    general_prompts=None,
    general_prompt_emb_path="/shark/zhiwen/LoRAHunter/carlos_difftxt_qwenemb.pt",
    img_prompt_batch_size=32,
    purity_weight_hybrid=0,
):
    retriever = LoRARetriever(
        index_path=index_path,
        metadata_path=metadata_path,
        device=device,
        alpha=alpha,
        dense_text_ratio=dense_text_ratio,
        task=task,
        use_subject_contrast=use_subject_contrast,
        negative_weight=negative_weight,
        general_prompt_emb_path=general_prompt_emb_path,
        img_prompt_batch_size=img_prompt_batch_size,
        purity_weight_hybrid=purity_weight_hybrid,
    )

    topk = retriever.retrieve(
        query_text=query,
        top_k=top_k,
        mode=mode,
    )

    metadata_map = retriever.metadata_map

    for i, item in enumerate(topk):
        print(f"{i}: {item['model_file']}, score: {item['score']}, dense_score_norm: {item['dense_score_norm']}, bm25_score_norm: {item['bm25_score_norm']}")
        print(f"text_pos_score :{item['text_pos_score']}, text_neg_score: {item['text_neg_score']} \n")
        data = metadata_map[item["model_file"]]
        print(f"  title: {data['title']}, description: {data['llm_description']}")
        print("\n")



    


if __name__ == "__main__":
    index_paths = ["lora_index/qwenemb_cosinsmoothl1_all_5-199.pt", "lora_index/clipemb_contastive_all_2-1248.pt"]
    call_lora(
        index_path=index_paths[0],
        metadata_path="../SD_adapter_metadata/sd_lora_1/exist_file_adapters.jsonl",
        test_data_path="test_data/retrieval_testdata_500_extract_2.jsonl",
        output_path="test_data/data_500_qwenemb_5-199_2.jsonl",
        device="cuda",
        mode="hybrid",   # dense / bm25 / hybrid
        top_k=40,
        alpha=0.75,
        dense_text_ratio=0.85,
        task="qwenemb",
        use_subject_contrast=False,
        purity_weight_dense=0,
        purity_weight_hybrid=0,
    )

    index_path = "/shark/zhiwen/LoRAHunter/sd_encoder/lora_index/qwenemb_cosinsmoothl1_all_5-199.pt"
    # index_path = "/shark/zhiwen/LoRAHunter/sd_encoder/lora_index/clipemb_contastive_all_2-1248.pt"
    task = 'qwenemb'
    query = "A koi fish in watercolor style"
    query = "A koi fish"
    # query = "watercolor style"
    # query = "Cyberpunk style"
    # query = "A dragon flying over a city"
    # query = "a city"
    # query = "A fantastical witch in a forest in vibrant style"
    # query = "a forest in vibrant style"
    # query = "A man wearing a helmet"
    # query = "A futuristic motorcycle with a neon-accented design"
    # query = "A futuristic motorcycle"
    # query = "An orange train traveling along train tracks near a train yard."
    # query = "cake"
    # query = "woman and child. "
    # query = "A detailed drawing that  looks like a graphite pencil  sketch with cross-hatching  and shading"
    query = "close up"
    query = "side by side"
    query = "skateboards"
    query = "helmet"
    query = "sidewalk"

    query_text = "city street, This adapter generates urban street environments with asphalt textures and building facades. It focuses on background elements with realistic lighting and city atmosphere, minimizing foreground objects."
    query_text = "children,This LoRA generates groups of young human subjects with varied expressions and casual clothing. It features natural skin textures, appropriate scaling for passengers, and dynamic poses suitable for riding. The aesthetic focuses on the demographics of kids in a fun setting without emphasizing specific individual identities"
    query_text = "men in cloaks, This adapter should generate male figures wearing long, flowing cloaks. Focus on the fabric texture, drape, and silhouette of the cloaks over the human form. Avoid specific fantasy classes like wizards unless specified, keeping it grounded in the clothing item itself."
    query_text = "skateboards, realistic skateboards. Focus on the deck graphics, wheel structure, and truck hardware. The object should be distinct and detailed, suitable for being ridden or held."
    query_text = "man, realistic human male subjects with natural skin textures and anatomical proportions. It focuses on facial features and general body structure suitable for standing poses. Best for central characters in various lighting conditions, emphasizing human presence without specific costume constraints."
   
    query_text = "hrow hug, This LoRA generates detailed throw rugs with various weaving patterns, soft fabric textures, and decorative fringe. It features intricate design motifs, rich colors, and realistic textile fibers. The style is photorealistic with soft lighting emphasizing the rug's texture. Best for interior shots where the rug is the central subject, focusing on material quality without surrounding furniture."
    # query_text = "watercolor style"
    # query_text = "snow, white snow-covered surfaces with soft textures and cold lighting effects. It features particle details, ground coverage, and winter atmospheric tones. The style emphasizes terrain texture and environmental mood, suitable for grounding subjects in a winter setting"
    # query_text = "white tent, This adapter generates realistic white tents, emphasizing fabric textures, structural poles, and canopy shapes. It features clean white surfaces, potential shading details, and beach or camping contexts. The style is photorealistic with natural lighting, focusing on the tent's material and form as the central subject."
    # query_text = "crossroads sign, Retrieve an adapter focused on road infrastructure signage, specifically crossroads or intersection signs. The adapter should generate clear signboards with typical traffic colors and symbols mounted on poles, prioritizing the sign's visual details over the surrounding environment"
    query_text = "vector illustration, vector art styles, featuring clean paths, solid colors, minimal shading, and graphic design aesthetics suitable for illustrations"
    query_text = "award, This adapter generates objects representing recognition, such as trophies, medals, or certificates. Focus on materials like gold, silver, or glass, with detailed engravings or ribbons. The object should be clearly visible as a symbol of achievement."
    # call_lora_one(query=query_text, index_path=index_path, task=task, alpha=0.85, dense_text_ratio=0.8,use_subject_contrast=False,negative_weight=0.5,purity_weight_hybrid=0)

# nohup python call_lora.py > build_lora_index_qwen.log 2>&1 &