import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'

import json
import itertools
from typing import List, Dict, Any, Optional

import torch
import torch.nn.functional as F

import json
import itertools
from typing import List, Dict, Any, Optional

import torch
import torch.nn.functional as F

from models import TextImageEncoder

# ==========================================================
# 1. LoRA embedding index
# ==========================================================
class LoRAIndexStore:
    """
    从 lora_index_train.pt 中加载:
    - model_files
    - embeddings

    并提供 model_file -> embedding 查询能力
    """
    def __init__(self, index_path: str, device: str = "cpu"):
        self.device = device

        index_data = torch.load(index_path, map_location="cpu")
        self.model_files = index_data["model_files"]
        self.embeddings = index_data["embeddings"].float()   # [N, D]
        self.embeddings = F.normalize(self.embeddings, dim=-1)

        self.model_file_to_idx = {mf: i for i, mf in enumerate(self.model_files)}
        print(f"[LoRAIndexStore] loaded {len(self.model_files)} embeddings from {index_path}")

    def get_embedding(self, model_file: str) -> Optional[torch.Tensor]:
        idx = self.model_file_to_idx.get(model_file, None)
        if idx is None:
            return None
        return self.embeddings[idx].to(self.device)


# ==========================================================
# 2. Combination optimizer with quality-aware DPP
# ==========================================================
class LoRACombinationOptimizer:
    """
    组合打分包含四项：
    1. individual match
    2. pairwise redundancy/conflict penalty
    3. quality-aware DPP diversity
    4. global prompt alignment

    最终再做 diversified reranking，输出多个高质量且不同的组合。
    """
    def __init__(
        self,
        # 总分四项权重
        alpha: float = 1.0,   # individual score 权重
        beta: float = 0.3,    # pairwise penalty 权重
        gamma: float = 0.2,   # DPP 权重
        delta: float = 0.5,   # global prompt alignment 权重

        # individual 内部权重
        lambda_concept: float = 1.0,   # LoRA 与对应 concept 的相似度权重
        lambda_prompt: float = 0.2,    # LoRA 与完整 prompt 的相似度权重

        # DPP 参数
        dpp_sigma: float = 0.5,        # RBF similarity kernel 带宽
        dpp_tau: float = 0.2,          # quality -> exp(score / tau) 的温度
        dpp_use_quality: bool = True,  # 是否启用 quality-aware DPP

        # 结果级多样化参数
        lambda_quality: float = 0.8,
        overlap_weight: float = 0.7,
        emb_weight: float = 0.3,

        device: str = "cpu",
    ):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

        self.lambda_concept = lambda_concept
        self.lambda_prompt = lambda_prompt

        self.dpp_sigma = dpp_sigma
        self.dpp_tau = dpp_tau
        self.dpp_use_quality = dpp_use_quality

        self.lambda_quality = lambda_quality
        self.overlap_weight = overlap_weight
        self.emb_weight = emb_weight

        self.device = device

    def _normalize_vec(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=0)

    # ------------------------------------------------------
    # item-level quality
    # ------------------------------------------------------
    def compute_item_quality_score(
        self,
        lora_emb: torch.Tensor,
        concept_emb: torch.Tensor,
        full_prompt_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        单个 LoRA 的质量分数：
            u_i = lambda_concept * cos(lora_i, concept_i)
                + lambda_prompt  * cos(lora_i, full_prompt)

        这是：
        - local concept match
        - global prompt prior
        的加权和
        """
        local_sim = torch.dot(lora_emb, concept_emb)
        prompt_sim = torch.dot(lora_emb, full_prompt_emb)
        return self.lambda_concept * local_sim + self.lambda_prompt * prompt_sim

    # ------------------------------------------------------
    # 1. individual score
    # ------------------------------------------------------
    def compute_individual_score(
        self,
        selected_lora_embs: List[torch.Tensor],
        concept_embs: List[torch.Tensor],
        full_prompt_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        individual score = alpha * sum_i quality_i
        """
        score = torch.tensor(0.0, device=selected_lora_embs[0].device)
        for lora_emb, concept_emb in zip(selected_lora_embs, concept_embs):
            score += self.compute_item_quality_score(
                lora_emb=lora_emb,
                concept_emb=concept_emb,
                full_prompt_emb=full_prompt_emb,
            )
        return self.alpha * score

    # ------------------------------------------------------
    # 2. pairwise score
    # ------------------------------------------------------
    def compute_pairwise_score(
        self,
        selected_lora_embs: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        pairwise redundancy/conflict penalty:
            sum_{i<j} -cos(e_i, e_j)

        两个 LoRA 越相似，惩罚越大。
        """
        score = torch.tensor(0.0, device=selected_lora_embs[0].device)
        n = len(selected_lora_embs)

        for i in range(n):
            for j in range(i + 1, n):
                score += -torch.dot(selected_lora_embs[i], selected_lora_embs[j])

        return self.beta * score

    # ------------------------------------------------------
    # 3. quality-aware DPP
    # ------------------------------------------------------
    def compute_similarity_kernel(
        self,
        selected_lora_embs: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        计算相似性核 S:
            S_ij = exp(-||e_i - e_j||^2 / (2*sigma^2))

        已归一化 embedding 下：
            ||e_i - e_j||^2 = 2 - 2*cos(e_i, e_j)
        """
        E = torch.stack(selected_lora_embs, dim=0)  # [k, D]
        cos_sim = E @ E.t()
        dist_sq = 2 - 2 * cos_sim
        S = torch.exp(-dist_sq / (2 * self.dpp_sigma ** 2))
        return S

    def compute_quality_vector(
        self,
        selected_lora_embs: List[torch.Tensor],
        concept_embs: List[torch.Tensor],
        full_prompt_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算 DPP 的质量向量 q，长度为 k:
            u_i = item_quality_score
            q_i = exp(u_i / dpp_tau)

        这里 q_i > 0，可作为 DPP 质量项。
        """
        qualities = []
        for lora_emb, concept_emb in zip(selected_lora_embs, concept_embs):
            u_i = self.compute_item_quality_score(
                lora_emb=lora_emb,
                concept_emb=concept_emb,
                full_prompt_emb=full_prompt_emb,
            )
            q_i = torch.exp(u_i / self.dpp_tau)
            qualities.append(q_i)

        q = torch.stack(qualities, dim=0)  # [k]
        return q

    def compute_dpp_kernel(
        self,
        selected_lora_embs: List[torch.Tensor],
        concept_embs: List[torch.Tensor],
        full_prompt_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        quality-aware DPP kernel:
            K_ij = q_i * S_ij * q_j

        如果 dpp_use_quality=False，则退化为：
            K = S
        """
        S = self.compute_similarity_kernel(selected_lora_embs)

        if not self.dpp_use_quality:
            # 为了兼容旧版行为，对角可以仍设为1
            K = S.clone()
            K.diagonal().fill_(1.0)
            return K

        q = self.compute_quality_vector(
            selected_lora_embs=selected_lora_embs,
            concept_embs=concept_embs,
            full_prompt_emb=full_prompt_emb,
        )  # [k]

        K = q[:, None] * S * q[None, :]
        return K

    def compute_dpp_score(
        self,
        selected_lora_embs: List[torch.Tensor],
        concept_embs: List[torch.Tensor],
        full_prompt_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        DPP score:
            gamma * log det(K + eps I)

        quality-aware DPP 时：
            K_ii = q_i^2
        """
        k = len(selected_lora_embs)
        device = selected_lora_embs[0].device

        if k <= 1:
            return torch.tensor(0.0, device=device)

        K = self.compute_dpp_kernel(
            selected_lora_embs=selected_lora_embs,
            concept_embs=concept_embs,
            full_prompt_emb=full_prompt_emb,
        )

        eps = 1e-5
        sign, logdet = torch.slogdet(K + eps * torch.eye(k, device=device))

        if sign.item() <= 0:
            return torch.tensor(0.0, device=device)

        return self.gamma * logdet

    # ------------------------------------------------------
    # 4. global score
    # ------------------------------------------------------
    def compute_global_score(
        self,
        selected_lora_embs: List[torch.Tensor],
        full_prompt_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        组合整体与完整 prompt 的对齐：
            delta * cos(mean(selected_loras), full_prompt)
        """
        combo_emb = torch.stack(selected_lora_embs, dim=0).mean(dim=0)
        combo_emb = F.normalize(combo_emb, dim=0)
        return self.delta * torch.dot(combo_emb, full_prompt_emb)

    # ------------------------------------------------------
    # total score
    # ------------------------------------------------------
    def compute_total_score(
        self,
        selected_lora_embs: List[torch.Tensor],
        concept_embs: List[torch.Tensor],
        full_prompt_emb: torch.Tensor,
    ) -> Dict[str, float]:
        individual = self.compute_individual_score(
            selected_lora_embs=selected_lora_embs,
            concept_embs=concept_embs,
            full_prompt_emb=full_prompt_emb,
        )

        pairwise = self.compute_pairwise_score(
            selected_lora_embs=selected_lora_embs,
        )

        dpp = self.compute_dpp_score(
            selected_lora_embs=selected_lora_embs,
            concept_embs=concept_embs,
            full_prompt_emb=full_prompt_emb,
        )

        global_score = self.compute_global_score(
            selected_lora_embs=selected_lora_embs,
            full_prompt_emb=full_prompt_emb,
        )

        total = individual + pairwise + dpp + global_score

        return {
            "total": total.item(),
            "individual": individual.item(),
            "pairwise": pairwise.item(),
            "dpp": dpp.item(),
            "global": global_score.item(),
        }

    # ------------------------------------------------------
    # enumerate combinations
    # ------------------------------------------------------
    def enumerate_combinations(
        self,
        candidate_items: List[List[Dict[str, Any]]],
        concept_embs: List[torch.Tensor],
        full_prompt_emb: torch.Tensor,
        top_pool_size: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        全枚举所有组合，精确计算总分，然后保留前 top_pool_size 个高分组合。
        """
        n_concepts = len(candidate_items)
        assert len(concept_embs) == n_concepts

        full_prompt_emb = self._normalize_vec(full_prompt_emb.to(self.device))
        concept_embs = [self._normalize_vec(x.to(self.device)) for x in concept_embs]

        normalized_candidates = []
        for concept_list in candidate_items:
            cur = []
            for item in concept_list:
                emb = self._normalize_vec(item["embedding"].to(self.device))
                new_item = dict(item)
                new_item["embedding"] = emb
                cur.append(new_item)
            normalized_candidates.append(cur)

        all_results = []
        index_ranges = [range(len(x)) for x in normalized_candidates]

        for combo_indices in itertools.product(*index_ranges):
            selected_items = [
                normalized_candidates[concept_idx][candidate_idx]
                for concept_idx, candidate_idx in enumerate(combo_indices)
            ]

            selected_embs = [x["embedding"] for x in selected_items]

            score_dict = self.compute_total_score(
                selected_lora_embs=selected_embs,
                concept_embs=concept_embs,
                full_prompt_emb=full_prompt_emb,
            )

            all_results.append({
                "indices": combo_indices,
                "items": selected_items,
                "score": score_dict["total"],
                "score_breakdown": score_dict,
            })

        all_results.sort(key=lambda x: x["score"], reverse=True)
        return all_results[:top_pool_size]

    # ------------------------------------------------------
    # diversified selection
    # ------------------------------------------------------
    def combo_embedding(self, combo: Dict[str, Any]) -> torch.Tensor:
        embs = [item["embedding"] for item in combo["items"]]
        emb = torch.stack(embs, dim=0).mean(dim=0)
        emb = F.normalize(emb, dim=0)
        return emb

    def overlap_similarity(self, combo_a: Dict[str, Any], combo_b: Dict[str, Any]) -> float:
        set_a = set(item["model_file"] for item in combo_a["items"])
        set_b = set(item["model_file"] for item in combo_b["items"])

        if len(set_a) == 0 and len(set_b) == 0:
            return 0.0

        return len(set_a & set_b) / max(len(set_a), len(set_b))

    def embedding_similarity(self, combo_a: Dict[str, Any], combo_b: Dict[str, Any]) -> float:
        if "_combo_emb" not in combo_a:
            combo_a["_combo_emb"] = self.combo_embedding(combo_a)
        if "_combo_emb" not in combo_b:
            combo_b["_combo_emb"] = self.combo_embedding(combo_b)

        return torch.dot(combo_a["_combo_emb"], combo_b["_combo_emb"]).item()

    def combo_similarity(self, combo_a: Dict[str, Any], combo_b: Dict[str, Any]) -> float:
        overlap_sim = self.overlap_similarity(combo_a, combo_b)
        emb_sim = self.embedding_similarity(combo_a, combo_b)
        return self.overlap_weight * overlap_sim + self.emb_weight * emb_sim

    def diversified_select(
        self,
        combos: List[Dict[str, Any]],
        final_top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        从高分候选池中选出多个高质量且彼此不同的组合。
        """
        if len(combos) <= final_top_k:
            return combos

        combos = sorted(combos, key=lambda x: x["score"], reverse=True)

        scores = torch.tensor([x["score"] for x in combos], dtype=torch.float)
        s_min = scores.min()
        s_max = scores.max()

        if (s_max - s_min).abs() < 1e-12:
            norm_scores = torch.ones_like(scores)
        else:
            norm_scores = (scores - s_min) / (s_max - s_min)

        for i in range(len(combos)):
            combos[i]["_norm_score"] = norm_scores[i].item()

        selected = [combos[0]]
        remaining = combos[1:]

        while len(selected) < final_top_k and len(remaining) > 0:
            best_idx = None
            best_mmr_score = -1e9

            for i, combo in enumerate(remaining):
                max_sim = max(self.combo_similarity(combo, s) for s in selected)

                mmr_score = (
                    self.lambda_quality * combo["_norm_score"]
                    - (1.0 - self.lambda_quality) * max_sim
                )

                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_idx = i

            chosen = remaining.pop(best_idx)
            chosen["_mmr_score"] = best_mmr_score
            selected.append(chosen)

        return selected

    # ------------------------------------------------------
    # optimize
    # ------------------------------------------------------
    def optimize(
        self,
        candidate_items: List[List[Dict[str, Any]]],
        concept_embs: List[torch.Tensor],
        full_prompt_emb: torch.Tensor,
        top_pool_size: int = 100,
        final_top_k: int = 5,
        return_all_pool: bool = False,
    ) -> Dict[str, Any]:
        candidate_pool = self.enumerate_combinations(
            candidate_items=candidate_items,
            concept_embs=concept_embs,
            full_prompt_emb=full_prompt_emb,
            top_pool_size=top_pool_size,
        )

        diverse_results = self.diversified_select(
            combos=candidate_pool,
            final_top_k=final_top_k,
        )

        result = {"diverse_results": diverse_results}
        if return_all_pool:
            result["candidate_pool"] = candidate_pool
        return result


# ==========================================================
# 3. direct pipeline for retrieval_results
# ==========================================================
class CombinationPipeline:
    """
    直接对接 call_lora() 的 retrieval_results 结构。
    """
    def __init__(
        self,
        lora_index_path: str,
        device: str = "cuda",

        alpha: float = 1.0,
        beta: float = 0.3,
        gamma: float = 0.2,
        delta: float = 0.5,

        lambda_concept: float = 1.0,
        lambda_prompt: float = 0.2,

        dpp_sigma: float = 0.5,
        dpp_tau: float = 0.2,
        dpp_use_quality: bool = True,

        lambda_quality: float = 0.8,
        overlap_weight: float = 0.7,
        emb_weight: float = 0.3,
    ):
        self.device = device

        self.clip_encoder = TextImageEncoder().to(device)
        self.clip_encoder.eval()

        self.index_store = LoRAIndexStore(
            index_path=lora_index_path,
            device=device,
        )

        self.optimizer = LoRACombinationOptimizer(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            delta=delta,
            lambda_concept=lambda_concept,
            lambda_prompt=lambda_prompt,
            dpp_sigma=dpp_sigma,
            dpp_tau=dpp_tau,
            dpp_use_quality=dpp_use_quality,
            lambda_quality=lambda_quality,
            overlap_weight=overlap_weight,
            emb_weight=emb_weight,
            device=device,
        )

    @torch.no_grad()
    def encode_text(self, text: str) -> torch.Tensor:
        emb = self.clip_encoder.encoding_text([text])[0]
        emb = F.normalize(emb, dim=0)
        return emb

    def build_candidate_items(
        self,
        data: Dict[str, Any],
        per_concept_top_k: Optional[int] = None,
    ):
        extract_concept = data["extract_concept"]
        retrieval_results = data["retrieval_results"]

        if "prompt" in data:
            full_prompt_text = data["prompt"]
        elif "rewrite_prompt" in data:
            full_prompt_text = data["rewrite_prompt"]
        else:
            parts = []
            for ec in extract_concept:
                keyword = ec["keyword"]
                desc = ec.get("retrieval_description", "")
                parts.append(f"{keyword}. {desc}")
            full_prompt_text = " ".join(parts)

        full_prompt_emb = self.encode_text(full_prompt_text)

        concept_names = []
        concept_embs = []
        candidate_items = []

        for ec in extract_concept:
            keyword = ec["keyword"]
            retrieval_description = ec.get("retrieval_description", "")

            concept_query = f"a '{keyword}' adapter. " + retrieval_description
            concept_emb = self.encode_text(concept_query)

            concept_names.append(keyword)
            concept_embs.append(concept_emb)

            cur_candidates = retrieval_results.get(keyword, [])
            if per_concept_top_k is not None:
                cur_candidates = cur_candidates[:per_concept_top_k]

            cur_items = []
            for item in cur_candidates:
                model_file = item["model_file"]
                lora_emb = self.index_store.get_embedding(model_file)
                if lora_emb is None:
                    continue

                cur_items.append({
                    "model_file": model_file,
                    "retrieval_score": item.get("score", None),
                    "embedding": lora_emb,
                })

            candidate_items.append(cur_items)

        return concept_names, candidate_items, concept_embs, full_prompt_emb

    def optimize_one_data(
        self,
        data: Dict[str, Any],
        per_concept_top_k: Optional[int] = None,
        top_pool_size: int = 100,
        final_top_k: int = 5,
        return_candidate_pool: bool = False,
    ) -> Dict[str, Any]:
        concept_names, candidate_items, concept_embs, full_prompt_emb = self.build_candidate_items(
            data=data,
            per_concept_top_k=per_concept_top_k,
        )

        if any(len(x) == 0 for x in candidate_items):
            data["combination_results"] = {
                "concept_names": concept_names,
                "diverse_combinations": [],
                "error": "some concepts have empty candidate list",
            }
            return data

        optimize_res = self.optimizer.optimize(
            candidate_items=candidate_items,
            concept_embs=concept_embs,
            full_prompt_emb=full_prompt_emb,
            top_pool_size=top_pool_size,
            final_top_k=final_top_k,
            return_all_pool=return_candidate_pool,
        )

        diverse_combinations = []
        for combo in optimize_res["diverse_results"]:
            diverse_combinations.append({
                "indices": list(combo["indices"]),
                "model_files": [x["model_file"] for x in combo["items"]],
                "retrieval_scores": [x.get("retrieval_score", None) for x in combo["items"]],
                "score": combo["score"],
                "score_breakdown": combo["score_breakdown"],
                "mmr_score": combo.get("_mmr_score", None),
            })

        result_dict = {
            "concept_names": concept_names,
            "diverse_combinations": diverse_combinations,
        }

        if return_candidate_pool:
            candidate_pool_json = []
            for combo in optimize_res["candidate_pool"]:
                candidate_pool_json.append({
                    "indices": list(combo["indices"]),
                    "model_files": [x["model_file"] for x in combo["items"]],
                    "retrieval_scores": [x.get("retrieval_score", None) for x in combo["items"]],
                    "score": combo["score"],
                    "score_breakdown": combo["score_breakdown"],
                })
            result_dict["candidate_pool"] = candidate_pool_json

        data["combination_results"] = result_dict
        return data


# ==========================================================
# 4. batch processing jsonl file
# ==========================================================
def optimize_retrieval_results_file(
    retrieval_result_path: str,
    lora_index_path: str,
    output_path: str,
    device: str = "cuda",

    per_concept_top_k: Optional[int] = 10,
    top_pool_size: int = 100,
    final_top_k: int = 5,

    alpha: float = 1.0,
    beta: float = 0.3,
    gamma: float = 0.2,
    delta: float = 0.5,

    lambda_concept: float = 1.0,
    lambda_prompt: float = 0.2,

    dpp_sigma: float = 0.5,
    dpp_tau: float = 0.2,
    dpp_use_quality: bool = True,

    lambda_quality: float = 0.8,
    overlap_weight: float = 0.7,
    emb_weight: float = 0.3,

    return_candidate_pool: bool = False,
):
    pipeline = CombinationPipeline(
        lora_index_path=lora_index_path,
        device=device,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        delta=delta,
        lambda_concept=lambda_concept,
        lambda_prompt=lambda_prompt,
        dpp_sigma=dpp_sigma,
        dpp_tau=dpp_tau,
        dpp_use_quality=dpp_use_quality,
        lambda_quality=lambda_quality,
        overlap_weight=overlap_weight,
        emb_weight=emb_weight,
    )

    with open(retrieval_result_path, "r", encoding="utf-8") as f:
        datas = [json.loads(line) for line in f]

    with open(output_path, "w", encoding="utf-8") as fout:
        for i, data in enumerate(datas):
            print(f"[Optimize] {i}/{len(datas)}")

            try:
                data = pipeline.optimize_one_data(
                    data=data,
                    per_concept_top_k=per_concept_top_k,
                    top_pool_size=top_pool_size,
                    final_top_k=final_top_k,
                    return_candidate_pool=return_candidate_pool,
                )
            except Exception as e:
                data["combination_results"] = {
                    "diverse_combinations": [],
                    "error": str(e),
                }

            fout.write(json.dumps(data, ensure_ascii=False) + "\n")

    print(f"[Done] saved optimized results to {output_path}")



if __name__ == "__main__":
    optimize_retrieval_results_file(
        retrieval_result_path="test_data/250_clipemb-7_all_res.jsonl",
        lora_index_path="lora_index_clipemb-7_all.pt",
        output_path="test_data/250_clipemb-7_all_res_combinations_reank.jsonl",
        device="cuda",
        per_concept_top_k=40,   # 每个 concept 取前10个候选做组合
        top_pool_size=120,      # 保留前100个高分组合作为候选池
        final_top_k=8,          # 最终输出5个多样化组合

        alpha=1.0,
        beta=0.3,
        gamma=0.2,
        delta=0.5,

        lambda_concept=1.0,
        lambda_prompt=0.3,


        dpp_sigma=0.5,
        dpp_tau=0.2,
        dpp_use_quality=True,

        lambda_quality=0.8,
        overlap_weight=0.7,
        emb_weight=0.3,

        return_candidate_pool=False,
    )
