import itertools
import json
from typing import List, Dict, Any, Optional

import torch
import torch.nn.functional as F


# ==========================================================
# 1. LoRA embedding index
# ==========================================================
class LoRAIndexStore:
    """
    从离线构建好的 LoRA embedding 索引中加载:
    - model_files: List[str]
    - embeddings:  Tensor [N, D]

    并提供:
    - model_file -> embedding 的查询能力

    约定:
    - embeddings 在加载后会做 L2 normalize
    - 后续所有相似度默认用点积 = cosine similarity
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
# 2. Combination optimizer
# ==========================================================
class LoRACombinationOptimizer:
    """
    组合打分目标（推荐版本）：

        total_score
        = individual_match
        + pairwise_utility
        + global_alignment

    其中：

    1) individual_match
       每个 concept 各选一个 LoRA，要求该 LoRA:
       - 与本 concept 相匹配
       - 同时不要过度偏离完整 prompt

    2) pairwise_utility
       对于组合中的每一对 LoRA:
       - reward: compatibility（两者都与完整 prompt 友好）
       - penalty: redundancy（两者太相似则惩罚）

    3) global_alignment
       用 quality-weighted 的组合 embedding 与完整 prompt 做对齐，
       防止“局部各自合理，但整体跑偏”。

    注意：
    - 不再把 DPP 放在主目标里
    - 多组合结果的多样性由后处理 MMR diversified_select 负责
    """

    def __init__(
        self,
        # -------------------------------
        # 主目标三项权重
        # -------------------------------
        alpha: float = 1.0,          # individual match 权重
        beta_compat: float = 0.3,    # pairwise compatibility reward 权重
        eta_redund: float = 0.4,     # pairwise redundancy penalty 权重
        delta: float = 0.5,          # global alignment 权重

        # -------------------------------
        # individual match 内部权重
        # score_i = lambda_concept * sim(lora_i, concept_i)
        #         + lambda_prompt  * sim(lora_i, full_prompt)
        # -------------------------------
        lambda_concept: float = 1.0,
        lambda_prompt: float = 0.2,

        # -------------------------------
        # global alignment 中，
        # 由 item match score 做 softmax 融合时的温度
        # -------------------------------
        global_tau: float = 0.2,

        # -------------------------------
        # diversified reranking (MMR) 参数
        # 用于最终返回多个高质量且不同的组合
        # -------------------------------
        lambda_quality: float = 0.8,
        overlap_weight: float = 0.7,
        emb_weight: float = 0.3,

        device: str = "cpu",
    ):
        self.alpha = alpha
        self.beta_compat = beta_compat
        self.eta_redund = eta_redund
        self.delta = delta

        self.lambda_concept = lambda_concept
        self.lambda_prompt = lambda_prompt
        self.global_tau = global_tau

        self.lambda_quality = lambda_quality
        self.overlap_weight = overlap_weight
        self.emb_weight = emb_weight

        self.device = device

    # ------------------------------------------------------
    # utility
    # ------------------------------------------------------
    def _normalize_vec(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=0)

    # ------------------------------------------------------
    # 1. item-level match score
    # ------------------------------------------------------
    def compute_match_score(
        self,
        lora_emb: torch.Tensor,
        concept_emb: torch.Tensor,
        full_prompt_emb: torch.Tensor,
        retrieval_score: Optional[float] = None,
    ) -> torch.Tensor:
        """
        单个 LoRA 对其 concept 的匹配分数。

        当前定义：
            match_i =
                lambda_concept * cos(lora_i, concept_i)
              + lambda_prompt  * cos(lora_i, full_prompt)

        解释：
        - 第一项：确保 LoRA 对当前 concept 有局部匹配能力
        - 第二项：避免 LoRA 虽然和 concept 相近，但整体 prompt 上偏得太远

        备注：
        - retrieval_score 暂时不直接纳入公式
        - 如果以后想融合 retrieval score，可在这里扩展
        """
        local_sim = torch.dot(lora_emb, concept_emb)
        prompt_sim = torch.dot(lora_emb, full_prompt_emb)
        score = self.lambda_concept * local_sim + self.lambda_prompt * prompt_sim
        return score

    def compute_individual_score(
        self,
        selected_items: List[Dict[str, Any]],
        concept_embs: List[torch.Tensor],
        full_prompt_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        组合中每个 LoRA 的 match 分数求和：
            individual = alpha * sum_i match_i
        """
        score = torch.tensor(0.0, device=full_prompt_emb.device)

        for item, concept_emb in zip(selected_items, concept_embs):
            lora_emb = item["embedding"]
            retrieval_score = item.get("retrieval_score", None)

            score += self.compute_match_score(
                lora_emb=lora_emb,
                concept_emb=concept_emb,
                full_prompt_emb=full_prompt_emb,
                retrieval_score=retrieval_score,
            )

        return self.alpha * score

    # ------------------------------------------------------
    # 2. pairwise utility
    # ------------------------------------------------------
    def compute_redundancy(
        self,
        emb_i: torch.Tensor,
        emb_j: torch.Tensor,
    ) -> torch.Tensor:
        """
        冗余惩罚项。

        定义：
            redund(i, j) = max(0, cos(e_i, e_j))

        解释：
        - 如果两个 LoRA 很相似，则认为功能上可能重复，惩罚增大
        - 只惩罚正相似，不奖励“负相似”
          （我们不希望为了“拉开距离”而鼓励两个 LoRA 彼此对着干）
        """
        sim = torch.dot(emb_i, emb_j)
        return torch.relu(sim)

    def compute_compatibility(
        self,
        emb_i: torch.Tensor,
        emb_j: torch.Tensor,
        full_prompt_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        兼容奖励项（当前的简化稳健版本）。

        定义：
            compat(i, j) =
                0.5 * [ cos(e_i, prompt) + cos(e_j, prompt) ]

        解释：
        - 如果两个 LoRA 都与完整 prompt 保持较好对齐，
          则它们更可能是“组合友好”的。
        - 这个 compat 并不是显式建模真实生成交互，
          而是一个低成本 proxy。
        - 冗余由 redundancy 单独惩罚，这里不再重复扣 overlap。
        """
        prompt_fit_i = torch.dot(emb_i, full_prompt_emb)
        prompt_fit_j = torch.dot(emb_j, full_prompt_emb)
        compat = 0.5 * (prompt_fit_i + prompt_fit_j)
        return compat

    def compute_pairwise_score(
        self,
        selected_items: List[Dict[str, Any]],
        full_prompt_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        组合中的成对 utility：

            pairwise
            = sum_{i<j} [
                beta_compat * compat(i, j)
                - eta_redund * redund(i, j)
              ]

        解释：
        - compat: 奖励“都对 prompt 友好”的 LoRA 对
        - redund: 惩罚功能重复/风格高度重合的 LoRA 对
        """
        score = torch.tensor(0.0, device=full_prompt_emb.device)
        n = len(selected_items)

        for i in range(n):
            emb_i = selected_items[i]["embedding"]
            for j in range(i + 1, n):
                emb_j = selected_items[j]["embedding"]

                compat = self.compute_compatibility(
                    emb_i=emb_i,
                    emb_j=emb_j,
                    full_prompt_emb=full_prompt_emb,
                )
                redund = self.compute_redundancy(
                    emb_i=emb_i,
                    emb_j=emb_j,
                )

                score += self.beta_compat * compat - self.eta_redund * redund

        return score

    # ------------------------------------------------------
    # 3. global alignment
    # ------------------------------------------------------
    def compute_weighted_combo_embedding(
        self,
        selected_items: List[Dict[str, Any]],
        concept_embs: List[torch.Tensor],
        full_prompt_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算组合 embedding（quality-weighted fusion）。

        步骤：
        1. 对每个 LoRA 先算 item match score
        2. 对这些 score 做 softmax 得到权重
        3. 加权求和得到组合 embedding

        公式：
            w_i = softmax(match_i / global_tau)
            e_combo = normalize(sum_i w_i * e_i)

        解释：
        - 比简单平均更合理
        - 让质量更高、对当前 prompt / concept 更匹配的 LoRA
          在组合表示中占更大权重
        """
        embs = []
        item_scores = []

        for item, concept_emb in zip(selected_items, concept_embs):
            emb = item["embedding"]
            match_score = self.compute_match_score(
                lora_emb=emb,
                concept_emb=concept_emb,
                full_prompt_emb=full_prompt_emb,
                retrieval_score=item.get("retrieval_score", None),
            )
            embs.append(emb)
            item_scores.append(match_score)

        E = torch.stack(embs, dim=0)                    # [k, D]
        item_scores = torch.stack(item_scores, dim=0)   # [k]

        weights = torch.softmax(item_scores / self.global_tau, dim=0)   # [k]
        combo_emb = (weights[:, None] * E).sum(dim=0)
        combo_emb = F.normalize(combo_emb, dim=0)
        return combo_emb

    def compute_global_score(
        self,
        selected_items: List[Dict[str, Any]],
        concept_embs: List[torch.Tensor],
        full_prompt_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        全局对齐分数：

            global = delta * cos(combo_emb, full_prompt)

        其中 combo_emb 使用 quality-weighted fusion 得到。
        """
        combo_emb = self.compute_weighted_combo_embedding(
            selected_items=selected_items,
            concept_embs=concept_embs,
            full_prompt_emb=full_prompt_emb,
        )
        return self.delta * torch.dot(combo_emb, full_prompt_emb)

    # ------------------------------------------------------
    # total score
    # ------------------------------------------------------
    def compute_total_score(
        self,
        selected_items: List[Dict[str, Any]],
        concept_embs: List[torch.Tensor],
        full_prompt_emb: torch.Tensor,
    ) -> Dict[str, float]:
        """
        组合总分：
            total = individual + pairwise + global
        """
        individual = self.compute_individual_score(
            selected_items=selected_items,
            concept_embs=concept_embs,
            full_prompt_emb=full_prompt_emb,
        )

        pairwise = self.compute_pairwise_score(
            selected_items=selected_items,
            full_prompt_emb=full_prompt_emb,
        )

        global_score = self.compute_global_score(
            selected_items=selected_items,
            concept_embs=concept_embs,
            full_prompt_emb=full_prompt_emb,
        )

        total = individual + pairwise + global_score

        return {
            "total": total.item(),
            "individual": individual.item(),
            "pairwise": pairwise.item(),
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
        全枚举所有组合，精确计算主目标总分，然后保留前 top_pool_size 个高分组合。

        适用场景：
        - concept 数量一般为 2~3
        - 每个 concept 候选数一般为 10~40
        - 这种规模下完全枚举通常可接受
        """
        n_concepts = len(candidate_items)
        assert len(concept_embs) == n_concepts

        full_prompt_emb = self._normalize_vec(full_prompt_emb.to(self.device))
        concept_embs = [self._normalize_vec(x.to(self.device)) for x in concept_embs]

        # 预先规范化 candidate embedding
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

            score_dict = self.compute_total_score(
                selected_items=selected_items,
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
    # diversified selection (result-level diversity)
    # ------------------------------------------------------
    def combo_embedding(
        self,
        combo: Dict[str, Any],
        concept_embs: Optional[List[torch.Tensor]] = None,
        full_prompt_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        计算一个组合自身的 embedding，用于结果级 reranking。

        这里尽量与主打分中的 global embedding 保持一致：
        - 如果提供 concept_embs + full_prompt_emb，则使用 weighted fusion
        - 否则退化为简单平均（兼容旧逻辑）
        """
        if concept_embs is not None and full_prompt_emb is not None:
            emb = self.compute_weighted_combo_embedding(
                selected_items=combo["items"],
                concept_embs=concept_embs,
                full_prompt_emb=full_prompt_emb,
            )
            return emb

        embs = [item["embedding"] for item in combo["items"]]
        emb = torch.stack(embs, dim=0).mean(dim=0)
        emb = F.normalize(emb, dim=0)
        return emb

    def overlap_similarity(self, combo_a: Dict[str, Any], combo_b: Dict[str, Any]) -> float:
        """
        组合间的离散重叠度：
            overlap = |A ∩ B| / max(|A|, |B|)
        """
        set_a = set(item["model_file"] for item in combo_a["items"])
        set_b = set(item["model_file"] for item in combo_b["items"])

        if len(set_a) == 0 and len(set_b) == 0:
            return 0.0

        return len(set_a & set_b) / max(len(set_a), len(set_b))

    def embedding_similarity(
        self,
        combo_a: Dict[str, Any],
        combo_b: Dict[str, Any],
        concept_embs: Optional[List[torch.Tensor]] = None,
        full_prompt_emb: Optional[torch.Tensor] = None,
    ) -> float:
        """
        组合 embedding 相似度。
        """
        if "_combo_emb" not in combo_a:
            combo_a["_combo_emb"] = self.combo_embedding(
                combo_a,
                concept_embs=concept_embs,
                full_prompt_emb=full_prompt_emb,
            )
        if "_combo_emb" not in combo_b:
            combo_b["_combo_emb"] = self.combo_embedding(
                combo_b,
                concept_embs=concept_embs,
                full_prompt_emb=full_prompt_emb,
            )

        return torch.dot(combo_a["_combo_emb"], combo_b["_combo_emb"]).item()

    def combo_similarity(
        self,
        combo_a: Dict[str, Any],
        combo_b: Dict[str, Any],
        concept_embs: Optional[List[torch.Tensor]] = None,
        full_prompt_emb: Optional[torch.Tensor] = None,
    ) -> float:
        """
        组合间综合相似度 = overlap + embedding similarity
        用于 diversified reranking 的相似度项。
        """
        overlap_sim = self.overlap_similarity(combo_a, combo_b)
        emb_sim = self.embedding_similarity(
            combo_a,
            combo_b,
            concept_embs=concept_embs,
            full_prompt_emb=full_prompt_emb,
        )
        return self.overlap_weight * overlap_sim + self.emb_weight * emb_sim

    def diversified_select(
        self,
        combos: List[Dict[str, Any]],
        final_top_k: int = 5,
        concept_embs: Optional[List[torch.Tensor]] = None,
        full_prompt_emb: Optional[torch.Tensor] = None,
    ) -> List[Dict[str, Any]]:
        """
        从高分候选池中进一步选出多个高质量且彼此不同的组合。

        使用 MMR:
            mmr = lambda_quality * quality
                - (1-lambda_quality) * max_similarity_to_selected

        这样主目标负责“单个组合质量”，
        reranking 负责“多个结果之间多样化”。
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
                max_sim = max(
                    self.combo_similarity(
                        combo,
                        s,
                        concept_embs=concept_embs,
                        full_prompt_emb=full_prompt_emb,
                    )
                    for s in selected
                )

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
        """
        完整优化流程：
        1. 全枚举所有组合并打主分
        2. 取前 top_pool_size 个作为候选池
        3. 用 MMR 选出 final_top_k 个彼此不同的高质量组合
        """
        candidate_pool = self.enumerate_combinations(
            candidate_items=candidate_items,
            concept_embs=concept_embs,
            full_prompt_emb=full_prompt_emb,
            top_pool_size=top_pool_size,
        )

        diverse_results = self.diversified_select(
            combos=candidate_pool,
            final_top_k=final_top_k,
            concept_embs=concept_embs,
            full_prompt_emb=full_prompt_emb,
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
    直接对接 retrieval_results 结构。

    输入 data 约定包括：
    - extract_concept
    - retrieval_results
    - 可选 prompt / rewrite_prompt

    流程：
    1. 编码 full prompt
    2. 编码每个 concept query
    3. 从 retrieval_results 中取候选 LoRA
    4. 从 LoRAIndexStore 中取对应 LoRA embedding
    5. 交给 LoRACombinationOptimizer 做组合优化
    """
    def __init__(
        self,
        lora_index_path: str,
        device: str = "cuda",

        alpha: float = 1.0,
        beta_compat: float = 0.3,
        eta_redund: float = 0.4,
        delta: float = 0.5,

        lambda_concept: float = 1.0,
        lambda_prompt: float = 0.2,
        global_tau: float = 0.2,

        lambda_quality: float = 0.8,
        overlap_weight: float = 0.7,
        emb_weight: float = 0.3,
    ):
        self.device = device

        # 这里默认继续用现有文本编码器
        # 要求它有 encoding_text([str]) -> [B, D]
        self.clip_encoder = TextImageEncoder().to(device)
        self.clip_encoder.eval()

        self.index_store = LoRAIndexStore(
            index_path=lora_index_path,
            device=device,
        )

        self.optimizer = LoRACombinationOptimizer(
            alpha=alpha,
            beta_compat=beta_compat,
            eta_redund=eta_redund,
            delta=delta,
            lambda_concept=lambda_concept,
            lambda_prompt=lambda_prompt,
            global_tau=global_tau,
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
        """
        将 retrieval_results 转为优化器需要的候选结构：
        candidate_items: List[List[item]]

        item 至少包含：
        - model_file
        - retrieval_score
        - embedding
        """
        extract_concept = data["extract_concept"]
        retrieval_results = data["retrieval_results"]

        # 优先使用原始 prompt / rewrite_prompt
        if "prompt" in data:
            full_prompt_text = data["prompt"]
        elif "rewrite_prompt" in data:
            full_prompt_text = data["rewrite_prompt"]
        else:
            # fallback: 用 concept 拼一个弱替代
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

            # concept query 文本可按需要继续优化
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
    beta_compat: float = 0.3,
    eta_redund: float = 0.4,
    delta: float = 0.5,

    lambda_concept: float = 1.0,
    lambda_prompt: float = 0.2,
    global_tau: float = 0.2,

    lambda_quality: float = 0.8,
    overlap_weight: float = 0.7,
    emb_weight: float = 0.3,

    return_candidate_pool: bool = False,
):
    pipeline = CombinationPipeline(
        lora_index_path=lora_index_path,
        device=device,
        alpha=alpha,
        beta_compat=beta_compat,
        eta_redund=eta_redund,
        delta=delta,
        lambda_concept=lambda_concept,
        lambda_prompt=lambda_prompt,
        global_tau=global_tau,
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


# ==========================================================
# 5. example
# ==========================================================
if __name__ == "__main__":
    optimize_retrieval_results_file(
        retrieval_result_path="test_data/250_clipemb-7_all_res.jsonl",
        lora_index_path="lora_index_clipemb-7_all.pt",
        output_path="test_data/250_clipemb-7_all_res_combinations_reank.jsonl",
        device="cuda",

        per_concept_top_k=40,   # 每个 concept 前多少个候选用于组合
        top_pool_size=120,      # 主目标打分后保留多少高分组合作为候选池
        final_top_k=8,          # 最终输出多少个多样化组合

        alpha=1.0,
        beta_compat=0.3,
        eta_redund=0.4,
        delta=0.5,

        lambda_concept=1.0,
        lambda_prompt=0.3,
        global_tau=0.2,

        lambda_quality=0.8,
        overlap_weight=0.7,
        emb_weight=0.3,

        return_candidate_pool=False,
    )
