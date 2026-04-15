import heapq
import itertools
import json
from typing import List, Dict, Any, Optional, Tuple

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
    - 后续相似度统一使用点积 = cosine similarity
    """
    def __init__(self, index_path: str, device: str = "cpu"):
        self.device = device

        index_data = torch.load(index_path, map_location="cpu")
        self.model_files = index_data["model_files"]
        self.embeddings = index_data["embeddings"].float()
        self.embeddings = F.normalize(self.embeddings, dim=-1)

        self.model_file_to_idx = {mf: i for i, mf in enumerate(self.model_files)}
        print(f"[LoRAIndexStore] loaded {len(self.model_files)} embeddings from {index_path}")

    def get_embedding(self, model_file: str) -> Optional[torch.Tensor]:
        idx = self.model_file_to_idx.get(model_file, None)
        if idx is None:
            return None
        return self.embeddings[idx].to(self.device)


# ==========================================================
# 2. Beam-search combination optimizer
# ==========================================================
class LoRACombinationOptimizer:
    """
    使用 Beam Search 的组合优化器。

    --------------------------------------------------------
    主思想
    --------------------------------------------------------
    我们把组合打分拆成两层：

    1) 搜索阶段 search_score（便宜）
       search_score =
           individual_score
         + pairwise_score

       用于 beam 扩展时快速筛掉差的 partial combinations。

    2) 最终阶段 final_score（完整）
       final_score =
           search_score
         + global_alignment

       对 beam 输出的完整组合再做精确重排。

    --------------------------------------------------------
    为什么这样设计
    --------------------------------------------------------
    - global alignment 是 set-level 的，放在 beam 每一步都算不划算
    - individual + pairwise 都有明确的增量形式，适合 beam 扩展
    - 最终只对有限个完整组合算 global，速度更稳

    --------------------------------------------------------
    目标函数
    --------------------------------------------------------
    total_score =
        individual_match
      + pairwise_utility
      + global_alignment

    其中：
    - individual_match:
        每个 concept 选出的 LoRA 是否匹配 concept / prompt
    - pairwise_utility:
        compatibility reward - redundancy penalty
    - global_alignment:
        组合整体与完整 prompt 的对齐
    """

    def __init__(
        self,
        # --------------------------------------------------
        # 主目标权重
        # --------------------------------------------------
        alpha: float = 1.0,          # individual match 权重
        beta_compat: float = 0.3,    # compatibility reward 权重
        eta_redund: float = 0.4,     # redundancy penalty 权重
        delta: float = 0.5,          # global alignment 权重

        # --------------------------------------------------
        # individual score 内部权重
        # match = lambda_concept * sim(lora, concept)
        #       + lambda_prompt  * sim(lora, full_prompt)
        # --------------------------------------------------
        lambda_concept: float = 1.0,
        lambda_prompt: float = 0.2,

        # --------------------------------------------------
        # global weighted fusion 温度
        # --------------------------------------------------
        global_tau: float = 0.2,

        # --------------------------------------------------
        # beam search 参数
        # --------------------------------------------------
        local_top_m: int = 40,       # 每个 concept 搜索前先保留多少候选
        beam_width: int = 200,       # beam 每层最多保留多少 partial combos
        final_pool_size: int = 200,  # 完整组合输出多少候选用于最终 rerank

        # --------------------------------------------------
        # diversified reranking (MMR) 参数
        # --------------------------------------------------
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

        self.local_top_m = local_top_m
        self.beam_width = beam_width
        self.final_pool_size = final_pool_size

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
    # item-level match
    # ------------------------------------------------------
    def compute_match_score(
        self,
        lora_emb: torch.Tensor,
        concept_emb: torch.Tensor,
        full_prompt_emb: torch.Tensor,
        retrieval_score: Optional[float] = None,
    ) -> torch.Tensor:
        """
        单个 LoRA 的局部匹配分数：

            match_i =
                lambda_concept * cos(lora_i, concept_i)
              + lambda_prompt  * cos(lora_i, full_prompt)

        说明：
        - 第一项保证和当前 concept 对齐
        - 第二项保证不要偏离完整 prompt 太远
        """
        local_sim = torch.dot(lora_emb, concept_emb)
        prompt_sim = torch.dot(lora_emb, full_prompt_emb)
        score = self.lambda_concept * local_sim + self.lambda_prompt * prompt_sim
        return score

    # ------------------------------------------------------
    # pairwise terms
    # ------------------------------------------------------
    def compute_redundancy(
        self,
        emb_i: torch.Tensor,
        emb_j: torch.Tensor,
    ) -> torch.Tensor:
        """
        冗余惩罚：
            redund(i, j) = max(0, cos(e_i, e_j))

        只惩罚相似/重复，不奖励“负相似”。
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
        兼容奖励（简化 proxy）：

            compat(i, j)
            = 0.5 * [ cos(e_i, prompt) + cos(e_j, prompt) ]

        含义：
        - 如果两个 LoRA 都对完整 prompt 比较友好，则更可能兼容。
        """
        prompt_fit_i = torch.dot(emb_i, full_prompt_emb)
        prompt_fit_j = torch.dot(emb_j, full_prompt_emb)
        compat = 0.5 * (prompt_fit_i + prompt_fit_j)
        return compat

    def compute_pair_utility(
        self,
        emb_i: torch.Tensor,
        emb_j: torch.Tensor,
        full_prompt_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        pair utility = compatibility reward - redundancy penalty

            pair_utility(i, j)
            = beta_compat * compat(i, j)
            - eta_redund * redund(i, j)
        """
        compat = self.compute_compatibility(
            emb_i=emb_i,
            emb_j=emb_j,
            full_prompt_emb=full_prompt_emb,
        )
        redund = self.compute_redundancy(
            emb_i=emb_i,
            emb_j=emb_j,
        )
        return self.beta_compat * compat - self.eta_redund * redund

    # ------------------------------------------------------
    # global alignment
    # ------------------------------------------------------
    def compute_weighted_combo_embedding(
        self,
        selected_items: List[Dict[str, Any]],
        concept_embs: List[torch.Tensor],
        full_prompt_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        组合 embedding（quality-weighted fusion）：

            w_i = softmax(match_i / global_tau)
            e_combo = normalize(sum_i w_i * e_i)

        比简单平均更稳定。
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

        E = torch.stack(embs, dim=0)
        item_scores = torch.stack(item_scores, dim=0)

        weights = torch.softmax(item_scores / self.global_tau, dim=0)
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
        """
        combo_emb = self.compute_weighted_combo_embedding(
            selected_items=selected_items,
            concept_embs=concept_embs,
            full_prompt_emb=full_prompt_emb,
        )
        return self.delta * torch.dot(combo_emb, full_prompt_emb)

    # ------------------------------------------------------
    # preprocessing / pruning
    # ------------------------------------------------------
    def preprocess_candidates(
        self,
        candidate_items: List[List[Dict[str, Any]]],
        concept_embs: List[torch.Tensor],
        full_prompt_emb: torch.Tensor,
    ) -> List[List[Dict[str, Any]]]:
        """
        对每个 concept 的候选做预处理：
        1. embedding normalize
        2. 预计算 item match score
        3. 按 item match score 排序
        4. 每个 concept 只保留前 local_top_m 个

        这样做可以显著减小 beam 扩展分支因子。
        """
        processed = []

        for concept_idx, items in enumerate(candidate_items):
            concept_emb = concept_embs[concept_idx]
            cur = []

            for item in items:
                emb = self._normalize_vec(item["embedding"].to(self.device))
                new_item = dict(item)
                new_item["embedding"] = emb

                match_score = self.compute_match_score(
                    lora_emb=emb,
                    concept_emb=concept_emb,
                    full_prompt_emb=full_prompt_emb,
                    retrieval_score=item.get("retrieval_score", None),
                )

                new_item["_match_score"] = float(match_score.item())
                cur.append(new_item)

            cur.sort(key=lambda x: x["_match_score"], reverse=True)

            if self.local_top_m is not None and self.local_top_m > 0:
                cur = cur[:self.local_top_m]

            processed.append(cur)

        return processed

    def reorder_concepts_for_search(
        self,
        candidate_items: List[List[Dict[str, Any]]],
        concept_embs: List[torch.Tensor],
        concept_names: Optional[List[str]] = None,
    ) -> Tuple[List[List[Dict[str, Any]]], List[torch.Tensor], Optional[List[str]], List[int]]:
        """
        可选：在 beam search 前调整 concept 的搜索顺序。

        一个简单且实用的策略：
        - 候选数量少的 concept 先扩展

        返回：
        - reordered_candidate_items
        - reordered_concept_embs
        - reordered_concept_names
        - order_map: 新顺序到原顺序的映射

        例如：
        原顺序 [0,1,2]
        若排序后 order_map = [2,0,1]
        表示新第0个concept对应原第2个concept
        """
        order = list(range(len(candidate_items)))
        order.sort(key=lambda i: len(candidate_items[i]))

        reordered_candidate_items = [candidate_items[i] for i in order]
        reordered_concept_embs = [concept_embs[i] for i in order]
        reordered_concept_names = [concept_names[i] for i in order] if concept_names is not None else None

        return reordered_candidate_items, reordered_concept_embs, reordered_concept_names, order

    # ------------------------------------------------------
    # beam search core
    # ------------------------------------------------------
    def initialize_beam(
        self,
        first_candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        用第一个 concept 的候选初始化 beam。

        每个 partial state 的结构：
        {
            "indices": [int],              # 每层候选在各自 concept list 内的索引
            "items":   [item, ...],        # 已选 items
            "search_score": float,         # 当前 beam 搜索分数 = individual + pairwise
        }

        这里初始状态只有一个 item，因此：
            search_score = alpha * item_match
        """
        beam = []

        for idx, item in enumerate(first_candidates):
            state = {
                "indices": [idx],
                "items": [item],
                "search_score": self.alpha * item["_match_score"],
            }
            beam.append(state)

        beam.sort(key=lambda x: x["search_score"], reverse=True)
        return beam[:self.beam_width]

    def expand_one_step(
        self,
        beam: List[Dict[str, Any]],
        next_candidates: List[Dict[str, Any]],
        full_prompt_emb: torch.Tensor,
    ) -> List[Dict[str, Any]]:
        """
        beam 扩展一层。

        假设当前 beam 中每个 state 已经选了 t 个 LoRA，
        现在给它加上第 t+1 个 concept 的某个候选 item。

        增量分数：
            delta =
                alpha * match(new_item)
              + sum_{old in state} pair_utility(old, new_item)

        然后：
            new_search_score = old_search_score + delta
        """
        new_states = []

        for state in beam:
            old_items = state["items"]
            old_score = state["search_score"]

            for next_idx, next_item in enumerate(next_candidates):
                delta = self.alpha * next_item["_match_score"]

                next_emb = next_item["embedding"]
                for old_item in old_items:
                    old_emb = old_item["embedding"]
                    delta += float(
                        self.compute_pair_utility(
                            emb_i=old_emb,
                            emb_j=next_emb,
                            full_prompt_emb=full_prompt_emb,
                        ).item()
                    )

                new_state = {
                    "indices": state["indices"] + [next_idx],
                    "items": old_items + [next_item],
                    "search_score": old_score + delta,
                }
                new_states.append(new_state)

        # 保留 top beam_width
        new_states.sort(key=lambda x: x["search_score"], reverse=True)
        return new_states[:self.beam_width]

    def beam_search_combinations(
        self,
        candidate_items: List[List[Dict[str, Any]]],
        concept_embs: List[torch.Tensor],
        full_prompt_emb: torch.Tensor,
    ) -> List[Dict[str, Any]]:
        """
        执行 beam search。

        返回的是完整组合，但此时它们只有 search_score，
        还没有加上 global score。
        """
        if len(candidate_items) == 0:
            return []

        beam = self.initialize_beam(candidate_items[0])

        for step in range(1, len(candidate_items)):
            beam = self.expand_one_step(
                beam=beam,
                next_candidates=candidate_items[step],
                full_prompt_emb=full_prompt_emb,
            )

        return beam

    # ------------------------------------------------------
    # final scoring
    # ------------------------------------------------------
    def finalize_combinations(
        self,
        beam_results: List[Dict[str, Any]],
        concept_embs: List[torch.Tensor],
        full_prompt_emb: torch.Tensor,
        top_pool_size: int,
    ) -> List[Dict[str, Any]]:
        """
        对 beam 输出的完整组合计算最终分数：

            final_score = search_score + global_score

        然后按 final_score 排序，保留前 top_pool_size 个。
        """
        results = []

        for state in beam_results:
            selected_items = state["items"]

            # 这里 individual / pairwise 可以直接从 search_score 恢复总量，
            # 但为了便于分析，我们仍显式计算一份 breakdown
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

            results.append({
                "indices": tuple(state["indices"]),
                "items": selected_items,
                "score": float(total.item()),
                "search_score": float(state["search_score"]),
                "score_breakdown": {
                    "total": float(total.item()),
                    "search_score": float(state["search_score"]),
                    "individual": float(individual.item()),
                    "pairwise": float(pairwise.item()),
                    "global": float(global_score.item()),
                }
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_pool_size]

    # ------------------------------------------------------
    # individual / pairwise explicit score (for final breakdown)
    # ------------------------------------------------------
    def compute_individual_score(
        self,
        selected_items: List[Dict[str, Any]],
        concept_embs: List[torch.Tensor],
        full_prompt_emb: torch.Tensor,
    ) -> torch.Tensor:
        score = torch.tensor(0.0, device=full_prompt_emb.device)

        for item, concept_emb in zip(selected_items, concept_embs):
            score += self.compute_match_score(
                lora_emb=item["embedding"],
                concept_emb=concept_emb,
                full_prompt_emb=full_prompt_emb,
                retrieval_score=item.get("retrieval_score", None),
            )

        return self.alpha * score

    def compute_pairwise_score(
        self,
        selected_items: List[Dict[str, Any]],
        full_prompt_emb: torch.Tensor,
    ) -> torch.Tensor:
        score = torch.tensor(0.0, device=full_prompt_emb.device)
        n = len(selected_items)

        for i in range(n):
            emb_i = selected_items[i]["embedding"]
            for j in range(i + 1, n):
                emb_j = selected_items[j]["embedding"]
                score += self.compute_pair_utility(
                    emb_i=emb_i,
                    emb_j=emb_j,
                    full_prompt_emb=full_prompt_emb,
                )

        return score

    # ------------------------------------------------------
    # result-level diversified reranking
    # ------------------------------------------------------
    def combo_embedding(
        self,
        combo: Dict[str, Any],
        concept_embs: Optional[List[torch.Tensor]] = None,
        full_prompt_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        计算组合自身 embedding，用于结果级 MMR。

        优先使用与主打分一致的 weighted combo embedding。
        """
        if concept_embs is not None and full_prompt_emb is not None:
            return self.compute_weighted_combo_embedding(
                selected_items=combo["items"],
                concept_embs=concept_embs,
                full_prompt_emb=full_prompt_emb,
            )

        embs = [item["embedding"] for item in combo["items"]]
        emb = torch.stack(embs, dim=0).mean(dim=0)
        emb = F.normalize(emb, dim=0)
        return emb

    def overlap_similarity(self, combo_a: Dict[str, Any], combo_b: Dict[str, Any]) -> float:
        """
        两个组合的离散重叠率。
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
        两个组合 embedding 的 cosine 相似度。
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
        组合间综合相似度：
            overlap_weight * overlap
          + emb_weight     * embedding_similarity
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
        用 MMR 从高分候选池里选出多个高质量且不同的组合。
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
        concept_names: Optional[List[str]] = None,
        top_pool_size: Optional[int] = None,
        final_top_k: int = 5,
        return_all_pool: bool = False,
        reorder_concepts: bool = True,
    ) -> Dict[str, Any]:
        """
        完整优化流程：

        1. normalize prompt / concepts
        2. preprocess candidates（预计算 item match，local prune）
        3. 可选：按候选数重排 concept 搜索顺序
        4. beam search 得到完整组合（只有 search_score）
        5. final scoring 加上 global alignment
        6. diversified reranking 得到多个不同结果
        """
        if top_pool_size is None:
            top_pool_size = self.final_pool_size

        full_prompt_emb = self._normalize_vec(full_prompt_emb.to(self.device))
        concept_embs = [self._normalize_vec(x.to(self.device)) for x in concept_embs]

        processed_candidates = self.preprocess_candidates(
            candidate_items=candidate_items,
            concept_embs=concept_embs,
            full_prompt_emb=full_prompt_emb,
        )

        if reorder_concepts:
            processed_candidates, concept_embs, concept_names, order_map = self.reorder_concepts_for_search(
                candidate_items=processed_candidates,
                concept_embs=concept_embs,
                concept_names=concept_names,
            )
        else:
            order_map = list(range(len(processed_candidates)))

        beam_results = self.beam_search_combinations(
            candidate_items=processed_candidates,
            concept_embs=concept_embs,
            full_prompt_emb=full_prompt_emb,
        )

        candidate_pool = self.finalize_combinations(
            beam_results=beam_results,
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

        result = {
            "diverse_results": diverse_results,
            "search_order_map": order_map,   # 新顺序 -> 原顺序
        }
        if return_all_pool:
            result["candidate_pool"] = candidate_pool
        return result


# ==========================================================
# 3. direct pipeline for retrieval_results
# ==========================================================
class CombinationPipeline:
    """
    直接对接 retrieval_results 的 pipeline。

    输入 data 约定包括：
    - extract_concept
    - retrieval_results
    - 可选 prompt / rewrite_prompt
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

        local_top_m: int = 40,
        beam_width: int = 200,
        final_pool_size: int = 200,

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
            beta_compat=beta_compat,
            eta_redund=eta_redund,
            delta=delta,
            lambda_concept=lambda_concept,
            lambda_prompt=lambda_prompt,
            global_tau=global_tau,
            local_top_m=local_top_m,
            beam_width=beam_width,
            final_pool_size=final_pool_size,
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
        把 retrieval_results 组织成:
        - concept_names
        - candidate_items
        - concept_embs
        - full_prompt_emb
        """
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
        top_pool_size: Optional[int] = None,
        final_top_k: int = 5,
        return_candidate_pool: bool = False,
        reorder_concepts: bool = True,
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
            concept_names=concept_names,
            top_pool_size=top_pool_size,
            final_top_k=final_top_k,
            return_all_pool=return_candidate_pool,
            reorder_concepts=reorder_concepts,
        )

        diverse_combinations = []
        for combo in optimize_res["diverse_results"]:
            diverse_combinations.append({
                "indices": list(combo["indices"]),
                "model_files": [x["model_file"] for x in combo["items"]],
                "retrieval_scores": [x.get("retrieval_score", None) for x in combo["items"]],
                "score": combo["score"],
                "search_score": combo.get("search_score", None),
                "score_breakdown": combo["score_breakdown"],
                "mmr_score": combo.get("_mmr_score", None),
            })

        result_dict = {
            "concept_names": concept_names,
            "diverse_combinations": diverse_combinations,
            "search_order_map": optimize_res.get("search_order_map", None),
        }

        if return_candidate_pool:
            candidate_pool_json = []
            for combo in optimize_res["candidate_pool"]:
                candidate_pool_json.append({
                    "indices": list(combo["indices"]),
                    "model_files": [x["model_file"] for x in combo["items"]],
                    "retrieval_scores": [x.get("retrieval_score", None) for x in combo["items"]],
                    "score": combo["score"],
                    "search_score": combo.get("search_score", None),
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

    per_concept_top_k: Optional[int] = 100,
    top_pool_size: Optional[int] = 200,
    final_top_k: int = 5,

    alpha: float = 1.0,
    beta_compat: float = 0.3,
    eta_redund: float = 0.4,
    delta: float = 0.5,

    lambda_concept: float = 1.0,
    lambda_prompt: float = 0.2,
    global_tau: float = 0.2,

    local_top_m: int = 40,
    beam_width: int = 200,
    final_pool_size: int = 200,

    lambda_quality: float = 0.8,
    overlap_weight: float = 0.7,
    emb_weight: float = 0.3,

    return_candidate_pool: bool = False,
    reorder_concepts: bool = True,
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
        local_top_m=local_top_m,
        beam_width=beam_width,
        final_pool_size=final_pool_size,
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
                    reorder_concepts=reorder_concepts,
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
        output_path="test_data/250_clipemb-7_all_res_combinations_reank_beam.jsonl",
        device="cuda",

        # retrieval 阶段每个 concept 先最多带多少候选进来
        per_concept_top_k=100,

        # final 输出相关
        top_pool_size=150,
        final_top_k=8,

        # scoring 权重
        alpha=1.0,
        beta_compat=0.3,
        eta_redund=0.4,
        delta=0.5,

        lambda_concept=1.0,
        lambda_prompt=0.3,
        global_tau=0.2,

        # beam search 超参
        local_top_m=25,     # 每个 concept 预裁剪到多少
        beam_width=200,     # 每一层 beam 保留多少 partial combos
        final_pool_size=150,

        # diversified reranking
        lambda_quality=0.8,
        overlap_weight=0.7,
        emb_weight=0.3,

        return_candidate_pool=False,
        reorder_concepts=True,
    )
