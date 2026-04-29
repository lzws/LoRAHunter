import json
import itertools
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.nn.functional as F

from models import TextImageEncoder, QwenVLEncoder




# ==========================================================
# 1. LoRA index store
# ==========================================================
class LoRAIndexStore:
    def __init__(self, index_path: str, device: str = "cpu"):
        self.device = device

        index_data = torch.load(index_path, map_location="cpu")
        self.model_files = index_data["model_files"]
        self.is_dual = index_data.get("is_dual", False)

        if "txt_embeddings" in index_data:
            self.txt_embeddings = index_data["txt_embeddings"].to(torch.bfloat16)
        elif "embeddings" in index_data:
            self.txt_embeddings = index_data["embeddings"].to(torch.bfloat16)
        else:
            raise KeyError("Index file must contain 'txt_embeddings' or 'embeddings'")

        self.txt_embeddings = F.normalize(self.txt_embeddings, dim=-1)

        self.img_embeddings = None
        if self.is_dual:
            if "img_embeddings" not in index_data:
                raise KeyError("Dual index must contain 'img_embeddings'")
            self.img_embeddings = index_data["img_embeddings"].to(torch.bfloat16)
            self.img_embeddings = F.normalize(self.img_embeddings, dim=-1)

        self.model_file_to_idx = {mf: i for i, mf in enumerate(self.model_files)}

        print(f"[LoRAIndexStore] loaded {len(self.model_files)} items from {index_path}")
        print(f"[LoRAIndexStore] is_dual={self.is_dual}")

    def get_embedding(self, model_file: str) -> Optional[Dict[str, torch.Tensor]]:
        idx = self.model_file_to_idx.get(model_file, None)
        if idx is None:
            return None

        item = {
            "txt_embedding": self.txt_embeddings[idx].to(self.device),
            "img_embedding": None,
            "is_dual": self.is_dual,
        }

        if self.img_embeddings is not None:
            item["img_embedding"] = self.img_embeddings[idx].to(self.device)

        return item


# ==========================================================
# 2. Hybrid optimizer: exhaustive for <=3, hierarchical for >3
# ==========================================================
class HybridCombinationDiverseSelector:
    def __init__(
        self,
        alpha: float = 1.0,
        beta_compat: float = 0.3,
        eta_redund: float = 0.4,
        delta: float = 0.5,

        lambda_concept: float = 1.0,
        lambda_prompt: float = 0.7,

        global_tau: float = 0.2,
        route_text_ratio: float = 1.0,
        device: str = "cpu",

        redundancy_tau: float = 0.3,
        compat_sim_suppress: float = 0.5,
        global_mix_uniform: float = 0.3,

        # 当 concept 数 <= 这个值时，用全枚举
        exhaustive_max_concepts: int = 3,

        # hierarchical merge 时每层保留多少子组合
        merge_pool_size: int = 500,

        # diversity 参数
        diversity_quality_weight: float = 0.7,
        diversity_sim_weight: float = 0.15,
        diversity_repeat_weight: float = 0.15,

        sim_gamma: float = 0.7,
        max_repeat_per_slot: Optional[int] = 2,
    ):
        self.alpha = alpha
        self.beta_compat = beta_compat
        self.eta_redund = eta_redund
        self.delta = delta

        self.lambda_concept = lambda_concept
        self.lambda_prompt = lambda_prompt
        self.global_tau = global_tau
        self.route_text_ratio = route_text_ratio
        self.device = device

        self.redundancy_tau = redundancy_tau
        self.compat_sim_suppress = compat_sim_suppress
        self.global_mix_uniform = global_mix_uniform

        self.exhaustive_max_concepts = exhaustive_max_concepts
        self.merge_pool_size = merge_pool_size

        self.diversity_quality_weight = diversity_quality_weight
        self.diversity_sim_weight = diversity_sim_weight
        self.diversity_repeat_weight = diversity_repeat_weight

        self.sim_gamma = sim_gamma
        self.max_repeat_per_slot = max_repeat_per_slot

    # ------------------------------------------------------
    # basic utils
    # ------------------------------------------------------
    def _normalize_vec(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=0)

    def _normalize_item_embedding(self, item: Dict[str, Any]) -> Dict[str, Any]:
        new_item = dict(item)

        if new_item.get("txt_embedding", None) is not None:
            new_item["txt_embedding"] = self._normalize_vec(new_item["txt_embedding"].to(self.device))

        if new_item.get("img_embedding", None) is not None:
            new_item["img_embedding"] = self._normalize_vec(new_item["img_embedding"].to(self.device))

        return new_item

    def _route_query_to_lora_score(
        self,
        query_emb: torch.Tensor,
        item: Dict[str, Any],
    ) -> torch.Tensor:
        txt_emb = item["txt_embedding"]
        img_emb = item.get("img_embedding", None)

        txt_score = torch.dot(query_emb, txt_emb)

        if img_emb is None:
            return txt_score

        img_score = torch.dot(query_emb, img_emb)
        return self.route_text_ratio * txt_score + (1.0 - self.route_text_ratio) * img_score

    def _item_to_item_similarity(
        self,
        item_a: Dict[str, Any],
        item_b: Dict[str, Any],
    ) -> torch.Tensor:
        txt_a = item_a["txt_embedding"]
        txt_b = item_b["txt_embedding"]
        txt_sim = torch.dot(txt_a, txt_b)

        img_a = item_a.get("img_embedding", None)
        img_b = item_b.get("img_embedding", None)

        if img_a is None or img_b is None:
            return txt_sim

        img_sim = torch.dot(img_a, img_b)
        return self.route_text_ratio * txt_sim + (1.0 - self.route_text_ratio) * img_sim

    def _get_fused_item_embedding(self, item: Dict[str, Any]) -> torch.Tensor:
        txt_emb = item["txt_embedding"]
        img_emb = item.get("img_embedding", None)

        if img_emb is None:
            return txt_emb

        if txt_emb.shape[-1] != img_emb.shape[-1]:
            return txt_emb

        fused = self.route_text_ratio * txt_emb + (1.0 - self.route_text_ratio) * img_emb
        fused = F.normalize(fused, dim=0)
        return fused

    # ------------------------------------------------------
    # scoring functions
    # ------------------------------------------------------
    def compute_match_score(
        self,
        item: Dict[str, Any],
        concept_emb: torch.Tensor,
        full_prompt_emb: torch.Tensor,
        retrieval_score: Optional[float] = None,
    ) -> torch.Tensor:
        local_sim = self._route_query_to_lora_score(concept_emb, item)
        prompt_sim = self._route_query_to_lora_score(full_prompt_emb, item)

        score = self.lambda_concept * local_sim + self.lambda_prompt * (local_sim * prompt_sim)
        return score

    def compute_redundancy(
        self,
        item_i: Dict[str, Any],
        item_j: Dict[str, Any],
    ) -> torch.Tensor:
        sim = self._item_to_item_similarity(item_i, item_j)
        return torch.relu(sim - self.redundancy_tau)

    def compute_compatibility(
        self,
        item_i: Dict[str, Any],
        item_j: Dict[str, Any],
        full_prompt_emb: torch.Tensor,
    ) -> torch.Tensor:
        prompt_fit_i = self._route_query_to_lora_score(full_prompt_emb, item_i)
        prompt_fit_j = self._route_query_to_lora_score(full_prompt_emb, item_j)

        pair_prompt_fit = torch.minimum(prompt_fit_i, prompt_fit_j)
        sim_ij = self._item_to_item_similarity(item_i, item_j)

        compat = pair_prompt_fit * (1.0 - self.compat_sim_suppress * torch.relu(sim_ij))
        return compat

    def compute_pair_utility(
        self,
        item_i: Dict[str, Any],
        item_j: Dict[str, Any],
        full_prompt_emb: torch.Tensor,
    ) -> torch.Tensor:
        compat = self.compute_compatibility(item_i, item_j, full_prompt_emb)
        redund = self.compute_redundancy(item_i, item_j)
        return self.beta_compat * compat - self.eta_redund * redund

    def compute_weighted_combo_embedding(
        self,
        selected_items: List[Dict[str, Any]],
        selected_concept_embs: List[torch.Tensor],
        full_prompt_emb: torch.Tensor,
    ) -> torch.Tensor:
        fused_embs = []
        item_scores = []

        for item, concept_emb in zip(selected_items, selected_concept_embs):
            fused_emb = self._get_fused_item_embedding(item)
            match_score = self.compute_match_score(
                item=item,
                concept_emb=concept_emb,
                full_prompt_emb=full_prompt_emb,
                retrieval_score=item.get("retrieval_score", None),
            )
            fused_embs.append(fused_emb)
            item_scores.append(match_score)

        E = torch.stack(fused_embs, dim=0)
        item_scores = torch.stack(item_scores, dim=0)

        soft_weights = torch.softmax(item_scores / self.global_tau, dim=0)
        uniform_weights = torch.ones_like(soft_weights) / soft_weights.numel()
        weights = (1.0 - self.global_mix_uniform) * soft_weights + self.global_mix_uniform * uniform_weights

        combo_emb = (weights[:, None] * E).sum(dim=0)
        combo_emb = F.normalize(combo_emb, dim=0)
        return combo_emb

    def compute_full_combination_score(
        self,
        selected_items: List[Dict[str, Any]],
        selected_concept_ids: List[int],
        concept_embs_all: List[torch.Tensor],
        full_prompt_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float], torch.Tensor]:
        selected_concept_embs = [concept_embs_all[i] for i in selected_concept_ids]

        individual = torch.tensor(0.0, device=full_prompt_emb.device)
        for item, concept_emb in zip(selected_items, selected_concept_embs):
            individual += self.compute_match_score(
                item=item,
                concept_emb=concept_emb,
                full_prompt_emb=full_prompt_emb,
                retrieval_score=item.get("retrieval_score", None),
            )
        individual = self.alpha * individual

        pairwise = torch.tensor(0.0, device=full_prompt_emb.device)
        n = len(selected_items)
        for i in range(n):
            for j in range(i + 1, n):
                pairwise += self.compute_pair_utility(
                    item_i=selected_items[i],
                    item_j=selected_items[j],
                    full_prompt_emb=full_prompt_emb,
                )

        combo_emb = self.compute_weighted_combo_embedding(
            selected_items=selected_items,
            selected_concept_embs=selected_concept_embs,
            full_prompt_emb=full_prompt_emb,
        )
        global_score = self.delta * torch.dot(combo_emb, full_prompt_emb)

        total = individual + pairwise + global_score

        breakdown = {
            "total": float(total.item()),
            "individual": float(individual.item()),
            "pairwise": float(pairwise.item()),
            "global": float(global_score.item()),
        }

        return total, breakdown, combo_emb

    # ------------------------------------------------------
    # preprocess
    # ------------------------------------------------------
    def preprocess_candidates(
        self,
        candidate_items: List[List[Dict[str, Any]]],
        concept_embs: List[torch.Tensor],
        full_prompt_emb: torch.Tensor,
    ) -> List[List[Dict[str, Any]]]:
        processed = []

        for concept_idx, items in enumerate(candidate_items):
            concept_emb = concept_embs[concept_idx]
            cur = []

            for item in items:
                new_item = self._normalize_item_embedding(item)
                match_score = self.compute_match_score(
                    item=new_item,
                    concept_emb=concept_emb,
                    full_prompt_emb=full_prompt_emb,
                    retrieval_score=item.get("retrieval_score", None),
                )
                new_item["_match_score"] = float(match_score.item())
                cur.append(new_item)

            cur.sort(key=lambda x: x["_match_score"], reverse=True)
            processed.append(cur)

        return processed

    # ------------------------------------------------------
    # exhaustive search
    # ------------------------------------------------------
    def enumerate_all_combinations(
        self,
        candidate_items: List[List[Dict[str, Any]]],
    ):
        all_index_ranges = [range(len(x)) for x in candidate_items]
        for index_tuple in itertools.product(*all_index_ranges):
            selected_items = []
            for concept_idx, cand_idx in enumerate(index_tuple):
                selected_items.append(candidate_items[concept_idx][cand_idx])
            yield index_tuple, selected_items

    def exhaustive_search(
        self,
        candidate_items: List[List[Dict[str, Any]]],
        concept_embs: List[torch.Tensor],
        full_prompt_emb: torch.Tensor,
    ) -> List[Dict[str, Any]]:
        results = []

        total_combinations = 1
        for x in candidate_items:
            total_combinations *= max(1, len(x))
        print(f"[Exhaustive] total combinations = {total_combinations}")

        concept_ids = list(range(len(candidate_items)))

        for combo_idx, (index_tuple, selected_items) in enumerate(self.enumerate_all_combinations(candidate_items)):
            total, breakdown, combo_emb = self.compute_full_combination_score(
                selected_items=selected_items,
                selected_concept_ids=concept_ids,
                concept_embs_all=concept_embs,
                full_prompt_emb=full_prompt_emb,
            )

            results.append({
                "concept_ids": concept_ids,
                "indices": tuple(index_tuple),
                "items": selected_items,
                "score": float(total.item()),
                "score_breakdown": breakdown,
                "combo_embedding": combo_emb.detach().cpu(),
            })

            if (combo_idx + 1) % 10000 == 0:
                print(f"[Exhaustive] scored {combo_idx + 1}/{total_combinations}")

        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    # ------------------------------------------------------
    # hierarchical search
    # ------------------------------------------------------
    def build_initial_pools(
        self,
        candidate_items: List[List[Dict[str, Any]]],
        concept_embs: List[torch.Tensor],
        full_prompt_emb: torch.Tensor,
    ) -> List[List[Dict[str, Any]]]:
        pools = []

        for concept_idx, items in enumerate(candidate_items):
            pool = []

            for cand_idx, item in enumerate(items):
                score, breakdown, combo_emb = self.compute_full_combination_score(
                    selected_items=[item],
                    selected_concept_ids=[concept_idx],
                    concept_embs_all=concept_embs,
                    full_prompt_emb=full_prompt_emb,
                )

                pool.append({
                    "concept_ids": [concept_idx],
                    "indices": (cand_idx,),
                    "items": [item],
                    "score": float(score.item()),
                    "score_breakdown": breakdown,
                    "combo_embedding": combo_emb.detach().cpu(),
                })

            pool.sort(key=lambda x: x["score"], reverse=True)
            pool = pool[:self.merge_pool_size]
            pools.append(pool)

        return pools

    def merge_two_pools(
        self,
        pool_a: List[Dict[str, Any]],
        pool_b: List[Dict[str, Any]],
        concept_embs: List[torch.Tensor],
        full_prompt_emb: torch.Tensor,
    ) -> List[Dict[str, Any]]:
        merged = []

        total_pairs = len(pool_a) * len(pool_b)
        print(f"[Merge] merging pools: {len(pool_a)} x {len(pool_b)} = {total_pairs}")

        for combo_a in pool_a:
            for combo_b in pool_b:
                concept_ids = combo_a["concept_ids"] + combo_b["concept_ids"]
                items = combo_a["items"] + combo_b["items"]
                indices = combo_a["indices"] + combo_b["indices"]

                zipped = list(zip(concept_ids, indices, items))
                zipped.sort(key=lambda x: x[0])

                concept_ids = [x[0] for x in zipped]
                indices = tuple(x[1] for x in zipped)
                items = [x[2] for x in zipped]

                total, breakdown, combo_emb = self.compute_full_combination_score(
                    selected_items=items,
                    selected_concept_ids=concept_ids,
                    concept_embs_all=concept_embs,
                    full_prompt_emb=full_prompt_emb,
                )

                merged.append({
                    "concept_ids": concept_ids,
                    "indices": indices,
                    "items": items,
                    "score": float(total.item()),
                    "score_breakdown": breakdown,
                    "combo_embedding": combo_emb.detach().cpu(),
                })

        merged.sort(key=lambda x: x["score"], reverse=True)
        merged = merged[:self.merge_pool_size]
        print(f"[Merge] kept top {len(merged)}")
        return merged

    def hierarchical_search(
        self,
        candidate_items: List[List[Dict[str, Any]]],
        concept_embs: List[torch.Tensor],
        full_prompt_emb: torch.Tensor,
    ) -> List[Dict[str, Any]]:
        pools = self.build_initial_pools(
            candidate_items=candidate_items,
            concept_embs=concept_embs,
            full_prompt_emb=full_prompt_emb,
        )

        level = 0
        while len(pools) > 1:
            print(f"[Hierarchical Search] level={level}, num_pools={len(pools)}")
            new_pools = []

            i = 0
            while i < len(pools):
                if i + 1 < len(pools):
                    merged_pool = self.merge_two_pools(
                        pool_a=pools[i],
                        pool_b=pools[i + 1],
                        concept_embs=concept_embs,
                        full_prompt_emb=full_prompt_emb,
                    )
                    new_pools.append(merged_pool)
                    i += 2
                else:
                    new_pools.append(pools[i])
                    i += 1

            pools = new_pools
            level += 1

        final_pool = pools[0]
        final_pool.sort(key=lambda x: x["score"], reverse=True)
        return final_pool

    # ------------------------------------------------------
    # diversity selection
    # ------------------------------------------------------
    def compute_slot_similarity(
        self,
        combo_a: Dict[str, Any],
        combo_b: Dict[str, Any],
    ) -> float:
        files_a = [x["model_file"] for x in combo_a["items"]]
        files_b = [x["model_file"] for x in combo_b["items"]]

        if len(files_a) == 0 or len(files_a) != len(files_b):
            return 0.0

        same = sum(1 for a, b in zip(files_a, files_b) if a == b)
        return same / len(files_a)

    def compute_embedding_similarity(
        self,
        combo_a: Dict[str, Any],
        combo_b: Dict[str, Any],
    ) -> float:
        emb_a = F.normalize(combo_a["combo_embedding"], dim=0)
        emb_b = F.normalize(combo_b["combo_embedding"], dim=0)

        sim = torch.dot(emb_a, emb_b).item()
        sim = max(0.0, min(1.0, sim))
        return sim

    def compute_mixed_similarity(
        self,
        combo_a: Dict[str, Any],
        combo_b: Dict[str, Any],
    ) -> Dict[str, float]:
        sim_slot = self.compute_slot_similarity(combo_a, combo_b)
        sim_emb = self.compute_embedding_similarity(combo_a, combo_b)
        sim_mix = self.sim_gamma * sim_slot + (1.0 - self.sim_gamma) * sim_emb

        return {
            "slot_sim": sim_slot,
            "emb_sim": sim_emb,
            "mixed_sim": sim_mix,
        }

    def build_slot_usage_counter(
        self,
        selected: List[Dict[str, Any]],
    ) -> Dict[int, Dict[str, int]]:
        slot_usage = defaultdict(lambda: defaultdict(int))

        for combo in selected:
            for pos, item in enumerate(combo["items"]):
                mf = item["model_file"]
                slot_usage[pos][mf] += 1

        return slot_usage

    def compute_slot_repeat_penalty(
        self,
        combo: Dict[str, Any],
        selected: List[Dict[str, Any]],
        slot_usage: Optional[Dict[int, Dict[str, int]]] = None,
        mode: str = "max",
    ) -> Dict[str, Any]:
        if len(selected) == 0:
            return {
                "penalty": 0.0,
                "per_slot_repeat_count": [],
                "normalized_per_slot_repeat": [],
            }

        if slot_usage is None:
            slot_usage = self.build_slot_usage_counter(selected)

        repeat_counts = []
        for pos, item in enumerate(combo["items"]):
            mf = item["model_file"]
            cnt = slot_usage[pos].get(mf, 0)
            repeat_counts.append(cnt)

        normalized = [cnt / len(selected) for cnt in repeat_counts]

        if len(normalized) == 0:
            penalty = 0.0
        elif mode == "max":
            penalty = max(normalized)
        else:
            penalty = sum(normalized) / len(normalized)

        return {
            "penalty": float(penalty),
            "per_slot_repeat_count": repeat_counts,
            "normalized_per_slot_repeat": normalized,
        }

    def violates_max_repeat_per_slot(
        self,
        combo: Dict[str, Any],
        selected: List[Dict[str, Any]],
        slot_usage: Optional[Dict[int, Dict[str, int]]] = None,
    ) -> bool:
        if self.max_repeat_per_slot is None:
            return False

        if slot_usage is None:
            slot_usage = self.build_slot_usage_counter(selected)

        for pos, item in enumerate(combo["items"]):
            mf = item["model_file"]
            cnt = slot_usage[pos].get(mf, 0)
            if cnt >= self.max_repeat_per_slot:
                return True

        return False

    def select_diverse_topk(
        self,
        scored_combinations: List[Dict[str, Any]],
        top_k: int,
        prefilter_top_n: Optional[int] = 500,
        repeat_penalty_mode: str = "max",
    ) -> List[Dict[str, Any]]:
        if len(scored_combinations) == 0:
            return []

        candidates = scored_combinations[:prefilter_top_n] if (prefilter_top_n is not None and prefilter_top_n > 0) else scored_combinations
        selected = []

        first = dict(candidates[0])
        first["diverse_score"] = first["score"]
        first["max_mixed_similarity_to_selected"] = 0.0
        first["slot_repeat_penalty"] = 0.0
        first["per_slot_repeat_count"] = [0] * len(first["items"])
        first["normalized_per_slot_repeat"] = [0.0] * len(first["items"])
        first["sim_detail_to_closest_selected"] = {
            "slot_sim": 0.0,
            "emb_sim": 0.0,
            "mixed_sim": 0.0,
        }
        selected.append(first)

        remaining = candidates[1:]

        while len(selected) < top_k and len(remaining) > 0:
            slot_usage = self.build_slot_usage_counter(selected)

            best_idx = None
            best_diverse_score = None
            best_payload = None

            for i, cand in enumerate(remaining):
                if self.violates_max_repeat_per_slot(cand, selected, slot_usage=slot_usage):
                    continue

                max_sim = None
                closest_sim_detail = None
                for s in selected:
                    sim_detail = self.compute_mixed_similarity(cand, s)
                    if (max_sim is None) or (sim_detail["mixed_sim"] > max_sim):
                        max_sim = sim_detail["mixed_sim"]
                        closest_sim_detail = sim_detail

                if max_sim is None:
                    max_sim = 0.0
                    closest_sim_detail = {
                        "slot_sim": 0.0,
                        "emb_sim": 0.0,
                        "mixed_sim": 0.0,
                    }

                repeat_info = self.compute_slot_repeat_penalty(
                    combo=cand,
                    selected=selected,
                    slot_usage=slot_usage,
                    mode=repeat_penalty_mode,
                )
                repeat_penalty = repeat_info["penalty"]

                diverse_score = (
                    self.diversity_quality_weight * cand["score"]
                    - self.diversity_sim_weight * max_sim
                    - self.diversity_repeat_weight * repeat_penalty
                )

                if (best_diverse_score is None) or (diverse_score > best_diverse_score):
                    best_diverse_score = diverse_score
                    best_idx = i
                    best_payload = {
                        "diverse_score": float(diverse_score),
                        "max_mixed_similarity_to_selected": float(max_sim),
                        "slot_repeat_penalty": float(repeat_penalty),
                        "per_slot_repeat_count": repeat_info["per_slot_repeat_count"],
                        "normalized_per_slot_repeat": repeat_info["normalized_per_slot_repeat"],
                        "sim_detail_to_closest_selected": closest_sim_detail,
                    }

            if best_idx is None:
                break

            chosen = dict(remaining.pop(best_idx))
            chosen.update(best_payload)
            selected.append(chosen)

        return selected

    # ------------------------------------------------------
    # main
    # ------------------------------------------------------
    def optimize(
        self,
        candidate_items: List[List[Dict[str, Any]]],
        concept_embs: List[torch.Tensor],
        full_prompt_emb: torch.Tensor,
        top_k: int = 10,
        prefilter_top_n: Optional[int] = 500,
        repeat_penalty_mode: str = "max",
    ) -> Dict[str, Any]:
        full_prompt_emb = self._normalize_vec(full_prompt_emb.to(self.device))
        concept_embs = [self._normalize_vec(x.to(self.device)) for x in concept_embs]

        processed_candidates = self.preprocess_candidates(
            candidate_items=candidate_items,
            concept_embs=concept_embs,
            full_prompt_emb=full_prompt_emb,
        )

        num_concepts = len(processed_candidates)

        if num_concepts <= self.exhaustive_max_concepts:
            print(f"[Search Mode] exhaustive (num_concepts={num_concepts})")
            candidate_pool = self.exhaustive_search(
                candidate_items=processed_candidates,
                concept_embs=concept_embs,
                full_prompt_emb=full_prompt_emb,
            )
        else:
            print(f"[Search Mode] hierarchical (num_concepts={num_concepts})")
            candidate_pool = self.hierarchical_search(
                candidate_items=processed_candidates,
                concept_embs=concept_embs,
                full_prompt_emb=full_prompt_emb,
            )

        diverse_topk = self.select_diverse_topk(
            scored_combinations=candidate_pool,
            top_k=top_k,
            prefilter_top_n=prefilter_top_n,
            repeat_penalty_mode=repeat_penalty_mode,
        )

        return {
            "candidate_pool": candidate_pool,
            "diverse_topk": diverse_topk,
            "search_mode": "exhaustive" if num_concepts <= self.exhaustive_max_concepts else "hierarchical",
        }


# ==========================================================
# 3. Pipeline
# ==========================================================
class HybridCombinationPipeline:
    def __init__(
        self,
        lora_index_path: str,
        device: str = "cuda",

        alpha: float = 1.0,
        beta_compat: float = 0.3,
        eta_redund: float = 0.4,
        delta: float = 0.5,

        lambda_concept: float = 1.0,
        lambda_prompt: float = 0.7,

        global_tau: float = 0.2,
        route_text_ratio: float = 0.8,
        task: str = "qwenemb",

        redundancy_tau: float = 0.3,
        compat_sim_suppress: float = 0.5,
        global_mix_uniform: float = 0.3,

        exhaustive_max_concepts: int = 3,
        merge_pool_size: int = 500,

        diversity_quality_weight: float = 0.7,
        diversity_sim_weight: float = 0.15,
        diversity_repeat_weight: float = 0.15,

        sim_gamma: float = 0.7,
        max_repeat_per_slot: Optional[int] = 2,
    ):
        self.device = device

        if task == "clipemb":
            self.clip_encoder = TextImageEncoder(device=device).to(device)
        else:
            self.clip_encoder = QwenVLEncoder(device=device).to(device)
        self.clip_encoder.eval()

        self.index_store = LoRAIndexStore(
            index_path=lora_index_path,
            device=device,
        )

        self.optimizer = HybridCombinationDiverseSelector(
            alpha=alpha,
            beta_compat=beta_compat,
            eta_redund=eta_redund,
            delta=delta,
            lambda_concept=lambda_concept,
            lambda_prompt=lambda_prompt,
            global_tau=global_tau,
            route_text_ratio=route_text_ratio,
            device=device,
            redundancy_tau=redundancy_tau,
            compat_sim_suppress=compat_sim_suppress,
            global_mix_uniform=global_mix_uniform,
            exhaustive_max_concepts=exhaustive_max_concepts,
            merge_pool_size=merge_pool_size,
            diversity_quality_weight=diversity_quality_weight,
            diversity_sim_weight=diversity_sim_weight,
            diversity_repeat_weight=diversity_repeat_weight,
            sim_gamma=sim_gamma,
            max_repeat_per_slot=max_repeat_per_slot,
        )

    @torch.no_grad()
    def encode_text(self, text: str) -> torch.Tensor:
        emb = self.clip_encoder.encoding_text([text])[0]
        emb = F.normalize(emb, dim=0)
        return emb

    def build_candidate_items(
        self,
        data: Dict[str, Any],
        per_concept_top_k: Optional[int] = 40,
    ):
        extract_concept = data["extract_concept"]
        retrieval_results = data["retrieval_results"]

        if "prompt" in data:
            full_prompt_text = data["prompt"]
        elif "rewrite_prompt" in data:
            full_prompt_text = data["rewrite_prompt"]
        else:
            full_prompt_text = " ".join([ec["keyword"] for ec in extract_concept])

        full_prompt_emb = self.encode_text(full_prompt_text)

        concept_names = []
        concept_embs = []
        candidate_items = []

        for ec in extract_concept:
            keyword = ec["keyword"]
            concept_emb = self.encode_text(keyword)

            concept_names.append(keyword)
            concept_embs.append(concept_emb)

            cur_candidates = retrieval_results.get(keyword, [])
            if per_concept_top_k is not None:
                cur_candidates = cur_candidates[:per_concept_top_k]

            cur_items = []
            for item in cur_candidates:
                model_file = item["model_file"]
                lora_item = self.index_store.get_embedding(model_file)
                if lora_item is None:
                    continue

                cur_items.append({
                    "model_file": model_file,
                    "retrieval_score": item.get("retrieval_score", None),
                    "txt_embedding": lora_item["txt_embedding"],
                    "img_embedding": lora_item["img_embedding"],
                    "is_dual": lora_item["is_dual"],
                })

            candidate_items.append(cur_items)

        return concept_names, candidate_items, concept_embs, full_prompt_emb

    def optimize_one_data(
        self,
        data: Dict[str, Any],
        per_concept_top_k: Optional[int] = 40,
        top_k: int = 10,
        prefilter_top_n: Optional[int] = 500,
        repeat_penalty_mode: str = "max",
    ) -> Dict[str, Any]:
        concept_names, candidate_items, concept_embs, full_prompt_emb = self.build_candidate_items(
            data=data,
            per_concept_top_k=per_concept_top_k,
        )

        if any(len(x) == 0 for x in candidate_items):
            data["combination_results"] = {
                "concept_names": concept_names,
                "candidate_pool": [],
                "diverse_topk": [],
                "error": "some concepts have empty candidate list",
            }
            return data

        optimize_res = self.optimizer.optimize(
            candidate_items=candidate_items,
            concept_embs=concept_embs,
            full_prompt_emb=full_prompt_emb,
            top_k=top_k,
            prefilter_top_n=prefilter_top_n,
            repeat_penalty_mode=repeat_penalty_mode,
        )

        candidate_pool_export = []
        pool_src = optimize_res["candidate_pool"]
        if prefilter_top_n is not None and prefilter_top_n > 0:
            pool_src = pool_src[:prefilter_top_n]

        for combo in pool_src:
            candidate_pool_export.append({
                "concept_ids": combo["concept_ids"],
                "indices": list(combo["indices"]),
                "model_files": [x["model_file"] for x in combo["items"]],
                "retrieval_scores": [x.get("retrieval_score", None) for x in combo["items"]],
                "score": combo["score"],
                "score_breakdown": combo["score_breakdown"],
            })

        diverse_export = []
        for combo in optimize_res["diverse_topk"]:
            diverse_export.append({
                "concept_ids": combo["concept_ids"],
                "indices": list(combo["indices"]),
                "model_files": [x["model_file"] for x in combo["items"]],
                "retrieval_scores": [x.get("retrieval_score", None) for x in combo["items"]],
                "score": combo["score"],
                "diverse_score": combo.get("diverse_score", None),
                "max_mixed_similarity_to_selected": combo.get("max_mixed_similarity_to_selected", None),
                "slot_repeat_penalty": combo.get("slot_repeat_penalty", None),
                "per_slot_repeat_count": combo.get("per_slot_repeat_count", None),
                "normalized_per_slot_repeat": combo.get("normalized_per_slot_repeat", None),
                "sim_detail_to_closest_selected": combo.get("sim_detail_to_closest_selected", None),
                "score_breakdown": combo["score_breakdown"],
            })

        data["combination_results"] = {
            "concept_names": concept_names,
            "search_mode": optimize_res["search_mode"],
            "candidate_pool": candidate_pool_export,
            "diverse_topk": diverse_export,
        }
        return data


# ==========================================================
# 4. Batch process
# ==========================================================
def optimize_retrieval_results_file_hybrid(
    retrieval_result_path: str,
    lora_index_path: str,
    output_path: str,
    device: str = "cuda",

    per_concept_top_k: Optional[int] = 40,
    top_k: int = 10,
    prefilter_top_n: Optional[int] = 500,
    repeat_penalty_mode: str = "max",

    alpha: float = 1.0,
    beta_compat: float = 0.3,
    eta_redund: float = 0.4,
    delta: float = 0.5,

    lambda_concept: float = 1.0,
    lambda_prompt: float = 0.7,

    global_tau: float = 0.2,
    route_text_ratio: float = 0.8,
    task: str = "qwenemb",

    redundancy_tau: float = 0.3,
    compat_sim_suppress: float = 0.5,
    global_mix_uniform: float = 0.3,

    exhaustive_max_concepts: int = 3,
    merge_pool_size: int = 500,

    diversity_quality_weight: float = 0.7,
    diversity_sim_weight: float = 0.15,
    diversity_repeat_weight: float = 0.15,

    sim_gamma: float = 0.7,
    max_repeat_per_slot: Optional[int] = 2,
):
    pipeline = HybridCombinationPipeline(
        lora_index_path=lora_index_path,
        device=device,
        alpha=alpha,
        beta_compat=beta_compat,
        eta_redund=eta_redund,
        delta=delta,
        lambda_concept=lambda_concept,
        lambda_prompt=lambda_prompt,
        global_tau=global_tau,
        route_text_ratio=route_text_ratio,
        task=task,
        redundancy_tau=redundancy_tau,
        compat_sim_suppress=compat_sim_suppress,
        global_mix_uniform=global_mix_uniform,
        exhaustive_max_concepts=exhaustive_max_concepts,
        merge_pool_size=merge_pool_size,
        diversity_quality_weight=diversity_quality_weight,
        diversity_sim_weight=diversity_sim_weight,
        diversity_repeat_weight=diversity_repeat_weight,
        sim_gamma=sim_gamma,
        max_repeat_per_slot=max_repeat_per_slot,
    )

    with open(retrieval_result_path, "r", encoding="utf-8") as f:
        datas = [json.loads(line) for line in f]

    with open(output_path, "w", encoding="utf-8") as fout:
        for i, data in enumerate(datas):
            print(f"[Optimize Hybrid] {i}/{len(datas)}")
            try:
                data = pipeline.optimize_one_data(
                    data=data,
                    per_concept_top_k=per_concept_top_k,
                    top_k=top_k,
                    prefilter_top_n=prefilter_top_n,
                    repeat_penalty_mode=repeat_penalty_mode,
                )
            except Exception as e:
                data["combination_results"] = {
                    "candidate_pool": [],
                    "diverse_topk": [],
                    "error": str(e),
                }

            fout.write(json.dumps(data, ensure_ascii=False) + "\n")

    print(f"[Done] saved optimized results to {output_path}")


# ==========================================================
# 5. Example
# ==========================================================
if __name__ == "__main__":
    index_paths = ["lora_index/qwenemb_cosinsmoothl1_all_5-199.pt", "lora_index/clipemb_contastive_all_2-1248.pt"]
    optimize_retrieval_results_file_hybrid(
        retrieval_result_path="test_data/data_500_qwenemb_5-199_2.jsonl",
        lora_index_path=index_paths[0],
        output_path="test_data/data_500_qwenemb_5-199_2_diverse.jsonl",
        device="cuda",

        per_concept_top_k=30,
        top_k=10,
        prefilter_top_n=300,
        repeat_penalty_mode="max",

        alpha=1.0,
        beta_compat=0.3,
        eta_redund=0.4,
        delta=0.5,

        lambda_concept=1.0,
        lambda_prompt=0.7,

        global_tau=0.2,
        route_text_ratio=0.8,
        task="qwenemb",

        redundancy_tau=0.3,
        compat_sim_suppress=0.5,
        global_mix_uniform=0.3,

        # <=3 concept 用全枚举，>3 用 hierarchical
        exhaustive_max_concepts=3,
        merge_pool_size=200,

        diversity_quality_weight=0.7,
        diversity_sim_weight=0.15,
        diversity_repeat_weight=0.15,

        sim_gamma=0.7,
        max_repeat_per_slot=2,
    )



# ==========================================================
# 5. Example
# # ==========================================================
# if __name__ == "__main__":
#     index_paths = ["lora_index/qwenemb_cosinsmoothl1_all_5-199.pt", "lora_index/clipemb_contastive_all_2-1248.pt"]
#     optimize_retrieval_results_file_exhaustive_diverse(
#         retrieval_result_path="test_data/data_500_qwenemb_5-199.jsonl",
#         lora_index_path=index_paths[0],
#         output_path="test_data/data_500_qwenemb_5-199_diverse.jsonl",
#         device="cuda:1",

#         # 每个 concept 初召回候选数
#         per_concept_top_k=35,

#         # 最终输出多少个多样化组合
#         top_k=10,

#         # 从高质量组合前 N 个里做多样性筛选
#         prefilter_top_n=400,

#         # 组合质量分参数
#         alpha=1.0,
#         beta_compat=0.3,  #组合成员之间“兼容/互补”的奖励强度
#         eta_redund=0.4,  #组合里两个 LoRA 太像/太重复时，扣多少分
#         delta=0.5,

#         lambda_concept=1.0,
#         lambda_prompt=0.7,

#         global_tau=0.2,
#         route_text_ratio=0.85,
#         task="qwenemb",

#         redundancy_tau=0.3,
#         compat_sim_suppress=0.5,
#         global_mix_uniform=0.3,

#         # 多样性筛选参数
#         diversity_quality_weight=0.7,   # 越大越偏向质量
#         diversity_sim_weight=0.15,      # 越大越惩罚“组合整体相似”
#         diversity_repeat_weight=0.15,   # 越大越惩罚“同槽位重复同一个 LoRA”

#         sim_gamma=0.7,                  # mixed similarity 里 slot similarity 占比
#         max_repeat_per_slot=2,          # 每个槽位同一个 LoRA 最多出现 2 次；设为 None 表示不限制
#     )
