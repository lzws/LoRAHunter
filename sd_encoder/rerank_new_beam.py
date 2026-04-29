import json
from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.nn.functional as F
from models import TextImageEncoder, QwenVLEncoder
from collections import Counter

# ==========================================================
# 1. LoRA embedding index
# ==========================================================
class LoRAIndexStore:
    """
    兼容:
    - single index:
        {
            "model_files": [...],
            "embeddings": Tensor[N, D]
        }
      或
        {
            "model_files": [...],
            "txt_embeddings": Tensor[N, D],
            "is_dual": False
        }

    - dual index:
        {
            "model_files": [...],
            "txt_embeddings": Tensor[N, Dt],
            "img_embeddings": Tensor[N, Di],
            "is_dual": True
        }

    对外提供:
    - model_file -> item embedding dict
    """
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
        print(f"[LoRAIndexStore] txt_embeddings shape={self.txt_embeddings.shape}")
        if self.img_embeddings is not None:
            print(f"[LoRAIndexStore] img_embeddings shape={self.img_embeddings.shape}")

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
# 2. Beam-search combination optimizer
# ==========================================================
class LoRACombinationOptimizer:
    """
    dual-head 兼容版：
    - query/concept/full_prompt 侧当前只有 text embedding
    - 如果 index 是 dual，则用同一个 text query embedding 同时和 txt/img route 打分
    - 所有 match / compatibility / redundancy / combo embedding 都支持双路融合

    route_text_ratio:
        text route 的权重
        img route 的权重 = 1 - route_text_ratio
    """

    def __init__(
        self,
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

        route_text_ratio: float = 1.0,

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

        self.route_text_ratio = route_text_ratio
        self.device = device

    # ------------------------------------------------------
    # utility
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
        """
        query_emb 当前只有 text emb
        若 item 有 dual routes:
            score = r * dot(query, txt_emb) + (1-r) * dot(query, img_emb)
        否则退化为 text route
        """
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
        """
        LoRA-LoRA 相似度:
        - single: dot(txt, txt)
        - dual:
            sim = r * dot(txt_a, txt_b) + (1-r) * dot(img_a, img_b)
        """
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
        """
        给组合 embedding / MMR 用的单个 item 融合 embedding。

        single:
            fused = txt_emb

        dual:
            fused = normalize(r * txt_emb + (1-r) * img_emb)
        """
        txt_emb = item["txt_embedding"]
        img_emb = item.get("img_embedding", None)

        if img_emb is None:
            return txt_emb

        if txt_emb.shape[-1] != img_emb.shape[-1]:
            # 如果维度不同，退化为 text route
            return txt_emb

        fused = self.route_text_ratio * txt_emb + (1.0 - self.route_text_ratio) * img_emb
        fused = F.normalize(fused, dim=0)
        return fused

    # ------------------------------------------------------
    # item-level match
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
        score = self.lambda_concept * local_sim + self.lambda_prompt * prompt_sim
        return score

    # ------------------------------------------------------
    # pairwise terms
    # ------------------------------------------------------
    def compute_redundancy(
        self,
        item_i: Dict[str, Any],
        item_j: Dict[str, Any],
    ) -> torch.Tensor:
        sim = self._item_to_item_similarity(item_i, item_j)
        return torch.relu(sim)

    def compute_compatibility(
        self,
        item_i: Dict[str, Any],
        item_j: Dict[str, Any],
        full_prompt_emb: torch.Tensor,
    ) -> torch.Tensor:
        prompt_fit_i = self._route_query_to_lora_score(full_prompt_emb, item_i)
        prompt_fit_j = self._route_query_to_lora_score(full_prompt_emb, item_j)
        compat = 0.5 * (prompt_fit_i + prompt_fit_j)
        return compat

    def compute_pair_utility(
        self,
        item_i: Dict[str, Any],
        item_j: Dict[str, Any],
        full_prompt_emb: torch.Tensor,
    ) -> torch.Tensor:
        compat = self.compute_compatibility(
            item_i=item_i,
            item_j=item_j,
            full_prompt_emb=full_prompt_emb,
        )
        redund = self.compute_redundancy(
            item_i=item_i,
            item_j=item_j,
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
        fused_embs = []
        item_scores = []

        for item, concept_emb in zip(selected_items, concept_embs):
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
        new_states = []

        for state in beam:
            old_items = state["items"]
            old_score = state["search_score"]

            for next_idx, next_item in enumerate(next_candidates):
                delta = self.alpha * next_item["_match_score"]

                for old_item in old_items:
                    delta += float(
                        self.compute_pair_utility(
                            item_i=old_item,
                            item_j=next_item,
                            full_prompt_emb=full_prompt_emb,
                        ).item()
                    )

                new_state = {
                    "indices": state["indices"] + [next_idx],
                    "items": old_items + [next_item],
                    "search_score": old_score + delta,
                }
                new_states.append(new_state)

        new_states.sort(key=lambda x: x["search_score"], reverse=True)
        return new_states[:self.beam_width]

    def beam_search_combinations(
        self,
        candidate_items: List[List[Dict[str, Any]]],
        concept_embs: List[torch.Tensor],
        full_prompt_emb: torch.Tensor,
    ) -> List[Dict[str, Any]]:
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
        results = []

        for state in beam_results:
            selected_items = state["items"]

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
    # individual / pairwise explicit score
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
                item=item,
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
            item_i = selected_items[i]
            for j in range(i + 1, n):
                item_j = selected_items[j]
                score += self.compute_pair_utility(
                    item_i=item_i,
                    item_j=item_j,
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
        if concept_embs is not None and full_prompt_emb is not None:
            return self.compute_weighted_combo_embedding(
                selected_items=combo["items"],
                concept_embs=concept_embs,
                full_prompt_emb=full_prompt_emb,
            )

        embs = [self._get_fused_item_embedding(item) for item in combo["items"]]
        emb = torch.stack(embs, dim=0).mean(dim=0)
        emb = F.normalize(emb, dim=0)
        return emb

    def overlap_similarity(self, combo_a: Dict[str, Any], combo_b: Dict[str, Any]) -> float:
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
            "search_order_map": order_map,
        }
        if return_all_pool:
            result["candidate_pool"] = candidate_pool
        return result


# ==========================================================
# 3. direct pipeline for retrieval_results
# ==========================================================
class CombinationPipeline:
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

        route_text_ratio: float = 1.0,
        task = "clipemb"
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
            route_text_ratio=route_text_ratio,
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
                # parts.append(f"{keyword}. {desc}")
                parts.append(f"{keyword}")
            full_prompt_text = " ".join(parts)

        full_prompt_emb = self.encode_text(full_prompt_text)

        concept_names = []
        concept_embs = []
        candidate_items = []

        for ec in extract_concept:
            keyword = ec["keyword"]
            retrieval_description = ec.get("retrieval_description", "")

            concept_query = f"{keyword}"
            concept_emb = self.encode_text(concept_query)

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

        if return_candidate_pool and "candidate_pool" in optimize_res:
            self.analyze_candidate_pool(optimize_res["candidate_pool"])

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

    

    def analyze_candidate_pool(self, candidate_pool):
        if not candidate_pool:
            print("[CandidatePool] empty")
            return

        combo_len = len(candidate_pool[0]["items"])
        pool_size = len(candidate_pool)

        print(f"[CandidatePool] size={pool_size}, combo_len={combo_len}")

        for pos in range(combo_len):
            counter = Counter()
            for combo in candidate_pool:
                if pos < len(combo["items"]):
                    mf = combo["items"][pos]["model_file"]
                    counter[mf] += 1

            uniq = len(counter)
            most_common = counter.most_common(10)

            print(f"[CandidatePool][Pos {pos}] unique={uniq}")
            for mf, cnt in most_common:
                print(f"    {mf}: {cnt} ({cnt / pool_size:.2%})")

        # 组合整体唯一数
        combo_counter = Counter()
        for combo in candidate_pool:
            key = tuple(item["model_file"] for item in combo["items"])
            combo_counter[key] += 1

        print(f"[CandidatePool] unique full combos={len(combo_counter)} / {pool_size}")
        print("[CandidatePool] top repeated combos:")
        for combo_key, cnt in combo_counter.most_common(10):
            print(f"    {combo_key}: {cnt} ({cnt / pool_size:.2%})")

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

    route_text_ratio: float = 1.0,

    return_candidate_pool: bool = False,
    reorder_concepts: bool = True,
    task="clipemb"
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
        route_text_ratio=route_text_ratio,
        task=task,
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
        retrieval_result_path="test_data/data_500_qwenemb_4-68.jsonl",
        lora_index_path="lora_index/qwenemb_cosinsmoothl1_all_4-68.pt",
        output_path="test_data/data_500_qwenemb_4-68_reank_beam.jsonl",
        device="cuda",

        per_concept_top_k=50,
        top_pool_size=500,
        final_top_k=20,

        alpha=1.0,
        beta_compat=0.3,
        eta_redund=0.4,
        delta=0.5,

        lambda_concept=1,
        lambda_prompt=0.7,
        global_tau=0.2,

        local_top_m=80,
        beam_width=400,
        final_pool_size=500,

        lambda_quality=0.8,
        overlap_weight=0.6,
        emb_weight=0.4,

        route_text_ratio=0.85,   # dual-head 下 text/img route 融合比例
        return_candidate_pool=True,
        reorder_concepts=True,
        task='qwenemb'
    )
