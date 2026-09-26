import argparse
import itertools
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor

from dataset import _extract_tensor, load_tensor, read_jsonl


class CLIPTextEncoder:
    def __init__(self, model_name, device):
        self.device = torch.device(device)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).float().eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)

    @torch.no_grad()
    def encode(self, texts):
        inputs = self.processor(
            text=list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        output = self.model.get_text_features(**inputs)
        if hasattr(output, "pooler_output"):
            output = output.pooler_output
        return F.normalize(output.float().cpu(), dim=-1)


def _value_embedding(value):
    if value is None:
        return None
    if torch.is_tensor(value) or isinstance(value, (list, tuple)):
        return _extract_tensor(value).flatten().float()
    if isinstance(value, dict):
        for key in ("embedding", "query_embedding", "concept_embedding", "vector"):
            if key in value:
                return _value_embedding(value[key])
    return None


def _concept_text(concept):
    if isinstance(concept, str):
        return concept
    if isinstance(concept, dict):
        values = []
        for key in ("retrieval_text", "text", "description", "desc", "name", "concept", "title"):
            value = concept.get(key)
            if value is not None and str(value) not in values:
                values.append(str(value))
        return " ".join(values)
    return str(concept)


def _concepts(row):
    concepts = row.get("concepts", row.get("extract_concept", []))
    if concepts is None:
        return []
    if isinstance(concepts, (str, dict)):
        return [concepts]
    return list(concepts)


def _query_embedding(row, task, clip_encoder, query_emb_root):
    if task == "clipemb":
        return clip_encoder.encode([row.get("prompt", row.get("text", ""))])[0]
    for key in ("query_embedding", "gcl_embedding", "prompt_embedding"):
        embedding = _value_embedding(row.get(key))
        if embedding is not None:
            return F.normalize(embedding, dim=-1)
    for key in ("query_embedding_path", "gcl_embedding_path", "prompt_embedding_path"):
        if row.get(key):
            return F.normalize(load_tensor(row[key]).flatten(), dim=-1)
    if query_emb_root is not None and row.get("iid") is not None:
        iid = str(row["iid"])
        candidates = [
            Path(query_emb_root) / iid[:2] / f"{iid}.pth",
            Path(query_emb_root) / iid[:2] / iid[2:4] / f"{iid}.pth",
        ]
        for path in candidates:
            if path.exists():
                return F.normalize(load_tensor(path).flatten(), dim=-1)
    raise ValueError("gclemb retrieval requires query_embedding, query_embedding_path, or --query-emb-root with iid")


def _concept_embeddings(row, concepts, task, clip_encoder):
    if not concepts:
        return torch.empty(0, 0)
    if task == "clipemb":
        return clip_encoder.encode([_concept_text(concept) for concept in concepts])
    row_embeddings = row.get("concept_embeddings")
    outputs = []
    for index, concept in enumerate(concepts):
        embedding = _value_embedding(concept)
        if embedding is None and isinstance(concept, dict):
            for key in ("embedding_path", "query_embedding_path", "gcl_embedding_path"):
                if concept.get(key):
                    embedding = load_tensor(concept[key]).flatten().float()
                    break
        if embedding is None and isinstance(row_embeddings, list) and index < len(row_embeddings):
            embedding = _value_embedding(row_embeddings[index])
        if embedding is None:
            raise ValueError("gclemb concepts require embeddings or embedding_path for every concept")
        outputs.append(F.normalize(embedding, dim=-1))
    return torch.stack(outputs, dim=0)


def _redundancy(embeddings, indices):
    if len(indices) < 2:
        return 0.0
    selected = embeddings[list(indices)]
    similarity = selected @ selected.transpose(0, 1)
    upper = similarity.triu(diagonal=1)
    count = len(indices) * (len(indices) - 1) // 2
    return float(upper.sum().item() / count)


def _metadata_fields(row):
    return {key: row[key] for key in ("title", "name", "description", "desc", "model_id") if key in row}


def select_combinations(
    model_files,
    metadata,
    embeddings,
    query_scores,
    concept_scores,
    num_combos,
    candidate_k_full,
    concept_top_k,
    max_unique_loras,
    lambda_full,
    lambda_cover,
    lambda_redundancy,
    max_enumerations,
):
    n = len(model_files)
    candidate_k_full = min(candidate_k_full, n)
    full_top = torch.topk(query_scores, candidate_k_full).indices.tolist()
    if concept_scores.numel() == 0:
        combinations = []
        for index in full_top[:num_combos]:
            combinations.append({
                "score": float(query_scores[index].item()),
                "full_relevance": float(query_scores[index].item()),
                "concept_coverage": 0.0,
                "redundancy": 0.0,
                "indices": [index],
                "assignment": [],
            })
    else:
        candidate_pool = set(full_top)
        for scores in concept_scores:
            candidate_pool.update(torch.topk(scores, min(concept_top_k, n)).indices.tolist())
        candidate_pool = sorted(candidate_pool)
        candidate_tensor = torch.tensor(candidate_pool, dtype=torch.long)
        concept_candidates = []
        for scores in concept_scores:
            local_scores = scores[candidate_tensor]
            order = torch.topk(local_scores, min(concept_top_k, len(candidate_pool))).indices.tolist()
            concept_candidates.append([candidate_pool[index] for index in order])
        scored = {}
        count = 0
        for assignment in itertools.product(*concept_candidates):
            count += 1
            if count > max_enumerations:
                break
            selected = tuple(sorted(set(assignment)))
            if not selected or len(selected) > max_unique_loras:
                continue
            selected_scores = query_scores[list(selected)]
            full_relevance = 0.5 * selected_scores.max() + 0.5 * selected_scores.mean()
            coverage = torch.stack([scores[list(selected)].max() for scores in concept_scores]).mean()
            redundancy = _redundancy(embeddings, selected)
            score = float((lambda_full * full_relevance + lambda_cover * coverage - lambda_redundancy * redundancy).item())
            current = scored.get(selected)
            if current is None or score > current["score"]:
                scored[selected] = {
                    "score": score,
                    "full_relevance": float(full_relevance.item()),
                    "concept_coverage": float(coverage.item()),
                    "redundancy": redundancy,
                    "indices": list(selected),
                    "assignment": list(assignment),
                }
        combinations = sorted(scored.values(), key=lambda item: item["score"], reverse=True)[:num_combos]
    output = []
    for rank, combination in enumerate(combinations, start=1):
        loras = []
        for index in combination["indices"]:
            item = {
                "model_file": model_files[index],
                "full_score": float(query_scores[index].item()),
            }
            if concept_scores.numel() != 0:
                item["concept_scores"] = [float(scores[index].item()) for scores in concept_scores]
            if index < len(metadata) and isinstance(metadata[index], dict):
                item.update(_metadata_fields(metadata[index]))
            loras.append(item)
        output.append({
            "rank": rank,
            "score": combination["score"],
            "full_relevance": combination["full_relevance"],
            "concept_coverage": combination["concept_coverage"],
            "redundancy": combination["redundancy"],
            "assignment": combination["assignment"],
            "loras": loras,
        })
    return output


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task", choices=["clipemb", "gclemb"], required=True)
    parser.add_argument("--clip-model", default=None)
    parser.add_argument("--query-emb-root", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-combos", type=int, default=5)
    parser.add_argument("--candidate-k-full", type=int, default=50)
    parser.add_argument("--concept-top-k", type=int, default=10)
    parser.add_argument("--max-unique-loras", type=int, default=4)
    parser.add_argument("--lambda-full", type=float, default=1.0)
    parser.add_argument("--lambda-cover", type=float, default=1.0)
    parser.add_argument("--lambda-redundancy", type=float, default=0.25)
    parser.add_argument("--max-enumerations", type=int, default=10000)
    return parser.parse_args()


def main():
    args = parse_args()
    index = torch.load(args.index, map_location="cpu")
    model_files = index["model_files"]
    embeddings = F.normalize(index["embeddings"].float(), dim=-1)
    metadata = index.get("metadata", [{} for _ in model_files])
    clip_encoder = None
    if args.task == "clipemb":
        if not args.clip_model:
            raise ValueError("--clip-model is required for clipemb")
        clip_encoder = CLIPTextEncoder(args.clip_model, args.device)
    rows = read_jsonl(args.input)
    with open(args.output, "w", encoding="utf-8") as handle:
        for row in rows:
            query = _query_embedding(row, args.task, clip_encoder, args.query_emb_root)
            query = F.normalize(query.float(), dim=-1)
            if query.numel() != embeddings.shape[1]:
                raise ValueError(f"query dimension {query.numel()} does not match index dimension {embeddings.shape[1]}")
            query_scores = query @ embeddings.transpose(0, 1)
            concepts = _concepts(row)
            concept_embeddings = _concept_embeddings(row, concepts, args.task, clip_encoder)
            if concept_embeddings.numel() == 0:
                concept_scores = concept_embeddings
            else:
                if concept_embeddings.shape[1] != embeddings.shape[1]:
                    raise ValueError("concept embedding dimension does not match index dimension")
                concept_scores = concept_embeddings @ embeddings.transpose(0, 1)
            output_row = dict(row)
            output_row["used_concepts"] = concepts
            output_row["lora_combinations"] = select_combinations(
                model_files=model_files,
                metadata=metadata,
                embeddings=embeddings,
                query_scores=query_scores,
                concept_scores=concept_scores,
                num_combos=args.num_combos,
                candidate_k_full=args.candidate_k_full,
                concept_top_k=args.concept_top_k,
                max_unique_loras=args.max_unique_loras,
                lambda_full=args.lambda_full,
                lambda_cover=args.lambda_cover,
                lambda_redundancy=args.lambda_redundancy,
                max_enumerations=args.max_enumerations,
            )
            handle.write(json.dumps(output_row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
