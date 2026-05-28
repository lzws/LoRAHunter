import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
import os
import json
import math
import argparse
from typing import Dict, List

from PIL import Image
from tqdm import tqdm

from my_scorers import build_scorers


DEFAULT_SCORER_TAUS = {
    "clip": 0.03,
    "pick": 0.03,
}

DEFAULT_SCORER_WEIGHTS = {
    "clip": 1.0,
    "pick": 1.0,
}


def get_model_id_from_model_file(model_file: str):
    return os.path.basename(model_file).replace(".safetensors", "")


def collect_seed_image_paths(img_dir: str, seeds: List[int]):
    paths = []
    valid_seeds = []
    for seed in seeds:
        p = os.path.join(img_dir, f"{seed}.jpg")
        if os.path.exists(p):
            paths.append(p)
            valid_seeds.append(seed)
    return paths, valid_seeds


def load_pil_images(image_paths: List[str]):
    images = []
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        images.append(img)
    return images


def ensure_list_of_floats(x):
    if hasattr(x, "tolist"):
        x = x.tolist()
    return [float(v) for v in x]


def safe_mean(xs: List[float]):
    if len(xs) == 0:
        return None
    return float(sum(xs) / len(xs))


def normalize_weights(weight_dict: Dict[str, float], scorer_names: List[str]):
    ws = []
    for name in scorer_names:
        ws.append(float(weight_dict.get(name, 1.0)))
    s = sum(ws)
    if s <= 0:
        ws = [1.0 for _ in scorer_names]
        s = len(ws)
    return {name: w / s for name, w in zip(scorer_names, ws)}


def fused_reward_from_delta_means(
    delta_mean_dict: Dict[str, float],
    scorer_taus: Dict[str, float],
    scorer_weights: Dict[str, float],
):
    norm_weights = normalize_weights(scorer_weights, list(delta_mean_dict.keys()))

    reward_parts = {}
    reward = 0.0
    for name, delta_mean in delta_mean_dict.items():
        tau = max(float(scorer_taus.get(name, 1.0)), 1e-6)
        norm_delta = math.tanh(delta_mean / tau)
        reward_parts[name] = float(norm_delta)
        reward += norm_weights[name] * norm_delta

    return float(reward), reward_parts, norm_weights


def score_base_images(
    prompt: str,
    base_image_paths: List[str],
    valid_seeds: List[int],
    scorers: Dict[str, object],
):
    images = load_pil_images(base_image_paths)

    base_scores = {}
    base_score_means = {}

    for scorer_name, scorer in scorers.items():
        scores = scorer.score(prompt, images)
        scores = ensure_list_of_floats(scores)

        seed_score_map = {
            int(seed): float(score)
            for seed, score in zip(valid_seeds, scores)
        }
        base_scores[scorer_name] = seed_score_map
        base_score_means[scorer_name] = safe_mean(scores)

    return base_scores, base_score_means


def score_candidate_lora(
    prompt: str,
    iid,
    candidate: Dict,
    image_root: str,
    seeds: List[int],
    scorers: Dict[str, object],
    base_scores_by_scorer: Dict[str, Dict[int, float]],
    scorer_taus: Dict[str, float],
    scorer_weights: Dict[str, float],
):
    model_file = candidate["model_file"]
    model_id = get_model_id_from_model_file(model_file)

    lora_dir = os.path.join(image_root, str(iid), model_id)
    lora_image_paths, lora_valid_seeds = collect_seed_image_paths(lora_dir, seeds)

    cand = dict(candidate)

    if len(lora_image_paths) == 0:
        cand["score_status"] = "missing_lora_images"
        cand["num_lora_images"] = 0
        cand["num_common_seeds"] = 0
        cand["lora_scores"] = {}
        cand["lora_score_mean"] = {}
        cand["delta_scores"] = {}
        cand["delta_mean"] = {}
        cand["reward"] = None
        cand["reward_parts"] = {}
        cand["reward_weights"] = {}
        return cand

    images = load_pil_images(lora_image_paths)

    lora_scores = {}
    lora_score_means = {}
    delta_scores = {}
    delta_means = {}

    min_common_seed_count = None

    for scorer_name, scorer in scorers.items():
        scores = scorer.score(prompt, images)
        scores = ensure_list_of_floats(scores)

        lora_seed_score_map = {
            int(seed): float(score)
            for seed, score in zip(lora_valid_seeds, scores)
        }
        lora_scores[scorer_name] = lora_seed_score_map
        lora_score_means[scorer_name] = safe_mean(scores)

        base_seed_score_map = base_scores_by_scorer.get(scorer_name, {})
        common_seeds = sorted(set(lora_seed_score_map.keys()) & set(base_seed_score_map.keys()))

        deltas = {
            int(seed): float(lora_seed_score_map[seed] - base_seed_score_map[seed])
            for seed in common_seeds
        }

        delta_scores[scorer_name] = deltas
        delta_means[scorer_name] = safe_mean(list(deltas.values())) if len(deltas) > 0 else None

        if min_common_seed_count is None:
            min_common_seed_count = len(common_seeds)
        else:
            min_common_seed_count = min(min_common_seed_count, len(common_seeds))

    valid_delta_means = {
        k: v for k, v in delta_means.items()
        if v is not None
    }

    cand["num_lora_images"] = len(lora_valid_seeds)
    cand["num_common_seeds"] = 0 if min_common_seed_count is None else int(min_common_seed_count)
    cand["lora_scores"] = lora_scores
    cand["lora_score_mean"] = lora_score_means
    cand["delta_scores"] = delta_scores
    cand["delta_mean"] = delta_means

    if len(valid_delta_means) == 0:
        cand["score_status"] = "no_common_seed_scores"
        cand["reward"] = None
        cand["reward_parts"] = {}
        cand["reward_weights"] = {}
        return cand

    reward, reward_parts, normalized_weights = fused_reward_from_delta_means(
        delta_mean_dict=valid_delta_means,
        scorer_taus=scorer_taus,
        scorer_weights=scorer_weights,
    )

    cand["score_status"] = "ok"
    cand["reward"] = reward
    cand["reward_parts"] = reward_parts
    cand["reward_weights"] = normalized_weights
    return cand


def compute_rewards(
    input_jsonl_path: str,
    output_jsonl_path: str,
    image_root: str,
    device: str,
    start: int,
    end: int,
    seeds: List[int],
    scorer_taus: Dict[str, float],
    scorer_weights: Dict[str, float],
):
    scorers = build_scorers(device)
    scorer_names = list(scorers.keys())

    print(f"[Info] device={device}, start={start}, end={end}")
    print(f"[Info] scorers={scorer_names}")

    for name in scorer_names:
        scorer_taus.setdefault(name, 1.0)
        scorer_weights.setdefault(name, 1.0)

    with open(input_jsonl_path, "r", encoding="utf-8") as f:
        all_lines = f.readlines()

    if end is None:
        end = len(all_lines)
    shard_lines = all_lines[start:end]
    print(f"[Info] shard size: {len(shard_lines)}")

    out_dir = os.path.dirname(output_jsonl_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_jsonl_path, "w", encoding="utf-8") as fout:
        for line_idx, line in enumerate(tqdm(shard_lines, desc="Computing rewards"), start=start):
            try:
                row = json.loads(line)
            except Exception as e:
                bad = {
                    "line_idx": line_idx,
                    "parse_status": f"json_load_failed: {str(e)}",
                    "raw_line_snippet": line[:500],
                }
                fout.write(json.dumps(bad, ensure_ascii=False) + "\n")
                continue

            iid = row["iid"]
            prompt = row["prompt"]
            candidates = row.get("candidate_loras", [])

            base_dir = os.path.join(image_root, str(iid), "sdv15")
            base_image_paths, base_valid_seeds = collect_seed_image_paths(base_dir, seeds)

            if len(base_image_paths) == 0:
                row["base_scores"] = {}
                row["base_score_mean"] = {}
                row["reward_status"] = "missing_base_images"
                row["candidate_loras"] = candidates
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                continue

            try:
                base_scores_by_scorer, base_score_means = score_base_images(
                    prompt=prompt,
                    base_image_paths=base_image_paths,
                    valid_seeds=base_valid_seeds,
                    scorers=scorers,
                )
            except Exception as e:
                row["base_scores"] = {}
                row["base_score_mean"] = {}
                row["reward_status"] = f"base_scoring_failed: {str(e)}"
                row["candidate_loras"] = candidates
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                continue

            row["base_scores"] = base_scores_by_scorer
            row["base_score_mean"] = base_score_means

            new_candidates = []
            for cand in candidates:
                try:
                    new_cand = score_candidate_lora(
                        prompt=prompt,
                        iid=iid,
                        candidate=cand,
                        image_root=image_root,
                        seeds=seeds,
                        scorers=scorers,
                        base_scores_by_scorer=base_scores_by_scorer,
                        scorer_taus=scorer_taus,
                        scorer_weights=scorer_weights,
                    )
                except Exception as e:
                    new_cand = dict(cand)
                    new_cand["score_status"] = f"candidate_scoring_failed: {str(e)}"
                    new_cand["num_lora_images"] = 0
                    new_cand["num_common_seeds"] = 0
                    new_cand["lora_scores"] = {}
                    new_cand["lora_score_mean"] = {}
                    new_cand["delta_scores"] = {}
                    new_cand["delta_mean"] = {}
                    new_cand["reward"] = None
                    new_cand["reward_parts"] = {}
                    new_cand["reward_weights"] = {}
                new_candidates.append(new_cand)

            row["candidate_loras"] = new_candidates
            row["reward_status"] = "ok"

            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[Done] saved to {output_jsonl_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, required=True)
    parser.add_argument("--image_root", type=str, default="images")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)

    args = parser.parse_args()

    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        total = sum(1 for _ in f)

    end = total if args.end is None else args.end
    seeds = [3428, 746347, 7855463, 64573, 8563]

    compute_rewards(
        input_jsonl_path=args.input_jsonl,
        output_jsonl_path=args.output_jsonl,
        image_root=args.image_root,
        device=args.device,
        start=args.start,
        end=end,
        seeds=seeds,
        scorer_taus=dict(DEFAULT_SCORER_TAUS),
        scorer_weights=dict(DEFAULT_SCORER_WEIGHTS),
    )


