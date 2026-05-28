import os
import json
import math
import glob
import argparse
from tqdm import tqdm


def compute_weight(reward: float, rank: int):
    if reward <= 0:
        return 0.05
    return 1.0 / math.sqrt(rank)


def process_one_row(row, min_candidates=2, topk_pos=8, topk_neg=2):
    prompt = row.get("prompt", None)
    iid = row.get("iid", None)
    source = row.get("source", None)
    source_id = row.get("source_id", None)
    candidates = row.get("candidate_loras", [])

    valid = []
    for cand in candidates:
        if cand.get("score_status") != "ok":
            continue
        reward = cand.get("reward", None)
        if reward is None:
            continue

        valid.append({
            "iid": iid,
            "source": source,
            "source_id": source_id,
            "prompt": prompt,
            "model_file": cand.get("model_file", None),
            "reward": float(reward),
        })

    if len(valid) < min_candidates:
        return []

    valid.sort(key=lambda x: x["reward"], reverse=True)

    pos = [x for x in valid if x["reward"] > 0][:topk_pos]
    neg = [x for x in valid if x["reward"] <= 0][:topk_neg]

    kept = pos + neg
    if len(kept) < min_candidates:
        return []

    kept.sort(key=lambda x: x["reward"], reverse=True)

    for i, item in enumerate(kept):
        rank = i + 1
        item["weight"] = float(compute_weight(item["reward"], rank))

    return kept


def iter_input_files(input_paths):
    all_files = []
    for p in input_paths:
        if os.path.isdir(p):
            all_files.extend(sorted(glob.glob(os.path.join(p, "*.jsonl"))))
        else:
            all_files.extend(sorted(glob.glob(p)))
    return sorted(set(all_files))


def convert_reward_jsonls_to_gcl_jsonl(
    input_paths,
    output_jsonl,
    min_candidates=2,
    topk_pos=8,
    topk_neg=2,
):
    input_files = iter_input_files(input_paths)
    print(f"[Info] found {len(input_files)} input files")

    out_dir = os.path.dirname(output_jsonl)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    total_rows = 0
    total_pairs = 0
    skipped_rows = 0

    with open(output_jsonl, "w", encoding="utf-8") as fout:
        for file_path in input_files:
            print(f"[Info] processing: {file_path}")
            with open(file_path, "r", encoding="utf-8") as f:
                for line in tqdm(f, desc=os.path.basename(file_path)):
                    total_rows += 1
                    try:
                        row = json.loads(line)
                    except Exception:
                        skipped_rows += 1
                        continue

                    samples = process_one_row(
                        row,
                        min_candidates=min_candidates,
                        topk_pos=topk_pos,
                        topk_neg=topk_neg,
                    )

                    if len(samples) == 0:
                        skipped_rows += 1
                        continue

                    for sample in samples:
                        fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
                        total_pairs += 1

    print(f"[Done] saved to: {output_jsonl}")
    print(f"[Stat] total_rows={total_rows}")
    print(f"[Stat] total_pairs={total_pairs}")
    print(f"[Stat] skipped_rows={skipped_rows}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=str, nargs="+", required=True)
    parser.add_argument("--output_jsonl", type=str, required=True)
    parser.add_argument("--min_candidates", type=int, default=2)
    parser.add_argument("--topk_pos", type=int, default=8)
    parser.add_argument("--topk_neg", type=int, default=2)
    args = parser.parse_args()

    convert_reward_jsonls_to_gcl_jsonl(
        input_paths=args.inputs,
        output_jsonl=args.output_jsonl,
        min_candidates=args.min_candidates,
        topk_pos=args.topk_pos,
        topk_neg=args.topk_neg,
    )



'''
python rank_convert_reward_to_gcl.py \
  --inputs "/shark/zhiwen/LoRAHunter/DiffSynth-Studio/reward_outputs3/*.jsonl" \
  --output_jsonl train_gcl_topk_dev.jsonl \
  --topk_pos 8 \
  --topk_neg 2
'''