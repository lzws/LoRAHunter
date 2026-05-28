import os
import subprocess


def main():
    input_jsonl = "rank_dataset/train_prompts2_candidates.jsonl"
    image_root = "rank_images_dmd"
    output_dir = "reward_outputs3"
    log_dir = "reward_logs3"

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    # 0-3200 dev2
    # 3200-6400 dev3

    # 10600-13800 123
    # 13800 - 15800 123
    # 17800-19800 dev2
    # 19800-21800 dev3

    shard_ranges = [
        (0, 800),
        (800, 1600),
        (1600, 2400),
        (2400, 3200),
        (3200, 4000),
        (4000, 4800),
        (4800, 5600),
        (5600, 6400),
    ]

    procs = []
    log_files = []

    for gpu_id, (start, end) in enumerate(shard_ranges):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        # 限制每个进程的 CPU 线程数，避免 8 个进程把 CPU 抢爆
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        env["OPENBLAS_NUM_THREADS"] = "1"
        env["NUMEXPR_NUM_THREADS"] = "1"
        env["TOKENIZERS_PARALLELISM"] = "false"

        output_jsonl = os.path.join(output_dir, f"reward_{start}_{end}.jsonl")
        log_path = os.path.join(log_dir, f"gpu_{gpu_id}_{start}_{end}.log")

        cmd = [
            "python", "rank_compute_reward.py",
            "--input_jsonl", input_jsonl,
            "--output_jsonl", output_jsonl,
            "--image_root", image_root,
            "--device", "cuda:0",
            "--start", str(start),
            "--end", str(end),
        ]

        print(f"[Launch] GPU {gpu_id} | range=({start}, {end})")
        print("[CMD]", " ".join(cmd))

        log_f = open(log_path, "w", encoding="utf-8")
        log_files.append(log_f)

        p = subprocess.Popen(
            cmd,
            env=env,
            stdout=log_f,
            stderr=log_f,
        )
        procs.append((gpu_id, start, end, p))

    for gpu_id, start, end, p in procs:
        ret = p.wait()
        print(f"[Done] GPU {gpu_id} | range=({start}, {end}) | return_code={ret}")

    for f in log_files:
        f.close()

    print("[All Done] all reward processes finished.")


if __name__ == "__main__":
    main()
