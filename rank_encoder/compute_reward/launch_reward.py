import os
import subprocess


def main():
    input_jsonl = "train_dataset/train_prompts_candidates.jsonl"
    image_root = "images"
    output_dir = "reward_outputs3"
    log_dir = "reward_logs3"

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    shard_ranges = [
        (700, 1000),
        (1700, 2000),
        (2700, 3000),
        (3700, 4000),
        (6700, 7000),
        (7700, 8000),
        (8700, 9000),
        (9700, 10000),
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
            "python", "compute_reward.py",
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
