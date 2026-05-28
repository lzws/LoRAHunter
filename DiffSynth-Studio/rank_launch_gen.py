import os
import math
import subprocess


def count_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def main():
    # 0-3200 dev2
    # 3200-6400 dev3
    # 6400-9600 h20
    # 9600-10600 a800
    # 10600-13800 123
    # 13800 - 15800 123
    # 15800-17800 h20
    # 17800-19800 dev2
    # 19800-21800 dev3
    metadata_path = "rank_dataset/train_prompts2_candidates.jsonl"
    num_gpus = 8
    prompt_batch_size = 1

    model_path = " "
    root_lora_dir = "/shark/zhiwen/LoRAHunter/Qwen_LoRA"
    image_root = "rank_images_dmd"

    os.makedirs("rank_logs", exist_ok=True)

    total = count_lines(metadata_path)
    chunk_size = math.ceil(total / num_gpus)

    chunk_size = 250

    print(f"[Launcher] total samples: {total}")
    print(f"[Launcher] num_gpus: {num_gpus}")
    print(f"[Launcher] chunk_size: {chunk_size}")

    procs = []
    log_files = []

    for gpu_id in range(num_gpus):
        start = 19800 + gpu_id * chunk_size
        # end = min((gpu_id + 1) * chunk_size, total)
        end = start + chunk_size

        if start >= end:
            continue

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        cmd = [
            "python", "rank_generate_imgs_dmd.py",
            "--gpu", "0",   # 进程内部只看到一张卡
            "--start", str(start),
            "--end", str(end),
            "--prompt_batch_size", str(prompt_batch_size),
            "--metadata_path", metadata_path,
            "--root_lora_dir", root_lora_dir,
            "--model_path", model_path,
            "--image_root", image_root,
        ]

        log_path = f"rank_logs/gpu_dev3_dmd2_{gpu_id}.log"
        log_f = open(log_path, "w", encoding="utf-8")
        log_files.append(log_f)

        print(f"[Launcher] Launching GPU {gpu_id}: start={start}, end={end}")
        print(f"[Launcher] CMD: {' '.join(cmd)}")

        p = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=log_f)
        procs.append((gpu_id, p))

    for gpu_id, p in procs:
        ret = p.wait()
        print(f"[Launcher] GPU {gpu_id} finished with return code {ret}")

    for f in log_files:
        f.close()

    print("[Launcher] All processes finished.")


if __name__ == "__main__":
    main()

# nohup python rank_launch_gen.py > rank_logs/launchdmd2_1.log 2>&1 &
