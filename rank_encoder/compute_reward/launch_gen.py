import os
import math
import subprocess


def count_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def main():
    metadata_path = "/shark/zhiwen/LoRAHunter/rank_encoder/train_dataset/train_prompts_candidates_all.jsonl"
    num_gpus = 8
    prompt_batch_size = 8

    model_path = "/shark/zhiwen/LoRAHunter/DiffSynth-Studio/models/AI-ModelScope/stable-diffusion-v1-5"
    root_lora_dir = "/shark/zhiwen/LoRAHunter/sd_lora/sd_lora_1"
    image_root = "images2"

    os.makedirs("logs", exist_ok=True)

    total = count_lines(metadata_path)
    chunk_size = math.ceil(total / num_gpus)
    # 0-4000 dev2
    # 4000-6000 h20
    # 6000-10000 dev3
    chunk_size = 1300

    print(f"[Launcher] total samples: {total}")
    print(f"[Launcher] num_gpus: {num_gpus}")
    print(f"[Launcher] chunk_size: {chunk_size}")

    procs = []
    log_files = []

    for gpu_id in range(num_gpus):
        start = 1300*8 + gpu_id * chunk_size
        # end = min((gpu_id + 1) * chunk_size, total)
        end = start + chunk_size

        if start >= end:
            continue

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        cmd = [
            "python", "gen_images.py",
            "--gpu", "0",   # 进程内部只看到一张卡
            "--start", str(start),
            "--end", str(end),
            "--prompt_batch_size", str(prompt_batch_size),
            "--metadata_path", metadata_path,
            "--root_lora_dir", root_lora_dir,
            "--model_path", model_path,
            "--image_root", image_root,
        ]

        log_path = f"logs/gpu_dev3_{gpu_id}.log"
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

# nohup python launch_gen.py > zlog/launch2.log 2>&1 &
