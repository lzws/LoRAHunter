import os
import json
import argparse
from collections import defaultdict

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler


NEGATIVE_PROMPTS = [
    "realisticvision-negative-embedding",
    "ng_deepnegative_v1_75t",
    "bad anatomy",
    "bad proportions",
    "blurry",
    "cloned face",
    "cropped",
    "deformed",
    "dehydrated",
    "disfigured",
    "duplicate",
    "error",
    "extra arms",
    "extra fingers",
    "extra legs",
    "extra limbs",
    "fused fingers",
    "gross proportions",
    "jpeg artifacts",
    "long neck",
    "(low quality: 2)",
    "lowres",
    "malformed limbs",
    "missing arms",
    "missing legs",
    "morbid",
    "mutated hands",
    "mutation",
    "mutilated",
    "out of frame",
    "poorly drawn face",
    "poorly drawn hands",
    "signature",
    "text",
    "too many fingers",
    "ugly",
    "username",
    "watermark",
    "(worst quality:2)",
]
NEGATIVE_PROMPT_STR = ", ".join(NEGATIVE_PROMPTS)


def generate_images_two_stage_batched(
    num=0,
    start=0,
    end=100,
    prompt_batch_size=4,
    metadata_path="/shark/zhiwen/LoRAHunter/rank_encoder/train_dataset/train_prompts_candidates.jsonl",
    root_lora_dir="/shark/zhiwen/LoRAHunter/sd_lora/sd_lora_1",
    model_path="/shark/zhiwen/LoRAHunter/DiffSynth-Studio/models/AI-ModelScope/stable-diffusion-v1-5",
    image_root="images",
):
    with open(metadata_path, "r", encoding="utf-8") as f:
        datas = [json.loads(line) for line in f.readlines()]

    datas = datas[start:end]

    seeds = [3428, 746347, 7855463, 64573, 8563, 3274385]
    bs = 5
    seeds = seeds[:bs]

    device = f"cuda:{num}"
    print(
        f"[Info] device: {device}, start: {start}, end: {end}, "
        f"num_samples: {len(datas)}, prompt_batch_size: {prompt_batch_size}"
    )

    # pipe = StableDiffusionPipeline.from_pretrained(
    #     model_path,
    #     torch_dtype=torch.bfloat16
    # )

    # pipe.safety_checker = None
    # pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    # pipe = pipe.to(device)

    prompt_bias = ' realistic, high quality'
    model_path = '/shark/zhiwen/LoRAHunter/DiffSynth-Studio/models/lzwecnu/SDv1-5-model/realisticVisionV60B1_v51VAE.safetensors'
    print(f"load from {model_path}")
    pipe = StableDiffusionPipeline.from_single_file(
        model_path,
        torch_dtype=torch.bfloat16,
        local_files_only=True
    )
    pipe.safety_checker = None
    pipe.requires_safety_checker = False
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        algorithm_type="dpmsolver++"
    )
    pipe = pipe.to(device)

    # 可选优化
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass

    error_log_path = f"error_log_gpu{num}.jsonl"

    def log_error(payload):
        with open(error_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def all_images_exist(save_dir):
        return all(os.path.exists(os.path.join(save_dir, f"{seed}.jpg")) for seed in seeds)

    def chunk_list(lst, chunk_size):
        for i in range(0, len(lst), chunk_size):
            yield lst[i:i + chunk_size]

    # ======================================================
    # Stage 1: base images, batched by prompts
    # ======================================================
    print("[Stage 1] Generating base images in prompt batches...")

    base_todo = []
    for data in datas:
        iid = data["iid"]
        prompt = data["prompt"]

        img_save_path_sd = os.path.join(image_root, str(iid), "sdv15")
        os.makedirs(img_save_path_sd, exist_ok=True)

        if not all_images_exist(img_save_path_sd):
            base_todo.append({
                "iid": iid,
                "prompt": prompt,
            })

    print(f"[Stage 1] base todo: {len(base_todo)}")

    for batch_idx, batch_items in enumerate(chunk_list(base_todo, prompt_batch_size)):
        batch_prompts = [x["prompt"]+prompt_bias for x in batch_items]
        batch_iids = [x["iid"] for x in batch_items]

        try:
            for seed in seeds:
                generator = torch.Generator(device=device).manual_seed(seed)

                images = pipe(
                    prompt=batch_prompts,
                    negative_prompt=[NEGATIVE_PROMPT_STR] * len(batch_prompts),
                    num_inference_steps=35,
                    num_images_per_prompt=1,
                    generator=generator,
                    guidance_scale=7.0
                ).images

                for iid, image in zip(batch_iids, images):
                    img_save_path_sd = os.path.join(image_root, str(iid), "sdv15")
                    os.makedirs(img_save_path_sd, exist_ok=True)
                    image.save(os.path.join(img_save_path_sd, f"{seed}.jpg"))

            if (batch_idx + 1) % 20 == 0:
                print(f"[Stage 1] processed {batch_idx + 1} base batches")

        except Exception as e:
            print("++" * 10)
            print(f"[Base Batch Error] iids: {batch_iids}, Error: {e}")
            print("++" * 10)
            log_error({
                "iid_list": batch_iids,
                "model_id": "sdv15",
                "msg": str(e),
                "stage": "base_batch"
            })

    # ======================================================
    # Build inverted index: model_file -> list of (iid, prompt)
    # ======================================================
    lora_to_items = defaultdict(list)
    for data in datas:
        iid = data["iid"]
        prompt = data["prompt"]
        candidates = data["candidate_loras"][:20]

        # 同一个 prompt 下去重，避免 candidate_loras 内重复
        seen = set()
        for candidate in candidates:
            model_file = candidate["model_file"]
            if model_file in seen:
                continue
            seen.add(model_file)

            lora_to_items[model_file].append({
                "iid": iid,
                "prompt": prompt,
            })

    print(f"[Stage 2] Total unique LoRAs to process in this shard: {len(lora_to_items)}")

    # ======================================================
    # Stage 2: grouped by LoRA, batched by prompts
    # ======================================================
    for lora_idx, (model_file, items) in enumerate(lora_to_items.items()):
        model_id = os.path.basename(model_file).replace(".safetensors", "")
        full_lora_path = model_file if os.path.isabs(model_file) else os.path.join(root_lora_dir, model_file)

        try:
            # 过滤出当前 LoRA 下还没生成完的 prompt
            todo_items = []
            for item in items:
                iid = item["iid"]
                img_save_path_lora = os.path.join(image_root, str(iid), model_id)
                os.makedirs(img_save_path_lora, exist_ok=True)

                if not all_images_exist(img_save_path_lora):
                    todo_items.append(item)

            if len(todo_items) == 0:
                if (lora_idx + 1) % 50 == 0:
                    print(f"[Stage 2] skip done LoRA {lora_idx + 1}/{len(lora_to_items)}")
                continue

            try:
                pipe.unload_lora_weights()
            except Exception:
                pass

            pipe.load_lora_weights(full_lora_path)

            for batch_items in chunk_list(todo_items, prompt_batch_size):
                batch_prompts = [x["prompt"] for x in batch_items]
                batch_iids = [x["iid"] for x in batch_items]

                try:
                    # for seed in seeds:
                    #     generator = torch.Generator(device=device).manual_seed(seed)

                    #     images = pipe(
                    #         prompt=batch_prompts,
                    #         negative_prompt=[""] * len(batch_prompts),
                    #         num_inference_steps=35,
                    #         num_images_per_prompt=1,
                    #         generator=generator,
                    #         guidance_scale=7.0,
                    #     ).images

                    #     for iid, image in zip(batch_iids, images):
                    #         img_save_path_lora = os.path.join(image_root, str(iid), model_id)
                    #         os.makedirs(img_save_path_lora, exist_ok=True)
                    #         image.save(os.path.join(img_save_path_lora, f"{seed}.jpg"))

                    expanded_prompts = []
                    expanded_negative_prompts = []
                    expanded_iids = []
                    expanded_seeds = []
                    generators = []

                    for iid, prompt in zip(batch_iids, batch_prompts):
                        for seed in seeds:
                            expanded_prompts.append(prompt+prompt_bias)
                            expanded_negative_prompts.append(NEGATIVE_PROMPT_STR)
                            expanded_iids.append(iid)
                            expanded_seeds.append(seed)
                            generators.append(torch.Generator(device=device).manual_seed(seed))

                    images = pipe(
                        prompt=expanded_prompts,
                        negative_prompt=expanded_negative_prompts,
                        num_inference_steps=35,
                        num_images_per_prompt=1,
                        generator=generators,
                        guidance_scale=7.0,
                    ).images

                    for iid, seed, image in zip(expanded_iids, expanded_seeds, images):
                        img_save_path_lora = os.path.join(image_root, str(iid), model_id)
                        os.makedirs(img_save_path_lora, exist_ok=True)
                        image.save(os.path.join(img_save_path_lora, f"{seed}.jpg"))

                except Exception as e:
                    print("++" * 10)
                    print(f"[LoRA Batch Gen Error] model_id: {model_id}, iids: {batch_iids}, Error: {e}")
                    print("++" * 10)
                    log_error({
                        "iid_list": batch_iids,
                        "model_id": model_id,
                        "model_file": model_file,
                        "msg": str(e),
                        "stage": "lora_batch_gen"
                    })

            try:
                pipe.unload_lora_weights()
            except Exception:
                pass

            if (lora_idx + 1) % 20 == 0:
                print(f"[Stage 2] processed LoRA {lora_idx + 1}/{len(lora_to_items)}")

        except Exception as e:
            print("++" * 10)
            print(f"[LoRA Load Error] model_id: {model_id}, model_file: {model_file}, Error: {e}")
            print("++" * 10)
            log_error({
                "iid": None,
                "model_id": model_id,
                "model_file": model_file,
                "msg": str(e),
                "stage": "lora_load"
            })

    print("[Done] Generation finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0, help="GPU id visible inside this process")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=100)
    parser.add_argument("--prompt_batch_size", type=int, default=4)

    parser.add_argument(
        "--metadata_path",
        type=str,
        default="/shark/zhiwen/LoRAHunter/rank_encoder/train_dataset/train_prompts_candidates.jsonl"
    )
    parser.add_argument(
        "--root_lora_dir",
        type=str,
        default="/shark/zhiwen/LoRAHunter/sd_lora/sd_lora_1"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/shark/zhiwen/LoRAHunter/DiffSynth-Studio/models/AI-ModelScope/stable-diffusion-v1-5"
    )
    parser.add_argument(
        "--image_root",
        type=str,
        default="images"
    )

    args = parser.parse_args()

    generate_images_two_stage_batched(
        num=args.gpu,
        start=args.start,
        end=args.end,
        prompt_batch_size=args.prompt_batch_size,
        metadata_path=args.metadata_path,
        root_lora_dir=args.root_lora_dir,
        model_path=args.model_path,
        image_root=args.image_root,
    )
