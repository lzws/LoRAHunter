import os,json
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
import torch
from diffusers import DiffusionPipeline
import argparse
from rank_LoRAFusion import replace_target_modules_with_lora_merger,load_lora, clear_lora
from collections import defaultdict

root_dir = '/shark/zhiwen/LoRAHunter'


def load_model(device='cuda',full_lora_path=None):
    pipe = QwenImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
            ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
            ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
        ],
        tokenizer_config=ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/"),
    )
    if full_lora_path is not None:
        pipe.load_lora(pipe.dit, full_lora_path)
    return pipe


def generate_images_two_stage_batched(
    num=0,
    start=0,
    end=100,
    prompt_batch_size=1,
    metadata_path="/shark/zhiwen/LoRAHunter/rank_encoder/train_dataset/train_prompts_candidates.jsonl",
    root_lora_dir="/shark/zhiwen/LoRAHunter/Qwen_LoRA",
    model_path="/shark/zhiwen/LoRAHunter/DiffSynth-Studio/models/AI-ModelScope/stable-diffusion-v1-5",
    image_root="rank_images",
):
    print(f"[Info] num: {num}, start: {start}, end: {end}")
    with open(metadata_path, "r", encoding="utf-8") as f:
        datas = [json.loads(line) for line in f.readlines()]
    print(f"[Info] num_samples: {len(datas)}")

    datas = datas[start:end]

    seeds = [3428, 746347, 7855463, 64573, 8563, 3274385]
    bs = 3
    seeds = seeds[:bs]

    device = f"cuda:{num}"
    print(
        f"[Info] device: {device}, start: {start}, end: {end}, "
        f"num_samples: {len(datas)}, prompt_batch_size: {prompt_batch_size}"
    )

    pipe = load_model(device,None)

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

        img_save_path_sd = os.path.join(image_root, str(iid), "QwenImage")
        os.makedirs(img_save_path_sd, exist_ok=True)

        if not all_images_exist(img_save_path_sd):
            base_todo.append({
                "iid": iid,
                "prompt": prompt,
            })

    print(f"[Stage 1] base todo: {len(base_todo)}")

    for batch_idx, batch_items in enumerate(chunk_list(base_todo, prompt_batch_size)):
        batch_prompts = [x["prompt"] for x in batch_items]
        batch_iids = [x["iid"] for x in batch_items]

        try:
            for seed in seeds:
                # generator = torch.Generator(device=device).manual_seed(seed)

                image = pipe(
                    prompt=batch_prompts[0],
                    negative_prompt="",
                    num_inference_steps=30,
                    seed=seed,
                    height=512,
                    width=512,
                )
                images = [image]

                for iid, image in zip(batch_iids, images):
                    img_save_path_sd = os.path.join(image_root, str(iid), "QwenImage")
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
                "model_id": "QwenImage",
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

    moelora_target_modules = "to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1"
    model_ = replace_target_modules_with_lora_merger(
        getattr(pipe, 'dit'),
        target_modules=moelora_target_modules.split(","),
    )
    setattr(pipe, 'dit', model_.to(pipe.device,pipe.torch_dtype))
    # load_lora(pipe, lora_path)

    for lora_idx, (model_file, items) in enumerate(lora_to_items.items()):
        # model_id = os.path.basename(model_file).replace(".safetensors", "")
        model_id = model_file.split("/")[:2]
        model_id = "__".join(model_id)

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

            # pipe.load_lora_weights(full_lora_path)

            clear_lora(pipe)
            load_lora(pipe, full_lora_path)

            for batch_items in chunk_list(todo_items, prompt_batch_size):
                batch_prompts = [x["prompt"] for x in batch_items]
                batch_iids = [x["iid"] for x in batch_items]

                try:
                    for seed in seeds:
                        # generator = torch.Generator(device=device).manual_seed(seed)

                        image = pipe(
                            prompt=batch_prompts[0],
                            negative_prompt="",
                            num_inference_steps=30,
                            seed=seed,
                            height=512,
                            width=512,
                        )
                        images = [image]
                        for iid, image in zip(batch_iids, images):
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


    


