
import torch
import os,json
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import gc
lora_base_path = "/shark/zhiwen/LoRAHunter/sd_lora/sd_lora_1"
from collections import OrderedDict

import random

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



def generate_images_cache(num=0, start=0, end=250, model_type='original', max_lora_cache=128):
    test_data_path = '/shark/zhiwen/LoRAHunter/sd_encoder/test_data/coco_part_500_qwenemb.jsonl'
    with open(test_data_path, 'r') as f:
        test_datas = [json.loads(line) for line in f.readlines()]

    lora_metadata_path = '/shark/zhiwen/LoRAHunter/SD_adapter_metadata/sd_lora_1/exist_file_adapters.jsonl'
    with open(lora_metadata_path, 'r') as f:
        lora_metadatas = [json.loads(line) for line in f.readlines()]

    lora_metadatas_map = {str(one['adapter_id']): one for one in lora_metadatas}

    save_path = f"outputs2/hunter_{test_data_path.split('/')[-1].split('.')[0]}/{model_type}"
    os.makedirs(save_path, exist_ok=True)

    device = f"cuda:{num}"
    torch_dtype = torch.float16

    # 你需要确认这个路径在你的环境里是对的
    # 如果外面已经有全局 lora_base_path，也可以删掉这一行
    # lora_base_path = "/shark/zhiwen/LoRAHunter/your_lora_root"

    seeds = [42, 6734, 3252, 23498, 62991, 4324, 54894, 12047592, 163884, 63485, 927429, 238451]
    seeds = seeds[:10]
    generators = [torch.Generator(device=device).manual_seed(seed) for seed in seeds]

    print(f"start: {start}, end: {end}, model_type: {model_type}, max_lora_cache: {max_lora_cache}")

    # load sd 1.5 pipe
    if model_type == 'original':
        pipe = StableDiffusionPipeline.from_pretrained(
            "/shark/zhiwen/LoRAHunter/DiffSynth-Studio/models/AI-ModelScope/stable-diffusion-v1-5",
            torch_dtype=torch_dtype
        )
        prompt_bias = ''

    elif model_type == 'realistic':
        prompt_bias = ' realistic, high quality'
        model_path = '/shark/zhiwen/LoRAHunter/DiffSynth-Studio/models/lzwecnu/SDv1-5-model/realisticVisionV60B1_v51VAE.safetensors'
        print(f"load from {model_path}")
        pipe = StableDiffusionPipeline.from_single_file(
            model_path,
            torch_dtype=torch_dtype,
            local_files_only=True
        )
    else:
        raise ValueError(f"unsupported model_type: {model_type}")

    pipe.safety_checker = None
    pipe.requires_safety_checker = False
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        algorithm_type="dpmsolver++"
    )
    pipe = pipe.to(device)

    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("xformers enabled")
    except Exception as e:
        print(f"xformers not available: {e}")

    try:
        pipe.enable_vae_slicing()
    except Exception as e:
        print(f"enable_vae_slicing failed: {e}")

    # ========== LRU LoRA Cache ==========
    # key: adapter_name
    # value: {"model_file": ..., "lora_path": ...}
    loaded_adapters = OrderedDict()

    # 统计信息
    cache_hit = 0
    cache_miss = 0
    cache_evict = 0
    fallback_full_reload = 0

    has_delete_adapters = hasattr(pipe, "delete_adapters")
    print(f"pipe has delete_adapters: {has_delete_adapters}")

    def get_adapter_name(model_file):
        return model_file.replace('/', '__').replace('.', '_')

    def disable_all_adapters():
        try:
            pipe.set_adapters([], [])
        except Exception:
            pass

    def unload_all_loras():
        nonlocal loaded_adapters
        try:
            pipe.unload_lora_weights()
        except Exception as e:
            print(f"unload_lora_weights failed: {e}")
        loaded_adapters.clear()

    def evict_one_if_needed(current_required_names=None):
        """
        如果缓存超限，尝试淘汰最久未使用的 adapter。
        current_required_names: 当前这轮马上要用到的 adapter，不能被淘汰
        """
        nonlocal cache_evict, fallback_full_reload

        if current_required_names is None:
            current_required_names = set()
        else:
            current_required_names = set(current_required_names)

        while len(loaded_adapters) >= max_lora_cache:
            # 找到最旧且不在当前必需集合里的 adapter
            victim_name = None
            victim_info = None
            for name, info in loaded_adapters.items():
                if name not in current_required_names:
                    victim_name = name
                    victim_info = info
                    break

            # 如果缓存里所有 adapter 都是当前必需的，就没法逐出
            if victim_name is None:
                break

            disable_all_adapters()

            if has_delete_adapters:
                try:
                    pipe.delete_adapters(victim_name)
                    loaded_adapters.pop(victim_name, None)
                    cache_evict += 1
                    print(f"[LRU EVICT] {victim_name}")
                except Exception as e:
                    print(f"delete_adapters failed for {victim_name}, err={e}")
                    print("[FALLBACK] unload all loras")
                    unload_all_loras()
                    fallback_full_reload += 1
                    break
            else:
                # 无法删单个 adapter，只能整体清空
                print("[FALLBACK] pipe has no delete_adapters, unload all loras")
                unload_all_loras()
                fallback_full_reload += 1
                break

    def ensure_lora_loaded(model_file, current_required_names=None):
        nonlocal cache_hit, cache_miss

        adapter_name = get_adapter_name(model_file)
        lora_path = f"{lora_base_path}/{model_file}"

        if adapter_name in loaded_adapters:
            loaded_adapters.move_to_end(adapter_name)
            cache_hit += 1
            return adapter_name

        cache_miss += 1

        evict_one_if_needed(current_required_names=current_required_names)

        # 如果 fallback 导致全清了，这里直接重新加载当前 adapter
        try:
            pipe.load_lora_weights(lora_path, adapter_name=adapter_name)
            loaded_adapters[adapter_name] = {
                "model_file": model_file,
                "lora_path": lora_path,
            }
            loaded_adapters.move_to_end(adapter_name)
            print(f"[CACHE MISS] loaded lora: {adapter_name}")
            return adapter_name
        except Exception as e:
            print(f"load lora failed: {lora_path}, err={e}")
            return None

    for i, data in enumerate(test_datas[start:end]):
        lora_combinations = data['lora_combinations']

        prompt = data['prompt']
        iid = data['iid']

        for seed_num in range(10):
            save_file = f"{save_path}/rerank_{iid}_{seed_num}.png"
            # if os.path.exists(save_file):
            #     continue

            model_files = []
            if seed_num < len(lora_combinations):
                res = lora_combinations[seed_num]
                # model_files.append(res['model_file'])

                loras = res['loras']
                for l in loras:
                    model_files.append(l['model_file'])

            # 先计算当前需要的 adapter 名称集合，避免刚要用的被 LRU 淘汰
            current_required_names = [get_adapter_name(mf) for mf in model_files]

            adapter_names = []
            for model_file in model_files:
                adapter_name = ensure_lora_loaded(
                    model_file,
                    current_required_names=current_required_names
                )
                if adapter_name is not None and adapter_name not in adapter_names:
                    adapter_names.append(adapter_name)

            if len(adapter_names) > 0:
                alpha = max(1 / len(adapter_names), 0.45)
                if len(adapter_names) == 1:
                    alpha = 0.8
                if len(adapter_names) == 2:
                    alpha = 0.4
                if len(adapter_names) == 3:
                    alpha = 0.3
                if len(adapter_names) > 3:
                    alpha = 0.25

                try:
                    pipe.set_adapters(adapter_names, adapter_weights=[alpha] * len(adapter_names))
                except Exception as e:
                    print(f"set_adapters failed, adapters={adapter_names}, err={e}")
                    continue
            else:
                disable_all_adapters()

            try:
                with torch.inference_mode():
                    image = pipe(
                        prompt + prompt_bias,
                        negative_prompt=NEGATIVE_PROMPT_STR,
                        num_inference_steps=35,
                        num_images_per_prompt=1,
                        generator=generators[seed_num],
                        guidance_scale=7
                    ).images[0]

                image.save(save_file)

            except Exception as e:
                print(f"generate failed: idx={i+start}, seed_num={seed_num}, err={e}")

    print(f"done. images saved to: {save_path}")
    print(f"cache_hit={cache_hit}, cache_miss={cache_miss}, cache_evict={cache_evict}, fallback_full_reload={fallback_full_reload}")



if __name__ == '__main__':
    # generate_images_com(num=7,start=450,end=500,model_type='realistic')
    # realistic original
    # generate_sd(num=2,start=0,end=250,model_type='realistic')
    # generate_images(num=7,start=450,end=500,model_type='realistic')
    num = 7
    start = 0 + num * 63
    end = start + 63
    generate_images_cache(num=num,start=start,end=end,model_type='realistic',max_lora_cache=256)

# nohup python generate_images_2.py > zlog/test_log/generate_qwenemb_realistic_7.log 2>&1 &