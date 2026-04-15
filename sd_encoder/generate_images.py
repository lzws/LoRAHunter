
import torch
import os,json
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import gc
lora_base_path = "/shark/zhiwen/LoRAHunter/sd_lora/sd_lora_1"



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



def generate_images(num=0):
    test_data_path = 'test_data/250_clipemb-7_all_res_combinations_reank.jsonl'
    with open(test_data_path, 'r') as f:
        test_datas = [json.loads(line) for line in f.readlines()]

    lora_metadata_path = '/shark/zhiwen/LoRAHunter/SD_adapter_metadata/sd_lora_1/exist_file_adapters.jsonl'
    with open(lora_metadata_path, 'r') as f:
        lora_metadatas = [json.loads(line) for line in f.readlines()]

    lora_metadatas_map = {str(one['adapter_id']): one for one in lora_metadatas}

    save_path = "outputs/testdata_250_debias_realistic_dataset2-all"
    os.makedirs(save_path, exist_ok=True)

    torch_dtype=torch.bfloat16,

    # num = 0
    device = f"cuda:{num}"

    start = 0 + num*250
    end = start + 250

    start,end = 0,250
    seed_num = 0
    seeds = [42,6734,3252,23498,62991]

    # load sd 1.5 pipe
    # pipe = StableDiffusionPipeline.from_pretrained(
    #     "/shark/zhiwen/LoRAHunter/DiffSynth-Studio/models/AI-ModelScope/stable-diffusion-v1-5", 
    #     torch_dtype=torch_dtype
    # )
    model_path = '/shark/zhiwen/LoRAHunter/DiffSynth-Studio/models/lzwecnu/SDv1-5-model/realisticVisionV60B1_v51VAE.safetensors'
    pipe = StableDiffusionPipeline.from_single_file(model_path, torch_dtype=torch_dtype, local_files_only=True)

    pipe.safety_checker = None
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config,algorithm_type="dpmsolver++")
    pipe = pipe.to(device)

    

    for i,data in enumerate(test_datas[start:end]):

        if os.path.exists(f"{save_path}/rerank_{i+start}_{seed_num}.png"):
            continue

        # rerank_results = data['rerank_results']
        combination_results = data['combination_results']
        res = combination_results['diverse_combinations'][seed_num]
        adapter_names = []
        alpha = max(1/len(rerank_results),0.5)
        for keyword,loras in rerank_results.items():
            lora_list = [x["lora"] for x in sorted(loras, key=lambda x: int(x["score"]), reverse=True)]
            # top1 = lora_list[0]
            if len(lora_list) == 0:
                continue
            for j in range(1):
                topk = lora_list[j]
                model_file = lora_metadatas_map[topk]['model_file']
                lora_path = f"{lora_base_path}/{model_file}"
                # pipe.load_lora(pipe.dit, lora_path, alpha=alpha)
                if str(topk) not in adapter_names:
                    pipe.load_lora_weights(lora_path, adapter_name=str(topk))
                    adapter_names.append(str(topk))
        pipe.set_adapters(adapter_names,adapter_weights=[alpha]*len(adapter_names))

        
        prompt = data['prompt']
        seed = data['seed']
        image = pipe(
                    prompt + " realistic, high quality", 
                    negative_prompt=NEGATIVE_PROMPT_STR, 
                    num_inference_steps=35,
                    num_images_per_prompt=1,
                    generator=torch.Generator(device=device).manual_seed(seed),
                    guidance_scale=7
                ).images[0]
        image.save(f"{save_path}/rerank_{i+start}.png")

        pipe.unload_lora_weights()
        gc.collect()
        torch.cuda.empty_cache()


def generate_sd(num=0):
    test_data_path = 'test_data/250_clipemb-7_all_res_combinations_reank.jsonl'
    with open(test_data_path, 'r') as f:
        test_datas = [json.loads(line) for line in f.readlines()]


    torch_dtype=torch.bfloat16,

    # num = 0
    device = f"cuda:{num}"

    start = 0 + num*250
    end = start + 250

    start, end = 0, 250

    seed_num = num
    seeds = [42,6734,3252,23498,62991]

    save_path = f"outputs/sdv15_{test_data_path.split('/')[-1].split('.')[0]}"
    os.makedirs(save_path, exist_ok=True)

    # load sd 1.5 pipe
    # pipe = StableDiffusionPipeline.from_pretrained(
    #     "/shark/zhiwen/LoRAHunter/DiffSynth-Studio/models/AI-ModelScope/stable-diffusion-v1-5", 
    #     torch_dtype=torch_dtype
    # )
    model_path = '/shark/zhiwen/LoRAHunter/DiffSynth-Studio/models/lzwecnu/SDv1-5-model/realisticVisionV60B1_v51VAE.safetensors'
    pipe = StableDiffusionPipeline.from_single_file(model_path, torch_dtype=torch_dtype, local_files_only=True)

    pipe.safety_checker = None
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, algorithm_type="dpmsolver++")
    pipe = pipe.to(device)

    for i,data in enumerate(test_datas[start:end]):


        prompt = data['prompt']
        seed = seeds[seed_num]
        image = pipe(
                    prompt + ' realistic, high quality', 
                    negative_prompt=NEGATIVE_PROMPT_STR, 
                    num_inference_steps=35,
                    num_images_per_prompt=1,
                    generator=torch.Generator(device=device).manual_seed(seed),
                    guidance_scale=7
                ).images[0]
        image.save(f"{save_path}/{i+start}_{seed_num}.png")



def generate_images_com(num=0):
    test_data_path = 'test_data/250_clipemb-7_all_res_combinations_reank.jsonl'
    with open(test_data_path, 'r') as f:
        test_datas = [json.loads(line) for line in f.readlines()]

    lora_metadata_path = '/shark/zhiwen/LoRAHunter/SD_adapter_metadata/sd_lora_1/exist_file_adapters.jsonl'
    with open(lora_metadata_path, 'r') as f:
        lora_metadatas = [json.loads(line) for line in f.readlines()]

    lora_metadatas_map = {str(one['adapter_id']): one for one in lora_metadatas}

    save_path = f"outputs/{test_data_path.split('/')[-1].split('.')[0]}"
    os.makedirs(save_path, exist_ok=True)

    torch_dtype=torch.bfloat16,

    # num = 0
    device = f"cuda:{num}"

    start = 0 + num*250
    end = start + 250

    start,end = 0,250
    seed_num = num
    seeds = [42,6734,3252,23498,62991]

    # load sd 1.5 pipe
    # pipe = StableDiffusionPipeline.from_pretrained(
    #     "/shark/zhiwen/LoRAHunter/DiffSynth-Studio/models/AI-ModelScope/stable-diffusion-v1-5", 
    #     torch_dtype=torch_dtype
    # )
    model_path = '/shark/zhiwen/LoRAHunter/DiffSynth-Studio/models/lzwecnu/SDv1-5-model/realisticVisionV60B1_v51VAE.safetensors'
    pipe = StableDiffusionPipeline.from_single_file(model_path, torch_dtype=torch_dtype, local_files_only=True)

    pipe.safety_checker = None
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config,algorithm_type="dpmsolver++")
    pipe = pipe.to(device)

    

    for i,data in enumerate(test_datas[start:end]):

        if os.path.exists(f"{save_path}/rerank_{i+start}_{seed_num}.png"):
            continue

        # rerank_results = data['rerank_results']
        combination_results = data['combination_results']
        res = combination_results['diverse_combinations'][seed_num]
        adapter_names = []
        
        for model_file in res['model_files']:

            # model_file = lora_metadatas_map[topk]['model_file']
            lora_path = f"{lora_base_path}/{model_file}"
            topk = model_file.split('/')[-1].split('.')[0]
            if str(topk) not in adapter_names:
                pipe.load_lora_weights(lora_path, adapter_name=str(topk))
                adapter_names.append(str(topk))
        alpha = max(1/len(adapter_names),0.45)
        pipe.set_adapters(adapter_names,adapter_weights=[alpha]*len(adapter_names))

        
        prompt = data['prompt']
        seed = seeds[seed_num]
        image = pipe(
                    prompt + " realistic, high quality", 
                    negative_prompt=NEGATIVE_PROMPT_STR, 
                    num_inference_steps=35,
                    num_images_per_prompt=1,
                    generator=torch.Generator(device=device).manual_seed(seed),
                    guidance_scale=7
                ).images[0]
        image.save(f"{save_path}/rerank_{i+start}_{seed_num}.png")

        pipe.unload_lora_weights()
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == '__main__':
    generate_images_com(4)
    # generate_sd(4)

# nohup python generate_images.py > zlog/generate_images_com_4.log 2>&1 &