import os,json
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
import torch
from diffusers import DiffusionPipeline
import argparse
from rank_LoRAFusion import replace_target_modules_with_lora_merger,load_lora, clear_lora
from collections import defaultdict
from diffsynth.core import load_state_dict
import math

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

    # snapshot_download("MusePublic/Qwen-Image-Distill", allow_file_pattern="qwen_image_distill_3step.safetensors", cache_dir="models")
    # lora_state_dict = load_state_dict("models/MusePublic/Qwen-Image-Distill/qwen_image_distill_3step.safetensors")
    # lora_state_dict = {i.replace("base_model.model.", ""): j for i, j in lora_state_dict.items()}
    # pipe.load_lora(pipe.dit, state_dict=lora_state_dict)

    if full_lora_path is not None:
        pipe.load_lora(pipe.dit, full_lora_path)
    return pipe


def generate_imgs(device='cuda',
    start=0,end=500,
    test_data_path="/mnt/nas2/zhiwen/LoRAHunter/DiffSynth-Studio-main/test_data/coco_part_500_loraverse_qwenlora_com.jsonl",
    root_lora_dir="/shark/zhiwen/LoRAHunter/Qwen_LoRA",
    image_root="outputs2"
):
    print(f"[Info] device: {device}, start: {start}, end: {end}")

    with open(test_data_path, "r", encoding="utf-8") as f:
        datas = [json.loads(line) for line in f.readlines()]
    print(f"[Info] num_samples: {len(datas)}")

    # load metadata
    lora_metadata_path = 'rank_dataset/Available_LoRA_all.jsonl'
    with open(lora_metadata_path, 'r') as f:
        lora_metadatas = [json.loads(line) for line in f.readlines()]

    lora_metadatas_map = {str(one['model_id']): one for one in lora_metadatas}

    # load pipe
    pipe = load_model(device=device)
    moelora_target_modules = "to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1"
    model_ = replace_target_modules_with_lora_merger(
        getattr(pipe, 'dit'),
        target_modules=moelora_target_modules.split(","),
    )
    setattr(pipe, 'dit', model_.to(pipe.device,pipe.torch_dtype))

    seeds = [42, 6734, 3252, 23498, 62991, 4324, 54894, 12047592, 163884, 63485, 927429, 238451]
    seeds = seeds[:10]

    save_path = f"{image_root}/{test_data_path.split('/')[-1].split('.')[0]}"
    os.makedirs(save_path, exist_ok=True)


    for i, data in enumerate(datas[start:end]):
        lora_combinations = data['lora_combinations']

        prompt = data['prompt']
        iid = data['iid']

        for seed_num in range(10):
            save_file = f"{save_path}/{iid}_{seed_num}.jpg"
            if os.path.exists(save_file):
                print(f"[SKIP] {save_file} exists")
                continue

            model_files = []
            alphas = []

            if 'loraverse' in test_data_path and 'gclembset' not in test_data_path:
                if seed_num < len(lora_combinations):
                    res = lora_combinations[seed_num]
                    # loras = res['loras']
                    for k,l in res.items():
                        model_files.append(l)
            else:
                if seed_num < len(lora_combinations):
                    res = lora_combinations[seed_num]
                    loras = res['loras']
                    for l in loras:
                        model_files.append(l['model_file'])

            if len(model_files) > 0:
                if len(model_files) == 1:
                    alpha = [0.95]
                if len(model_files) == 2:
                    alpha = [0.6, 0.5]
                if len(model_files) == 3:
                    alpha = [0.6, 0.45, 0.35]
                if len(model_files) > 3:
                    alpha = [0.35] * len(model_files)
            
            clear_lora(pipe)
            for j in range(len(model_files)):
                full_lora_path = os.path.join(root_lora_dir, model_files[j])
                try:
                    load_lora(pipe, full_lora_path, alpha=alpha[j])
                except Exception as e:
                    print(f"[ERROR] {full_lora_path} not found,error :{str(e)}")
            
            image = pipe(
                prompt=prompt,
                negative_prompt="",
                num_inference_steps=30,
                seed=seeds[seed_num],
                height=768,
                width=768,
                # cfg_scale=1, 
                # exponential_shift_mu=math.log(2.5),
            )
            image.save(save_file)

def generate_imgs_ori(device='cuda',
    start=0,end=500,
    test_data_path="/shark/zhiwen/LoRAHunter/DiffSynth-Studio/test_data/stylus_500_calllora_composer.jsonl",
    root_lora_dir="/shark/zhiwen/LoRAHunter/Qwen_LoRA",
    image_root="outputs2"
):
    print(f"[Info] device: {device}, start: {start}, end: {end}")

    with open(test_data_path, "r", encoding="utf-8") as f:
        datas = [json.loads(line) for line in f.readlines()]
    print(f"[Info] num_samples: {len(datas)}")

    # load metadata
    lora_metadata_path = '/shark/zhiwen/LoRAHunter/DiffSynth-Studio/rank_dataset/Available_LoRA_all.jsonl'
    with open(lora_metadata_path, 'r') as f:
        lora_metadatas = [json.loads(line) for line in f.readlines()]

    lora_metadatas_map = {str(one['model_id']): one for one in lora_metadatas}

    # load pipe
    pipe = load_model(device=device)
    moelora_target_modules = "to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1"
    model_ = replace_target_modules_with_lora_merger(
        getattr(pipe, 'dit'),
        target_modules=moelora_target_modules.split(","),
    )
    setattr(pipe, 'dit', model_.to(pipe.device,pipe.torch_dtype))

    seeds = [42, 6734, 3252, 23498, 62991, 4324, 54894, 12047592, 163884, 63485, 927429, 238451]
    seeds = seeds[:10]

    save_path = f"{image_root}/QwenImage_diffusiondb200"
    os.makedirs(save_path, exist_ok=True)

    for i in range(start, end):
        data = datas[i]
        prompt = data["prompt"]

        for seed_num in range(10):
            seed = seeds[seed_num]
            output_path = os.path.join(save_path, f"{i}_{seed_num}.jpg")
            if os.path.exists(output_path):
                continue
            image = pipe(
                prompt=prompt,
                negative_prompt="",
                num_inference_steps=30,
                seed=seed,
                height=768,
                width=768,
                # cfg_scale=1, 
                # exponential_shift_mu=math.log(2.5),
            )
            image.save(output_path)  

if __name__ == "__main__":
    num=7
    device=f"cuda:{num}"
    start = 0 + num * 25
    end = start + 25
    # test_data_path='test_data/diffusiondb_test_200_concept_qwenlora_gcl-trainfilter-110_set.jsonl'
    test_data_path='test_data/diffuisondb_200_loraverse_qwenlora_com.jsonl'
    generate_imgs(device=device,start=start,end=end,test_data_path=test_data_path, image_root="outputs2")

# nohup python test_generate_imgs.py > zlog/test_log/generate_loraverse-db200_7.log 2>&1 &