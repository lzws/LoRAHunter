from CARLoS_prompt import prompts_for_indexing
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
import torch
import torch.nn as nn
import os
from diffsynth.utils.lora import GeneralLoRALoader
from diffsynth.core import load_state_dict
import math
import torch.nn.functional as F
from LoRAFusion import replace_target_modules_with_lora_merger, load_lora, clear_lora   
import pandas as pd

test_obj_names = ['dog8', 'fancy_boot', 'robot_toy', 'wolf_plushie']
test_style_names = ['Line', 'Poly', 'Snoopy', 'Vector']
def main():
    pipe = QwenImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=f"cuda:4",
        model_configs=[
            ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
            ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
            ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
        ],
        tokenizer_config=ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/"),
    )


    # moelora_target_modules = "to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1"
    # model_ = replace_target_modules_with_lora_merger(
    #     getattr(pipe, 'dit'),
    #     target_modules=moelora_target_modules.split(","),
    # )
    # # load model
    # model_path = '/shark/zhiwen/LoRAHunter/DiffSynth-Studio/models/lora_merger/LoRA_Merger_train_data/21/epoch-4.safetensors'
    # state_dict = load_state_dict(model_path)
    # state_dict = {k.replace('.transformer_blocks', 'transformer_blocks'): v for k, v in state_dict.items()}
    # load_moe_res = model_.load_state_dict(state_dict,strict=False)
    # if len(load_moe_res[1]) > 0:
    #     print(f"Warning, LoRA key mismatch! Unexpected keys in LoRA checkpoint: {load_moe_res[1]}")
    # setattr(pipe, 'dit', model_.to(pipe.device,pipe.torch_dtype))

    # load_lora(pipe, "/shark/zhiwen/LoRAHunter/DiffSynth-Studio/models/train/Qwen-Image_lora/epoch-4.safetensors")
    lora_map = {
        'backpack_dog':'/shark/zhiwen/LoRAHunter/DiffSynth-Studio/models/train/Qwen-Image_lora/obj_lora/backpack_dog/epoch-3.safetensors',
        'clock':'/shark/zhiwen/LoRAHunter/DiffSynth-Studio/models/train/Qwen-Image_lora/obj_lora/clock/epoch-4.safetensors',
        '3D_Chibi': '/shark/zhiwen/LoRAHunter/DiffSynth-Studio/models/train/Qwen-Image_lora/style_lora/3D_Chibi/epoch-1.safetensors',
        'Chinese_ink': '/shark/zhiwen/LoRAHunter/DiffSynth-Studio/models/train/Qwen-Image_lora/style_lora/Chinese_Ink/epoch-1.safetensors',
        # 测试
        'robot_toy': '/shark/zhiwen/LoRAHunter/DiffSynth-Studio/models/train/Qwen-Image_lora/obj_lora/robot_toy/epoch-4.safetensors',
        'wolf_plushie': '/shark/zhiwen/LoRAHunter/DiffSynth-Studio/models/train/Qwen-Image_lora/obj_lora/wolf_plushie/epoch-4.safetensors',
        'Line': '/shark/zhiwen/LoRAHunter/DiffSynth-Studio/models/train/Qwen-Image_lora/style_lora/Line/epoch-1.safetensors',
        'Poly': '/shark/zhiwen/LoRAHunter/DiffSynth-Studio/models/train/Qwen-Image_lora/style_lora/Poly/epoch-1.safetensors'
    }

    test_datas = [
        {'loras':['backpack_dog','3D_Chibi'], 'prompt':'In 3D_Chibi style，A Dog-themed backpack on table'},
        {'loras':['backpack_dog','clock','3D_Chibi'], 'prompt':'In 3D_Chibi style，A Dog-themed backpack and a yellow clock on the table'},
        {'loras':['robot_toy','Line'], 'prompt':'In Stick Style, A robot toy on the table'},
        {'loras':['robot_toy','wolf_plushie','Line'], 'prompt':'In Stick Style, A robot toy and wolf plushie on the table'},
    ]
    i = 0
    for data in test_datas:
        lora_paths = [lora_map[l] for l in data['loras']]
        clear_lora(pipe)
        for lora_path in lora_paths:
            pipe.load_lora(pipe.dit, lora_path,alpha=0.5)
            # load_lora(pipe, lora_path)
        prompt = data['prompt']
        image = pipe(prompt, seed=42, num_inference_steps=30)
        image.save(f"zmerger/oori_{i}.png")
        i+=1
    

if __name__ == "__main__":

    main()
    # metadata_path = '/shark/zhiwen/LoRAHunter/DiffSynth-Studio/LoRA_Merger_train_data.csv'

    # df = pd.read_csv(metadata_path)

    # obj_data_dir = '/shark/zhiwen/LoRAHunter/DiffSynth-Studio/data/style_data'
    # obj_data_files = os.listdir(obj_data_dir)
    # obj_names = [f.split('.')[0] for f in obj_data_files]
    # train_obj_nmaes = []
    # for index, row in df.iterrows():
    #     if row['lora_name'] in obj_names and row['lora_type'] == 'style':
    #         if row['lora_name'] not in train_obj_nmaes:
    #             train_obj_nmaes.append(row['lora_name'])

    # test_obj_nmaes = []
    # for n in obj_names:
    #     if n not in train_obj_nmaes:
    #         test_obj_nmaes.append(n)
    # print(f"测试obj lora：{test_obj_nmaes}")

