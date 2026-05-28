
from CARLoS_prompt import prompts_for_indexing
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from diffsynth.core import load_state_dict
from diffsynth.core.loader import hash_model_file,convert_keys_dict_to_single_str,load_keys_dict
import torch
import os,json


root_dir = '/shark/zhiwen/LoRAHunter'
save_path = '/shark/zhiwen/LoRAHunter/DiffSynth-Studio/DiffImage-2'

seeds = [42, 43788, 1234, 9876, 6329]
infer_steps = 30
prompts = prompts_for_indexing()
# print(prompts)
categories = []
for category, sub_categories in prompts.items():
    # print(category)
    categories.append(category)
    # for sub_category, p in sub_categories.items():
    #     print(f'--{sub_category}')
        
print(categories)


lora_files = '/shark/zhiwen/LoRAHunter/Available_LoRA_carlos_tags2.jsonl'
lora_root_path = '/shark/zhiwen/LoRAHunter/Qwen_LoRA'
lora_datas = []
with open(lora_files, 'r') as f:
    lora_datas = [json.loads(line) for line in f.readlines()]
print(f"共 {len(lora_datas)} 个lora")

# for lora in lora_datas[:]:
#     model_id = lora['model_id']
#     model_file = lora['model_file']
#     carlos_tags = lora['carlos_tags']
#     if model_id in ['MusePublic/Qwen-Image-Distill']:

        
#         pipe.load_lora(pipe.dit, f'{lora_root_path}/{model_file}', hotload=False)

#         break
#         if model_id == 'MusePublic/Qwen-Image-Distill':
#             break


# 每个LoRA 选3个类，每个类有4个小类，每个小类选前4条prompt，每条prompt生成2张图 = 3 x 4 x 4 x 2 = 96
# 400 - 800 个lora dev_2 机器
# 800 - 830 个lora dev_3 机器
# 830 - 860 个lora dev_3 机器 a11 a12 a13 gpu0 1 2
# 860 - 880 个lora dev_3 机器 a14 a15 gpu 3 4
# 880 - 940 个lora dev_3 机器 a16 a17 a18 gpu 0 1 2
# 940 - 1000 dev_2 a19,a20,a21
# 1400 - 1800 dev_2 机器

# 


num = 7
start = 1400 + num * 50
end = start + 50
print(f"============= {start}-{end} ================")
pipe = None
for lora in reversed(lora_datas[start+40:end]):
    model_id = lora['model_id']
    model_file = lora['model_file']
    if model_id in ['FFFFFFoo/EmotionalPhotography','Hanjiangxue0010/face','qiaozhuooo/create_aigc_model_6','qiaozhuooo/create_aigc_model_8','diffsynth-i2L-gallery/Z-Image-oriental_aesthetic-realistic_rendering-suggestive-1770466842.566137','diffsynth-i2L-gallery/Z-Image-realistic_portrait-roleplay-prop_detail-1770363034.962795']:
        continue
    lora_path = f'{lora_root_path}/{model_file}'
    if not os.path.exists(lora_path):
        with open('skip_lora.jsonl', 'a', encoding='utf-8') as f:
            json_str = json.dumps({'model_id': model_id,'msg':f"路径不存在：{lora_path}"}, ensure_ascii=False)
            f.write(json_str + '\n')
        continue
    model_hash = hash_model_file(f'{lora_root_path}/{model_file}', with_shape=False)
    if model_hash == '38c6e7e3e93b59d38cf8813572cad7a6':
        with open('skip_lora.jsonl', 'a', encoding='utf-8') as f:
            json_str = json.dumps({'model_id': model_id}, ensure_ascii=False)
            f.write(json_str + '\n')
        continue
    
    print(f'========= {model_id} ===========')
    del pipe
    pipe = QwenImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=f"cuda:{num-5}",
        model_configs=[
            ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
            ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
            ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
        ],
        tokenizer_config=ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/"),
    )
    
    pipe.load_lora(pipe.dit, f'{lora_root_path}/{model_file}')

    
    carlos_tags = lora['carlos_tags']
    c_tag_num = 0
    for c_tag in carlos_tags[:]:
        if c_tag_num > 2:
            break
        if c_tag not in prompts:
            continue
        c_tag_num += 1
        sub_categories = prompts[c_tag]

        for k,v in sub_categories.items():
            # print(f"------- {k} --------")
            for i in range(4):
                prompt = v[i]
                img_save_path = f"{save_path}/{model_id}/{c_tag}/{k}/{i}"
                os.makedirs(img_save_path, exist_ok=True)
                for seed in seeds[:2]:
                    if os.path.exists(f"{img_save_path}/{seed}.jpg"):
                        continue
                    image = pipe(prompt, seed=seed, num_inference_steps=infer_steps)
                    image.save(f"{img_save_path}/{seed}.jpg")
    # pipe.clear_lora()


# nohup python generate_img_qwenimg_lora.py > zlog/Qwen-Image-lora-c7-1.log 2>&1 &