
from CARLoS_prompt import prompts_for_indexing
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
import torch
import os
root_dir = '/shark/zhiwen/LoRAHunter'

# DiffImage-2 是用30step生成的

save_path = '/shark/zhiwen/LoRAHunter/DiffSynth-Studio/DiffImage-2/Qwen'

seeds = [42, 43788, 1234, 9876, 6329]
infer_steps = 30
prompts = prompts_for_indexing()
# print(prompts)
categories = []
for category, sub_categories in prompts.items():
    print(category)
    categories.append(category)
    for sub_category, p in sub_categories.items():
        print(f'--{sub_category}')
        
print(categories)

num = 9
category = categories[num]

pipe = QwenImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device=f"cuda:{num-8}",
    model_configs=[
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    tokenizer_config=ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/"),
)

sub_categories = prompts[category]
for k,v in sub_categories.items():
    print(f"------- {k} --------")
    for i in range(8):
        prompt = v[i]
        img_save_path = f"{save_path}/{category}/{k}/{i}"
        os.makedirs(img_save_path, exist_ok=True)
        for seed in seeds[:2]:
            image = pipe(prompt, seed=seed, num_inference_steps=infer_steps)
            image.save(f"{img_save_path}/{seed}.jpg")

# nohup python generate_img_qwenimg.py > zlog/Qwen-Image-c9.log 2>&1 &