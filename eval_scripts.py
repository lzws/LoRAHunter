from ImageQualityMetric import download_preference_model, load_preference_model

import os
import json
from PIL import Image
import clip
import torch
def score_quality(model_name='PickScore',prompts_path='', methods=[], images_original_path='', seed_num=0, cache_dir='/shark/zhiwen/LoRA-fusion/models',device='cuda:3'):

    test_data_path = prompts_path
    with open(test_data_path, 'r') as f:
        test_datas = [json.loads(line) for line in f.readlines()]

    base_path = "/shark/zhiwen/LoRAHunter/sd_encoder/outputs"

    path = download_preference_model(model_name, cache_dir=cache_dir)
    preference_model = load_preference_model(model_name, device=device, path=path)
    total_nums = 0
    all_scores=[0] * (len(methods)+1)
    for i,row in enumerate(test_datas[75:]):
        total_nums+=1
        text = row['prompt']
        ori_img = f"{images_original_path}/{i}_{seed_num}.png"

        images_path = [ori_img]
        for method in methods:
            lora_img = f"{base_path}/{method}/rerank_{i}_{seed_num}.png"
            images_path.append(lora_img)

        images = [Image.open(img) for img in images_path]

        scores = preference_model.score(images,text)
        for j in range(len(methods)+1):
            all_scores[j] += scores[j]

    print(f"eval : {model_name}, seed_num: {seed_num}")
    print(f'scores_ori: {all_scores[0]/total_nums}')
    for n in range(len(methods)):
        print(f'{methods[n]}: {all_scores[n+1]/total_nums}')
    
    print(f'total_nums: {total_nums}')


def eval_clip_score(prompts_path='', methods=[], images_original_path='', seed_num=0,device='cuda'):
    
    clip_path = '/shark/zhiwen/LoRA-fusion/models/CLIP/ViT-B-32.pt'
    model, preprocess = clip.load(clip_path, device=device) 

    base_path = "/shark/zhiwen/LoRAHunter/sd_encoder/outputs"

    test_data_path = prompts_path
    with open(test_data_path, 'r') as f:
        test_datas = [json.loads(line) for line in f.readlines()]

    print(f'metrics : clip socre, eval test_prompt:{test_data_path}')
    # print(f'images_path : {images_path}')

    flag = False
    total_nums = 0
    all_scores=[0] * (len(methods)+1)
    for i,row in enumerate(test_datas[75:]):
        total_nums+=1
        text = row['prompt']
        ori_img = f"{images_original_path}/{i}_{seed_num}.png"

        images_path = [ori_img]
        for method in methods:
            lora_img = f"{base_path}/{method}/rerank_{i}_{seed_num}.png"
            images_path.append(lora_img)

        text_input = clip.tokenize([text],truncate=True).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text_input)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        images = [preprocess(Image.open(path)) for path in images_path]
        image_input = torch.stack(images).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        similarity = (image_features @ text_features.T).squeeze(1)
        similarity = similarity.cpu().numpy().tolist()

        for j in range(len(methods)+1):
            all_scores[j] += similarity[j]
        
    
    print(f"eval : CLIP, seed_num: {seed_num}")
    print(f'scores_ori: {all_scores[0]/total_nums}')
    for n in range(len(methods)):
        print(f'{methods[n]}: {all_scores[n+1]/total_nums}')
    
    print(f'total_nums: {total_nums}')

if __name__ == '__main__':
    # PickScore MPS ImageReward HPSv2.1
    # 
    # 'testdata_250_totalpool_stylus'
    model_name = "HPSv2.1"
    prompts_path = '/shark/zhiwen/LoRAHunter/sd_encoder/test_data/250_clipemb-7_all_res_combinations_reank.jsonl'
    methods = ['250_clipemb-7_all_res_combinations_reank','testdata_250_totalpool_stylus']
    images_original_path = "/shark/zhiwen/LoRAHunter/sd_encoder/outputs/sdv15_250_clipemb-7_all_res_combinations_reank"
    score_quality(model_name=model_name, prompts_path=prompts_path, methods=methods[:], images_original_path=images_original_path, seed_num=4, device='cuda:0')
    # eval_clip_score(prompts_path=prompts_path, methods=methods[:], images_original_path=images_original_path, seed_num=3, device='cuda:1')