import torch.nn.functional as F
from transformers import T5TokenizerFast

import torch
import torch.nn as nn

# from diffsynth.extensions.ImageQualityMetric import download_preference_model, load_preference_model
# from modelscope import dataset_snapshot_download
from PIL import Image
# from torch.utils.data import DataLoader
import pandas as pd
# from prompts_dataset import PromptsDataset

from pathlib import Path
# import t2v_metrics
import os
import torch
import clip
from PIL import Image
import json
def eval_clip_score(device='cuda'):
    
    clip_path = '/shark/zhiwen/LoRA-fusion/models/CLIP/ViT-B-32.pt'
    model, preprocess = clip.load(clip_path, device=device) 

    
    test_data_path = '/shark/zhiwen/LoRAHunter/DiffSynth-Studio/z_retrieval_res/retrieval_testdata_100_extract.jsonl'
    with open(test_data_path, 'r') as f:
        test_datas = [json.loads(line) for line in f.readlines()]

    print(f'metrics : clip socre, eval test_prompt:{test_data_path}')
    # print(f'images_path : {images_path}')
    num_topk = [0,0]
    scores_topk = [0,0]
    flag = False
    for i,row in enumerate(test_datas[:-1]):

        text = row['prompt']
        ori_img = f"/shark/zhiwen/LoRAHunter/sd_encoder/outputs/sdv15_realistic/{i}.png"
        # lora_img = f"/shark/zhiwen/LoRAHunter/sd_encoder/outputs/retrieval_testdata_100_realistic/rerank_{i}.png"
        # lora_img = f"/shark/zhiwen/LoRAHunter/sd_encoder/outputs/retrieval_testdata_100/rerank_{i}.png"
        
        lora_img = f"/shark/zhiwen/LoRAHunter/sd_encoder/outputs/retrieval_testdata_100_realistic_train/rerank_{i}.png"

        images = [ori_img, lora_img]

        text_input = clip.tokenize([text],truncate=True).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text_input)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        images = [preprocess(Image.open(path)) for path in images]
        image_input = torch.stack(images).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        similarity = (image_features @ text_features.T).squeeze(1)
        similarity = similarity.cpu().numpy().tolist()

        for j in range(2):
            num_topk[j] += 1
            scores_topk[j] += similarity[j]
    
    for j in range(2):
        print(f'top{j}_score, nums {num_topk[j]} : {scores_topk[j]/num_topk[j]}')

def score_quality(model_name='MPS',prompts_path='', images_path='', images_original_path='', pattern='alllora', cache_dir='/shark/zhiwen/LoRA-fusion/models',device='cuda:3'):

    test_data_path = '/shark/zhiwen/LoRAHunter/DiffSynth-Studio/retrieval_testdata_100_extract.jsonl'
    with open(test_data_path, 'r') as f:
        test_datas = [json.loads(line) for line in f.readlines()]


    path = download_preference_model(model_name, cache_dir=cache_dir)
    preference_model = load_preference_model(model_name, device=device, path=path)

    for i,row in enumerate(test_datas[:-1]):
        total_nums+=1
        text = row['prompt']
        ori_img = f"/shark/zhiwen/LoRAHunter/DiffSynth-Studio/outputs/retrieval_testdata_100/original/qwen_{i}.png"
        lora_img = f"/shark/zhiwen/LoRAHunter/DiffSynth-Studio/outputs/retrieval_testdata_100/rerank_{i}.png"

        images = [ori_img, lora_img]

        image = Image.open(lora_img)
        image_original = Image.open(ori_img)

        scores = preference_model.score([image,image_original],text)
        scores_lora += scores[0]
        scores_ori += scores[1]


    print(f'scores_lora: {scores_lora/total_nums}')
    print(f'scores_ori: {scores_ori/total_nums}')
    print(f'total_nums: {total_nums}')


if __name__ == "__main__":
    eval_clip_score()