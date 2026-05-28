import torch.nn.functional as F
from transformers import T5TokenizerFast

import torch
import torch.nn as nn

from diffsynth.extensions.ImageQualityMetric import download_preference_model, load_preference_model
from modelscope import dataset_snapshot_download
from PIL import Image
from torch.utils.data import DataLoader
import pandas as pd
# from prompts_dataset import PromptsDataset

from pathlib import Path
import t2v_metrics
import os
import torch
import clip
from PIL import Image

def image_score_1(prompts=[],images=[],model_name='PickScore',cache_dir='./models',device='cuda:7'):
    path = download_preference_model(model_name, cache_dir=cache_dir)
    preference_model = load_preference_model(model_name, device=device, path=path)
    scores = preference_model.score(images,prompts)
    return scores


def score_quality(model_name='MPS',prompts_path='', images_path='', images_original_path='', pattern='alllora', cache_dir='/shark/zhiwen/LoRA-fusion/models',device='cuda:3'):

    df_read = pd.read_csv(prompts_path)
    df_read[model_name] = 0
    df_read[f'{model_name}_original'] = 0

    csv_name = prompts_path.split('/')[-1]
    typename = images_path.split('/')[-3]
    save_dir = f'quality_res/{typename}'
    os.makedirs(save_dir,exist_ok=True)
    save_path = f'{save_dir}/{csv_name.replace(".csv",f"_{model_name}.csv")}'

    path = download_preference_model(model_name, cache_dir=cache_dir)
    preference_model = load_preference_model(model_name, device=device, path=path)

    total_nums=0
    scores_lora = 0
    scores_ori = 0
    for i,row in df_read.iterrows():
        if i % 50 == 0:
            print(i)
        total_nums+=1
        text = row['text']
        lora_1_name = row['top_1_name'].replace('__','.')
        lora_2_name = row['top_2_name'].replace('__','.')
        
        image_name = f'{i}+{lora_1_name}+{lora_2_name}.png'

        image_path = f'{images_path}/{image_name}'
        image_path2 = f'{images_original_path}/{i}+original.png'
        image = Image.open(image_path)
        image_original = Image.open(image_path2)

        scores = preference_model.score([image,image_original],text)
        scores_lora += scores[0]
        scores_ori += scores[1]
        df_read.loc[i,model_name] = scores[0]
        df_read.loc[i,f'{model_name}_original'] = scores[1]
    print(f'scores_lora: {scores_lora/total_nums}')
    print(f'scores_ori: {scores_ori/total_nums}')
    print(f'total_nums: {total_nums}')
    df_read.to_csv(save_path,index=False)
    
        

# compute text images similarity
def text_image_similarity(model_name='VQAScore',prompts_path='', images_path='', images_original_path='',device='cuda:4'):

    clip_flant5_score = t2v_metrics.VQAScore(model='clip-flant5-xxl',device=device) # our recommended scoring model

    df_read = pd.read_csv(prompts_path)

    df_read[model_name] = 0
    df_read[f'{model_name}_original'] = 0
    csv_name = prompts_path.split('/')[-1]
    typename = images_path.split('/')[-3]
    save_dir = f'quality_res/{typename}'
    os.makedirs(save_dir,exist_ok=True)
    save_path = f'{save_dir}/{csv_name.replace(".csv",f"_{model_name}.csv")}'


    total_nums=0
    scores_lora = 0
    scores_ori = 0
    for i,row in df_read.iterrows():
        if i % 50 == 0:
            print(i)
        total_nums+=1
        text = row['text']
        lora_1_name = row['top_1_name'].replace('__','.')
        lora_2_name = row['top_2_name'].replace('__','.')
        
        image_name = f'{i}+{lora_1_name}+{lora_2_name}.png'

        image_path = f'{images_path}/{image_name}'
        image_path2 = f'{images_original_path}/{i}+original.png'
        # image = Image.open(image_path)
        # image_original = Image.open(image_path2)

        # 能不能改成 batch size 推理
        score = clip_flant5_score(images=[image_path,image_path2], texts=[text])
        scores_lora += score[0][0].item()
        scores_ori += score[1][0].item()
        df_read.loc[i,model_name] = score[0][0].item()
        df_read.loc[i,f'{model_name}_original'] = score[1][0].item()
    print(f'图文相似度 scores_lora: {scores_lora/total_nums}')
    print(f'图文相似度 scores_ori: {scores_ori/total_nums}')
    print(f'total_nums: {total_nums}')
    df_read.to_csv(save_path,index=False)
        


# compute avg score in new csv
def avg_score(prompts_path=''):
    save_path = prompts_path.replace('.csv','_avg.csv')
    df_read = pd.read_csv(prompts_path)

    score_columns = ['MPS', 'MPS_original', 'VQAScore', 'VQAScore_original']
    selected_columns = ['model_file'] + score_columns
    selected_df = df_read[selected_columns]
    # 3. 按 name 分组，计算每个 score 列的平均值
    result = selected_df.groupby('model_file', as_index=False)[score_columns].mean()
    result.to_csv(save_path,index=False)

def diffusiondb_score(model_name='MPS',lora_nums=1,prompts_path='', images_path='', images_original_path='', pattern='alllora', cache_dir='/shark/zhiwen/LoRA-fusion/models',device='cuda:3'):
    df_read = pd.read_csv(prompts_path)
    # df_read[model_name] = 0
    # df_read[f'{model_name}_original'] = 0

    csv_name = prompts_path.split('/')[-1]
    typename = images_path.split('/')[-3]
    save_dir = f'quality_res/{typename}'
    os.makedirs(save_dir,exist_ok=True)
    save_path = f'{save_dir}/{csv_name.replace(".csv",f"_{model_name}.csv")}'

    path = download_preference_model(model_name, cache_dir=cache_dir)
    preference_model = load_preference_model(model_name, device=device, path=path)

    total_nums=0
    scores_lora = 0
    scores_ori = 0
    for i,row in df_read.iterrows():
        if i % 50 == 0:
            print(i)
        lora_1_score = float(row['top_1_score'])
        lora_2_score = float(row['top_2_score'])
        if lora_1_score < 0.2:
            continue
        if lora_nums == 2 and lora_2_score < 0.15:
            continue
        total_nums+=1
        text = row['text']
        lora_1_name = row['top_1_name'].replace('__','.')
        lora_2_name = row['top_2_name'].replace('__','.')
        if lora_nums == 1:
            
            image_name = f'{i}+{lora_1_name}--{lora_1_score:.3f}.png'
        else: 
            image_name = f'{i}+{lora_1_name}--{lora_1_score}+{lora_2_name}--{lora_2_score}.png'

        image_path = f'{images_path}/{image_name}'
        image_path2 = f'{images_original_path}/{i}+original.png'
        image = Image.open(image_path)
        image_original = Image.open(image_path2)

        scores = preference_model.score([image,image_original],text)
        scores_lora += scores[0]
        scores_ori += scores[1]
        df_read.loc[i,model_name] = scores[0]
        df_read.loc[i,f'{model_name}_original'] = scores[1]
    print(f'scores_lora: {scores_lora/total_nums}')
    print(f'scores_ori: {scores_ori/total_nums}')
    print(f'total_nums: {total_nums}')
    df_read.to_csv(save_path,index=False)

def diffusiondb_similarity(model_name='VQAScore',lora_nums=1,prompts_path='', images_path='', images_original_path='',device='cuda:4'):
    df_read = pd.read_csv(prompts_path)
    clip_flant5_score = t2v_metrics.VQAScore(model='clip-flant5-xxl',device=device) # our recommended scoring model


    save_path = prompts_path.replace('.csv',f'_{model_name}.csv')


    total_nums=0
    scores_lora = 0
    scores_ori = 0
    for i,row in df_read.iterrows():
        if i % 50 == 0:
            print(i)
        lora_1_score = float(row['top_1_score'])
        lora_2_score = float(row['top_2_score'])
        if lora_1_score < 0.15:
            continue
        if lora_nums == 2 and lora_2_score < 0.15:
            continue
        total_nums+=1
        text = row['text']
        lora_1_name = row['top_1_name'].replace('__','.')
        lora_2_name = row['top_2_name'].replace('__','.')
        if lora_nums == 1:
            
            image_name = f'{i}+{lora_1_name}--{lora_1_score:.3f}.png'
        else: 
            image_name = f'{i}+{lora_1_name}--{lora_1_score}+{lora_2_name}--{lora_2_score}.png'

        image_path = f'{images_path}/{image_name}'
        image_path2 = f'{images_original_path}/{i}+original.png'

        score = clip_flant5_score(images=[image_path,image_path2], texts=[text])
        scores_lora += score[0][0].item()
        scores_ori += score[1][0].item()
    print(f'图文相似度 scores_lora: {scores_lora/total_nums}')
    print(f'图文相似度 scores_ori: {scores_ori/total_nums}')

def score_from_metadata(model_name='MPS',prompts_path='', images_path='', min_lora_score=0, cache_dir='/shark/zhiwen/LoRA-fusion/models',device='cuda:3'):
    path = download_preference_model(model_name, cache_dir=cache_dir)
    preference_model = load_preference_model(model_name, device=device, path=path)
    metadata = pd.read_csv(prompts_path)
    metadata[model_name] = ''
    top0_score = 0
    top1_score = 0
    top2_score = 0
    top3_score = 0
    top1_snum = 0
    top2_snum = 0
    top3_snum = 0
    total_nums = 0
    for i,row in metadata.iterrows():
        case_num = row['case_number']
        text = row['text']
        lora_score = row['score']
        lora_score = [float(i) for i in lora_score.split(',')]
        images = []
        total_nums += 1
        for j in range(0,4):
            image_path = f'{images_path}/image_{case_num}_top{j}.jpg'
            images.append(Image.open(image_path))
        scores = preference_model.score(images,text)
        top0_score += scores[0]
        top1_score += scores[1]
        top2_score += scores[2]
        top3_score += scores[3]
        if scores[1] > scores[0]:
            top1_snum += 1
        if scores[2] > scores[1]:
            top2_snum += 1
        if scores[3] > scores[2]:
            top3_snum += 1
        metadata.loc[i,model_name] = ','.join([str(s) for s in scores])
    metadata.to_csv(prompts_path.replace('.csv',f'_{model_name}.csv'),index=False)
    print(f'top0_score: {top0_score/total_nums}')
    print(f'top1_score: {top1_score/total_nums}, top1_snum: {top1_snum / total_nums}')
    print(f'top2_score: {top2_score/total_nums}, top2_snum: {top2_snum / total_nums}')
    print(f'top3_score: {top3_score/total_nums}, top3_snum: {top3_snum / total_nums}')
    print(f'total_nums: {total_nums}')


def diff_lora_score(res_prompts_path='',min_lora_score=0):
    metadata = pd.read_csv(res_prompts_path)
    res = {
            'top1':[0,0,0],
            'top2':[0,0,0],
            'top3':[0,0,0]
        }
    for i, row in metadata.iterrows():
        lora_score = row['score']
        lora_score = [float(i) for i in lora_score.split(',')]
        MPS_score = row['MPS']
        MPS_score = [float(i) for i in MPS_score.split(',')]
        
        if lora_score[0] > min_lora_score:
            res['top1'][0] += 1
            res['top1'][1] += MPS_score[0]
            res['top1'][2] += MPS_score[1]
            if lora_score[1] > min_lora_score:
                res['top2'][0] += 1
                res['top2'][1] += MPS_score[0]
                res['top2'][2] += MPS_score[2]
                if lora_score[2] > min_lora_score:
                    res['top3'][0] += 1
                    res['top3'][1] += MPS_score[0]
                    res['top3'][2] += MPS_score[3]
    print(f"top1_score: nums :{res['top1'][0]}, {res['top1'][1]/res['top1'][0]}, {res['top1'][2]/res['top1'][0]}")
    print(f"top2_score: nums :{res['top2'][0]}, {res['top2'][1]/res['top2'][0]}, {res['top2'][2]/res['top2'][0]}")
    print(f"top3_score: nums :{res['top3'][0]}, {res['top3'][1]/res['top3'][0]}, {res['top3'][2]/res['top3'][0]}")


def eval_text_alignment(test_prompt,ori_images_path,images_path,topk,model_name,preference_model=None,nums=500,device='cuda'):

    if preference_model is not None:
        clip_flant5_score = preference_model
    else:
        clip_flant5_score = t2v_metrics.VQAScore(model='clip-flant5-xxl',device=device) # our recommended scoring model


    print(f'metrics : VQA, eval test_prompt:{test_prompt}')
    print(f'images_path : {images_path}')

    total_nums=0
    scores_topk = [0,0,0,0]
    num_topk = [0,0,0,0]
    scores_ori = 0

    metadata = pd.read_csv(test_prompt)
    flag = False
    for i,row in metadata.iterrows():
        if i >= nums:
            break

        text = row['text']
        images = []
        total_nums+=1
        for j in range(topk+1):
            if j == 0:
                image_path = f'{ori_images_path}/{i}_top{j}.jpg'
            else:
                image_path = f'{images_path}/{i}_top{j}.jpg'
                if not os.path.exists(image_path):
                    flag = True
                    break
            images.append(image_path)
        if flag:
            break
        score = clip_flant5_score(images=images, texts=[text])

        for j in range(topk+1):
            num_topk[j] += 1
            scores_topk[j] += score[j][0].item()
    
    for j in range(topk+1):
        print(f'top{j}_score, nums {num_topk[j]} : {scores_topk[j]/num_topk[j]}')

def eval_clip_score(test_prompt,ori_images_path,images_path,topk,model_name,device='cuda'):
    
    clip_path = '/shark/zhiwen/LoRA-fusion/models/CLIP/ViT-B-32.pt'
    model, preprocess = clip.load(clip_path, device=device) 

    print(f'metrics : clip socre, eval test_prompt:{test_prompt}')
    print(f'images_path : {images_path}')

    total_nums=0
    scores_topk = [0,0,0,0]
    num_topk = [0,0,0,0]
    scores_ori = 0

    metadata = pd.read_csv(test_prompt)
    flag = False
    for i,row in metadata.iterrows():

        text = row['text']
        images = []
        total_nums+=1
        for j in range(topk+1):
            if j == 0:
                image_path = f'{ori_images_path}/{i}_top{j}.jpg'
            else:
                image_path = f'{images_path}/{i}_top{j}.jpg'
                if not os.path.exists(image_path):
                    flag = True
                    break
            images.append(image_path)
        if flag:
            break
            
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

        for j in range(topk+1):
            num_topk[j] += 1
            scores_topk[j] += similarity[j]
    
    for j in range(topk+1):
        print(f'top{j}_score, nums {num_topk[j]} : {scores_topk[j]/num_topk[j]}')

def eval_image_quality(test_prompt,ori_images_path,images_path,topk,model_name,preference_model,nums=10000,device='cuda'):
    cache_dir='/shark/zhiwen/LoRA-fusion/models'

    path = download_preference_model(model_name, cache_dir=cache_dir)
    preference_model = load_preference_model(model_name, device=device, path=path)

    print(f'metrics {model_name}, eval test_prompt:{test_prompt}')
    print(f'images_path : {images_path}')

    total_nums=0
    scores_topk = [0,0,0,0]
    num_topk = [0,0,0,0]
    scores_ori = 0

    metadata = pd.read_csv(test_prompt)
    flag = False
    for i,row in metadata.iterrows():
        if i >= nums:
            break
        text = row['text']
        images = []
        total_nums+=1
        for j in range(topk+1):
            if j == 0:
                image_path = f'{ori_images_path}/{i}_top{j}.jpg'
                # image_path = f'{ori_images_path}/{i}.jpg'
            else:
                image_path = f'{images_path}/{i}_top{j}.jpg'
                if not os.path.exists(image_path):
                    flag = True
                    continue
            images.append(Image.open(image_path))
        if flag:
            continue
        scores = preference_model.score(images,text)
        for j in range(topk+1):
            num_topk[j] += 1
            scores_topk[j] += scores[j]
            metadata.loc[i,f'top{j}_score'] = scores[j]
        
    
    for j in range(topk+1):
        print(f'top{j}_score, nums {num_topk[j]} : {scores_topk[j]/num_topk[j]}:')
    # metadata.to_csv(test_prompt.replace('.csv',f'_{model_name}.csv'),index=False)


def eval_image_quality_single(test_prompt,ori_images_path,images_path,topk,model_name,nums=10000,device='cuda'):
    cache_dir='/shark/zhiwen/LoRA-fusion/models'

    path = download_preference_model(model_name, cache_dir=cache_dir)
    preference_model = load_preference_model(model_name, device=device, path=path)

    print(f'metrics {model_name}, eval test_prompt:{test_prompt}')
    print(f'images_path : {images_path}')

    total_nums=0
    scores_topk = [0,0,0,0]
    num_topk = [0,0,0,0]
    scores_ori = 0

    metadata = pd.read_csv(test_prompt)
    flag = False
    for i,row in metadata.iterrows():
        if i >= nums:
            break
        text = row['text']
        images = []
        total_nums+=1


        image_path = f'{ori_images_path}/{i}_top0.jpg'
        # image_path = f'{ori_images_path}/{i}.jpg'
        images.append(Image.open(image_path))
        image_path = f'{images_path}/{i}_top{topk}.jpg'

        images.append(Image.open(image_path))

        scores = preference_model.score(images,text)

        
        num_topk[0] += 1
        scores_topk[0] += scores[0]

        num_topk[1] += 1
        scores_topk[1] += scores[1]
    
    for j in range(2):
        print(f'top{j}_score, nums {num_topk[j]} : {scores_topk[j]/num_topk[j]}:')

def eval_image_quality_score(test_prompt,ori_images_path,images_path,topk,model_name,nums=10000,device='cuda'):
    cache_dir='/shark/zhiwen/LoRA-fusion/models'

    path = download_preference_model(model_name, cache_dir=cache_dir)
    preference_model = load_preference_model(model_name, device=device, path=path)

    print(f'metrics {model_name}, eval test_prompt:{test_prompt}')
    print(f'images_path : {images_path}')

    total_nums = 0
    scores_topk = [0,0,0,0]
    num_topk = [0,0,0,0]
    scores_ori = 0

    metadata = pd.read_csv(test_prompt)
    flag = False
    for i,row in metadata.iterrows():
        if i >= nums:
            break
        scores = [row['top_1_score'],row['top_2_score'],row['top_3_score']]
        if scores[0] < 0.45:
            continue
        text = row['text']
        images = []
        total_nums+=1

        image_path = f'{ori_images_path}/{i}_top0.jpg'
        images.append(Image.open(image_path))

        image_path = f'{images_path}/{i}.jpg'
        images.append(Image.open(image_path))

        scores = preference_model.score(images,text)
        for j in range(2):
            num_topk[j] += 1
            scores_topk[j] += scores[j]
    
    for j in range(topk+1):
        print(f'top{j}_score, nums {num_topk[j]} : {scores_topk[j]/num_topk[j]}:')



# PickScore
# Aesthetic
# ImageReward
# HPSv2.1
# MPS
if __name__ == '__main__':

    
    prompts_base_path = '/shark/zhiwen/lora_diff_length/create_dataset/prompts/retrieve_test'
    prompts_base_path = '/shark/zhiwen/lora_diff_length/create_dataset/prompts/retrieve_test/good_loras'
    # prompts_base_path = '/shark/zhiwen/lora_diff_length/test_retriever_rank/prompts/test_prompt'

    model_name = 'MPS'
    device = 'cuda:3'

    ### load model first
    cache_dir='/shark/zhiwen/LoRA-fusion/models'

    path = download_preference_model(model_name, cache_dir=cache_dir)
    preference_model = load_preference_model(model_name, device=device, path=path)

    # vqamodel = t2v_metrics.VQAScore(model='clip-flant5-xxl',device=device) # our recommended scoring model

    topk = 3
    # /shark/zhiwen/lora_diff_length/data/outputs/Good_loras_and_Civitai/retriever_new-all2_crlloss/Good_loras_and_Civitai_2-loras_200_p6
    base_path = '/shark/zhiwen/lora_diff_length/data/eval_retriever_2'
    # base_path = '/shark/zhiwen/lora_diff_length/test_retriever_rank/data/eval_retriever_2/prompts/test_prompt'


    method = 'good-all-cfg25-1.1'
    method = 'good-all-wei=0.512-r8-cfg0.3-crlloss'
    method = 'no_base-wei=0.512-r2-cfg0.3-crlloss'

    retriever_type = 'retriever_enhancer'
    # retriever_type = 'textimage_retriever'
    # retriever_type = 'retriever_rankfix-ep49'

    test_set_list = ['good_loras_1-loras_300_p7_retriever_result','good_loras_2-loras_300_p6_retriever_result','good_loras_3-loras_300_p6_retriever_result']
    # test_set_list = ['no_base-wei=0.512-r2-cfg0.3-crlloss']
    # test_set_list = ['diffusiondb_1000_ex_retriever_result']

    
    for test_set in test_set_list[:]:


        test_prompt = f'{prompts_base_path}/{test_set}.csv'

        ori_images_path = f'{base_path}/{test_set}/original'
        images_path = f'{base_path}/{test_set}/{retriever_type}/{method}'



        # eval_image_quality(test_prompt,ori_images_path,images_path,topk,model_name,nums=305,device='cuda:2')
        
        topk = 3
        # eval_image_quality_single(test_prompt,ori_images_path,images_path,topk,model_name,nums=1000,device='cuda:2')
        eval_image_quality(test_prompt,ori_images_path,images_path,topk,model_name,preference_model=preference_model,nums=500,device=device)

        eval_text_alignment(test_prompt,ori_images_path,images_path,topk,model_name,preference_model=vqamodel,nums=500,device=device)
        # eval_clip_score(test_prompt,ori_images_path,images_path,topk,model_name,device=device)





