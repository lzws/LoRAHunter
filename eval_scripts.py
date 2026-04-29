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

    with open(prompts_path, 'r') as f:
        test_datas = [json.loads(line) for line in f.readlines()]

    print(f'metrics: CLIP Score, eval test_prompt: {prompts_path}')

    total_nums = 0
    all_scores = [0.0] * (len(methods) + 1)

    for global_i, row in enumerate(test_datas[:], start=0):
        text = row['prompt']
        iid = str(row["iid"]).replace("/", "_")

        # original图片如果还是旧命名方式
        ori_img = f"{images_original_path}/{global_i}_{seed_num}.png"

        # LoRA方法图片如果已经改成iid命名
        images_path = [ori_img]
        for method in methods:
            lora_img = f"{base_path}/{method}/rerank_{iid}_{seed_num}.png"
            images_path.append(lora_img)

        valid = True
        for p in images_path:
            if not os.path.exists(p):
                print(f"[MISSING] {p}")
                valid = False
                break
        if not valid:
            continue

        total_nums += 1

        text_input = clip.tokenize([text], truncate=True).to(device)

        with torch.no_grad():
            text_features = model.encode_text(text_input)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        images = [preprocess(Image.open(path).convert("RGB")) for path in images_path]
        image_input = torch.stack(images).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        similarity = (image_features @ text_features.T).squeeze(1)
        clip_scores = torch.clamp(similarity, min=0.0) * 100.0
        clip_scores = clip_scores.cpu().numpy().tolist()

        for j in range(len(methods) + 1):
            all_scores[j] += clip_scores[j]

    print(f"eval: CLIP Score, seed_num: {seed_num}")
    if total_nums == 0:
        print("No valid samples found.")
        return

    print(f'scores_ori: {all_scores[0] / total_nums:.4f}')
    for n in range(len(methods)):
        print(f'{methods[n]}: {all_scores[n + 1] / total_nums:.4f}')



def score_quality_multi_seed_per_prompt(
    model_name='PickScore',
    prompts_path='',
    methods=[],
    images_original_path='',
    num_seeds=10,
    cache_dir='/shark/zhiwen/LoRA-fusion/models',
    device='cuda:3',
    start_idx=0,
    base_path="/shark/zhiwen/LoRAHunter/sd_encoder/outputs2",
    model_type="realistic",
):
    with open(prompts_path, 'r') as f:
        test_datas = [json.loads(line) for line in f.readlines()]

    print(f"total prompts: {len(test_datas)}")
    print(f"metrics: {model_name}, eval test_prompt: {prompts_path}")

    path = download_preference_model(model_name, cache_dir=cache_dir)
    preference_model = load_preference_model(model_name, device=device, path=path)

    all_prompt_scores = [0.0] * (len(methods) + 1)
    total_prompts = 0

    for global_i, row in enumerate(test_datas[start_idx:], start=start_idx):
        text = row["prompt"]
        iid = str(row["iid"]).replace("/", "_")

        prompt_scores_sum = [0.0] * (len(methods) + 1)
        valid_seed_count = 0

        for seed_num in range(num_seeds):
            ori_img = f"{images_original_path}/{global_i}_{seed_num}.png"
            # 如果你的 original 也是 iid 命名，就改成：
            # ori_img = f"{images_original_path}/rerank_{iid}_{seed_num}.png"

            images_path = [ori_img]
            for method in methods:
                lora_img = f"{base_path}/{method}/{model_type}/rerank_{iid}_{seed_num}.png"
                images_path.append(lora_img)

            if not all(os.path.exists(p) for p in images_path):
                continue

            try:
                images = [Image.open(img).convert("RGB") for img in images_path]
                scores = preference_model.score(images, text)
            except Exception as e:
                print(f"[Warning] failed on prompt idx={global_i}, iid={iid}, seed={seed_num}, err={e}")
                continue

            for j in range(len(methods) + 1):
                prompt_scores_sum[j] += float(scores[j])

            valid_seed_count += 1

        if valid_seed_count == 0:
            continue

        for j in range(len(methods) + 1):
            all_prompt_scores[j] += prompt_scores_sum[j] / valid_seed_count

        total_prompts += 1

    print(f"\neval: {model_name}, num_seeds: {num_seeds}")
    if total_prompts == 0:
        print("No valid prompts found.")
        return

    print(f"scores_ori: {all_prompt_scores[0] / total_prompts:.4f}")
    for n, method in enumerate(methods):
        print(f"{method}: {all_prompt_scores[n + 1] / total_prompts:.4f}")

    print(f"total_prompts: {total_prompts}")


def eval_clip_score_multi_seed_per_prompt(
    prompts_path,
    images_original_path,
    methods,
    device="cuda:0",
    start_idx=0,
    num_seeds=10,
):
    clip_path = '/shark/zhiwen/LoRA-fusion/models/CLIP/ViT-B-32.pt'
    model, preprocess = clip.load(clip_path, device=device)
    model = model.to(device)
    model.eval()
    print("model param device:", next(model.parameters()).device)

    base_path = "/shark/zhiwen/LoRAHunter/sd_encoder/outputs2"

    with open(prompts_path, 'r') as f:
        test_datas = [json.loads(line) for line in f.readlines()]
    print(f"total prompts: {len(test_datas)}")

    print(f"metrics: CLIP Score, eval test_prompt: {prompts_path}")

    all_prompt_scores = [0.0] * (len(methods) + 1)
    total_prompts = 0

    model_type = 'realistic'

    for global_i, row in enumerate(test_datas[start_idx:], start=start_idx):
        text = row["prompt"]
        iid = str(row["iid"]).replace("/", "_")

        text_input = clip.tokenize([text], truncate=True).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text_input)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        prompt_scores_sum = [0.0] * (len(methods) + 1)
        valid_seed_count = 0

        for seed_num in range(num_seeds):
            ori_img = f"{images_original_path}/{global_i}_{seed_num}.png"
            # 如果 original 也是 iid 命名，改这里
            # ori_img = f"{images_original_path}/rerank_{iid}_{seed_num}.png"

            images_path = [ori_img]
            for method in methods:
                lora_img = f"{base_path}/{method}/{model_type}/rerank_{iid}_{seed_num}.png"
                images_path.append(lora_img)

            if not all(os.path.exists(p) for p in images_path):
                continue

            images = [preprocess(Image.open(p).convert("RGB")) for p in images_path]
            image_input = torch.stack(images).to(device)

            with torch.no_grad():
                image_features = model.encode_image(image_input)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            similarity = (image_features @ text_features.T).squeeze(1)
            clip_scores = (torch.clamp(similarity, min=0.0) * 100.0).cpu().tolist()

            for j in range(len(methods) + 1):
                prompt_scores_sum[j] += clip_scores[j]

            valid_seed_count += 1

        if valid_seed_count == 0:
            continue

        for j in range(len(methods) + 1):
            all_prompt_scores[j] += prompt_scores_sum[j] / valid_seed_count

        total_prompts += 1

    print(f"\neval: CLIP Score, num_seeds: {num_seeds}")
    if total_prompts == 0:
        print("No valid prompts found.")
        return

    print(f"scores_ori: {all_prompt_scores[0] / total_prompts:.4f}")
    for n, method in enumerate(methods):
        print(f"{method}: {all_prompt_scores[n + 1] / total_prompts:.4f}")




if __name__ == '__main__':
    # PickScore MPS ImageReward HPSv2.1
    # 
    # 'testdata_250_totalpool_stylus'
    model_name = "HPSv2.1"
    prompts_path = '/shark/zhiwen/LoRAHunter/sd_encoder/test_data/250_clipemb-7_all_res_combinations_reank.jsonl'
    methods = ['250_clipemb-7_all_res_combinations_reank','testdata_250_totalpool_stylus']
    images_original_path = "/shark/zhiwen/LoRAHunter/sd_encoder/test_data/retrieval_testdata_500.jsonl"
    # score_quality(model_name=model_name, prompts_path=prompts_path, methods=methods[:], images_original_path=images_original_path, seed_num=4, device='cuda:0')
    # eval_clip_score(prompts_path=prompts_path, methods=methods[:], images_original_path=images_original_path, seed_num=3, device='cuda:1')

    methods = ['stylus_testdata_500_totalpool_stylus',"hunter_data_500_qwenemb_4-28_reank_beam","hunter_data_500_qwenemb_5-199_2-0_diverse"]
    prompts_path = "/shark/zhiwen/LoRAHunter/sd_encoder/test_data/retrieval_testdata_500.jsonl"
    images_original_path = "/shark/zhiwen/LoRAHunter/sd_encoder/outputs2/sdv15_retrieval_testdata_500/realistic"
    eval_clip_score_multi_seed_per_prompt(prompts_path=prompts_path, images_original_path=images_original_path, methods=methods[-1:], device='cuda:1', start_idx=250, num_seeds=10)