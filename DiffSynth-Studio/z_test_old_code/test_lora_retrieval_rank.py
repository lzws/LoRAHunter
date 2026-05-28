# from CARLoS_prompt import prompts_for_indexing
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
# from diffsynth.core import load_state_dict
# from diffsynth.core.loader import hash_model_file,convert_keys_dict_to_single_str,load_keys_dict
import torch
import os,json
from encoder import TextImageEncoder
import torch.nn.functional as F
from dashscope import Generation
import dashscope

root_lora_path = "/shark/zhiwen/LoRAHunter/Qwen_LoRA"

def generate_imgs(num):
    device = f"cuda:{num}"
    # load data
    test_data_path = '/shark/zhiwen/LoRAHunter/DiffSynth-Studio/retrieval_testdata_100_extract.jsonl'
    with open(test_data_path, 'r') as f:
        test_datas = [json.loads(line) for line in f.readlines()]

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
    # if len(loras) > 0:
    #     for lora in loras:
    #         pipe.load_lora(pipe.dit, lora)
    start = 0 + num*12
    end = start + 12
    for i,data in enumerate(test_datas[start:end]):
        prompt = data['prompt']
        seed = data['seed']
        image = pipe(prompt, seed=seed,num_inference_steps=30)
        image.save(f"outputs/retrieval_testdata_100/original/qwen_{i+start}.png")


# lora_list = [x["lora"] for x in sorted(data, key=lambda x: int(x["score"]), reverse=True)]
# print(lora_list)

def generate_imgs_lora(num):
    test_data_path = '/shark/zhiwen/LoRAHunter/DiffSynth-Studio/retrieval_testdata_100_rerank_totalpool.jsonl'
    with open(test_data_path, 'r') as f:
        test_datas = [json.loads(line) for line in f.readlines()]
    
    lora_metadata_path = '/shark/zhiwen/LoRAHunter/DiffSynth-Studio/train_available_lora_dataset.jsonl'
    with open(lora_metadata_path, 'r') as f:
        lora_metadatas = [json.loads(line) for line in f.readlines()]
    
    lora_metadata_path2 = '/shark/zhiwen/LoRAHunter/Available_LoRA_4k-13k.jsonl'
    with open(lora_metadata_path2, 'r') as f:
        lora_metadatas2 = [json.loads(line) for line in f.readlines()]
    lora_metadatas = lora_metadatas + lora_metadatas2
    # lora_metadatas_map = {one['model_file']: one for one in lora_metadatas}
    lora_metadatas_map = {one['model_id']: one for one in lora_metadatas}

    torch_dtype=torch.bfloat16,
    device = f"cuda:{num}"
    # if len(loras) > 0:
    #     for lora in loras:
    #         pipe.load_lora(pipe.dit, lora)
    # rerank_results
    # num = 0
    start = 0 + num*12
    end = start + 12
    pipe = None
    for i,data in enumerate(test_datas[start:end]):
        del pipe
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
        
        rerank_results = data['rerank_results']
        for keyword,loras in rerank_results.items():
            lora_list = [x["lora"] for x in sorted(loras, key=lambda x: int(x["score"]), reverse=True)]
            top1 = lora_list[0]
            alpha = max(1/len(rerank_results),0.5)
            for j in range(1):
                topk = lora_list[j]
                model_file = lora_metadatas_map[topk]['model_file']
                lora_path = f"{root_lora_path}/{model_file}"
                pipe.load_lora(pipe.dit, lora_path, alpha=alpha)

        prompt = data['prompt']
        seed = data['seed']
        image = pipe(prompt, seed=seed,num_inference_steps=30)
        image.save(f"outputs/retrieval_testdata_100/rerank_{i+start}.png")


def build_lora_index(lora_pool, device):
    model_ids = list(lora_pool.keys())
    lora_mat = []
    for mid in model_ids:
        v = lora_pool[mid]
        if v.dim() == 2 and v.shape[0] == 1:
            v = v.squeeze(0)
        lora_mat.append(v)
    lora_mat = torch.stack(lora_mat, dim=0).to(device)
    lora_mat = F.normalize(lora_mat, p=2, dim=1)
    return model_ids, lora_mat


def retrieve_topk(text_emb, model_ids, lora_mat, k=5):
    text_emb = text_emb.to(lora_mat.device)
    if text_emb.dim() == 1:
        text_emb = text_emb.unsqueeze(0)
    text_emb = F.normalize(text_emb, p=2, dim=1)

    sims = text_emb @ lora_mat.T
    topk_vals, topk_idxs = torch.topk(sims, k=k, dim=1)

    results = []
    for b in range(text_emb.shape[0]):
        one = []
        for idx, score in zip(topk_idxs[b].tolist(), topk_vals[b].tolist()):
            one.append({
                "model_id": model_ids[idx],
                "score": score
            })
        results.append(one)
    return results


def call_lora():

    # lora text encoder
    dtype, device = torch.bfloat16, 'cuda:1'
    clip_encoder = TextImageEncoder().to(dtype=dtype, device=device)

    test_data_path = '/shark/zhiwen/LoRAHunter/DiffSynth-Studio/retrieval_testdata_100_extract.jsonl'
    with open(test_data_path, 'r') as f:
        test_datas = [json.loads(line) for line in f.readlines()]

    
    
    # lora pool
    lora_pool_1 = torch.load('/shark/zhiwen/LoRAHunter/DiffSynth-Studio/train_available_lora_index_vec_ep29_1.pth',weights_only=True, map_location=device)
    lora_pool_2 = torch.load('/shark/zhiwen/LoRAHunter/DiffSynth-Studio/train_available_lora_index_vec_ep29_2.pth',weights_only=True, map_location=device)
    # lora_pool_3 = torch.load('/shark/zhiwen/LoRAHunter/DiffSynth-Studio/Available_LoRA_4k-13k_index_vec_2.pth',weights_only=True, map_location=device)
    lora_pool = lora_pool_1 | lora_pool_2
    res_datas = []
    for i,data in enumerate(test_datas):
        print(i)
        extract_concept = data['extract_concept']
        data['retrieval_results'] = {}
        for ec in extract_concept:
            keyword = ec['keyword']
            print(keyword)
            retrieval_description = ec['retrieval_description']
            
            model_ids, lora_mat = build_lora_index(lora_pool, device)
            # text_emb 和 lora_pool_1 里面的所有向量计算余弦相似度，选出top 5 个最相似的
            text_emb = clip_encoder.encoding_text(f"a {keyword} LoRA. "+retrieval_description)
            top5 = retrieve_topk(text_emb, model_ids, lora_mat, k=5)[0]
            data['retrieval_results'][keyword] = top5
            # res_datas.append(data)
        with open("retrieval_testdata_100_calllora_totalpool.jsonl", 'a') as f:
            f.write(json.dumps(data)+'\n')


def llm_call(lora_list,prompt,keyword):
    text = (
        f"以下是按相关性排序的适配器描述列表：\n\n"
        f"{lora_list}"
        f"\n\n"
        f"以及有一个用来生成图片的prompt ‘{prompt}’ 和关键词 ‘{keyword}’。确定与该关键词及其prompt上下文直接匹配的相关适配器，返回每个适配器对应的分数，分数从 0-10，分数越高越相关，只能是整数。\n"
        'Stick to the specified format: The output must follow the JSON format provided below within <output format> and </output format> tags. \n'
        '''<output format> 
        [ 
            {  "lora": ”[lora name]”, "score": ”[Correlation score]” }, {"lora": ”[lora name]”,"score": ”[Correlation score]”}, ... 
        ]  
        </output format>'''
        f"\n\n"
        f"请以上面指定的 JSON 格式提供您的回复，不要包含 <output format> 和“json” 标签。"
    )

    dashscope.base_http_api_url = "https://dashscope.aliyuncs.com/api/v1"
    # dashscope.base_http_api_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": text},
    ]
    # MultiModalConversation
    response = dashscope.MultiModalConversation.call(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key = "sk-xxx",
        # api_key="sk-1827f4c92fa849c59eb88fb122b0401b",
        api_key="sk-943d9e3de8394d85a639d4facc7b8bbf",
        model="qwen3.5-plus",
        messages=messages,
        result_format="message",
        # 开启深度思考
        enable_thinking=False,
    )

    if response.status_code == 200:
        # 打印思考过程
        print("=" * 20 + "思考过程" + "=" * 20)
        # print(response.output.choices[0].message.reasoning_content)
        
        # 打印回复
        print("=" * 20 + "完整回复" + "=" * 20)
        print(response.output.choices[0].message.content[0]['text'])
        
        return response.output.choices[0].message.content[0]['text']
    else:
        print(f"HTTP返回码：{response.status_code}")
        print(f"错误码：{response.code}")
        print(f"错误信息：{response.message}")
        return None

def llm_rerank():
    test_data_path = '/shark/zhiwen/LoRAHunter/DiffSynth-Studio/retrieval_testdata_100_calllora_totalpool.jsonl'
    with open(test_data_path, 'r') as f:
        test_datas = [json.loads(line) for line in f.readlines()]
    
    lora_metadata_path = '/shark/zhiwen/LoRAHunter/DiffSynth-Studio/train_available_lora_dataset.jsonl'
    with open(lora_metadata_path, 'r') as f:
        lora_metadatas = [json.loads(line) for line in f.readlines()]

    lora_metadata_path2 = '/shark/zhiwen/LoRAHunter/Available_LoRA_4k-13k.jsonl'
    with open(lora_metadata_path2, 'r') as f:
        lora_metadatas2 = [json.loads(line) for line in f.readlines()]
    lora_metadatas = lora_metadatas + lora_metadatas2
    lora_metadatas_map = {one['model_file']: one for one in lora_metadatas}



    for data in test_datas[6:]:
        prompt = data['prompt']
        retrieval_results = data['retrieval_results']
        data['rerank_results'] = {}
        for keyword, top5 in retrieval_results.items():
            print(keyword)
            lora_list_info=""
            for one in top5:
                model_id = one['model_id']
                lora_info = lora_metadatas_map[model_id]
                short_description = lora_info['llm_description']
                mid = lora_info['model_id']
                lora_list_info += f"# {mid}: {short_description}\n"
            res = llm_call(lora_list_info, prompt, keyword)
            res = json.loads(res)
            data['rerank_results'][keyword] = res
        with open('retrieval_testdata_100_rerank_totalpool.jsonl', 'a') as f:
            f.write(json.dumps(data)+'\n')




if __name__ == "__main__":
    generate_imgs(7)
    # call_lora()
    # llm_rerank()
    # generate_imgs_lora(7)

# nohup python test_lora_retrieval_rank.py > zlog/retrieval/generate_imgs-7.log 2>&1 &