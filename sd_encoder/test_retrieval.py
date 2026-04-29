import os,json
import torch
from models import TextImageEncoder
from encoder import LoRAEncoder
from safetensors.torch import load_file
from diffusers.loaders import StableDiffusionLoraLoaderMixin

from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import torch.nn.functional as F
from dashscope import Generation
import dashscope

class LoRAPoolDataset(torch.utils.data.Dataset):
    def __init__(self, metadata_file, lora_base_path):
        with open(metadata_file, "r", encoding="utf-8") as f:
            self.datas = [json.loads(line) for line in f]
        self.lora_base_path = lora_base_path

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        data = self.datas[idx]
        model_file = data["model_file"]
        lora_path = f"{self.lora_base_path}/{model_file}"

        try:
            lora_dict = StableDiffusionLoraLoaderMixin.lora_state_dict(lora_path)[0]
            return {
                "model_file": model_file,
                "lora": lora_dict,
            }
        except Exception as e:
            print(f"[Dataset Warning] failed to load: {lora_path}, error: {e}")
            return None


def build_lora_index_no_tqdm(
    lora_pool_metadata_file,
    lora_base_path,
    encoder_path,
    save_path="lora_index.pt",
    batch_size=16,
    num_workers=4,
    device="cuda",
    dtype=torch.float,
):
    dataset = LoRAPoolDataset(lora_pool_metadata_file, lora_base_path)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lora_pool_collate_fn,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )

    lora_encoder = LoRAEncoder(L=1)
    lora_encoder.load_state_dict(load_file(encoder_path))
    lora_encoder = lora_encoder.to(device=device, dtype=dtype)
    lora_encoder.eval()

    all_paths = []
    all_embs = []

    with torch.no_grad():
        for batch in dataloader:
            model_files = batch["model_file"]
            loras = batch["lora"]

            batch_embs = []
            for lora in loras:
                try:
                    lora = {
                        k: v.to(device=device, dtype=dtype)
                        for k, v in lora.items()
                    }
                    emb = lora_encoder(lora)   # [1, D]
                    emb = torch.nn.functional.normalize(emb, dim=-1)
                    batch_embs.append(emb)
                except Exception as e:
                    print(f"encode failed: {e}")
                    batch_embs.append(None)

            valid_files = []
            valid_embs = []
            for mf, emb in zip(model_files, batch_embs):
                if emb is not None:
                    valid_files.append(mf)
                    valid_embs.append(emb.cpu())

            if len(valid_embs) > 0:
                valid_embs = torch.cat(valid_embs, dim=0)   # [B', D]
                all_paths.extend(valid_files)
                all_embs.append(valid_embs)

    all_embs = torch.cat(all_embs, dim=0)   # [N, D]

    torch.save({
        "model_files": all_paths,
        "embeddings": all_embs,
    }, save_path)

    print(f"saved index to {save_path}")
    print(f"num loras: {len(all_paths)}")
    print(f"embedding shape: {all_embs.shape}")







def build_lora_index(
    lora_pool_metadata_file,
    lora_base_path,
    encoder_path,
    save_path="lora_index_train.pt",
    batch_size=16,
    num_workers=4,
    device="cuda",
    dtype=torch.float,
):
    dataset = LoRAPoolDataset(lora_pool_metadata_file, lora_base_path)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lora_pool_collate_fn,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )

    print(f"[Index] dataset size: {len(dataset)}")
    print(f"[Index] batch_size: {batch_size}, num_workers: {num_workers}")
    print(f"[Index] loading encoder from: {encoder_path}")

    lora_encoder = LoRAEncoder(L=1,num_encoder_layers=8,num_probes=8)
    lora_encoder.load_state_dict(load_file(encoder_path))
    lora_encoder = lora_encoder.to(device=device, dtype=dtype)
    lora_encoder.eval()

    all_paths = []
    all_embs = []

    num_success = 0
    num_failed = 0
    start_time = time.time()

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Building LoRA Index", total=len(dataloader))

        for batch_id, batch in enumerate(pbar):
            if batch is None:
                print(f"[Index Warning] batch {batch_id} is empty, skipped.")
                continue
            batch_start = time.time()

            model_files = batch["model_file"]
            loras = batch["lora"]

            batch_embs = []
            for mf, lora in zip(model_files, loras):
                try:
                    lora = {
                        k: v.to(device=device, dtype=dtype)
                        for k, v in lora.items()
                    }
                    emb = lora_encoder(lora)   # [1, D]
                    emb = torch.nn.functional.normalize(emb, dim=-1)
                    batch_embs.append(emb)
                except Exception as e:
                    print(f"[Warning] encode failed: {mf}, error: {e}")
                    batch_embs.append(None)
                    num_failed += 1

            valid_files = []
            valid_embs = []
            for mf, emb in zip(model_files, batch_embs):
                if emb is not None:
                    valid_files.append(mf)
                    valid_embs.append(emb.cpu())
                    num_success += 1

            if len(valid_embs) > 0:
                valid_embs = torch.cat(valid_embs, dim=0)   # [B', D]
                all_paths.extend(valid_files)
                all_embs.append(valid_embs)

            batch_time = time.time() - batch_start
            elapsed = time.time() - start_time

            pbar.set_postfix({
                "success": num_success,
                "failed": num_failed,
                "batch_time": f"{batch_time:.2f}s",
                "elapsed": f"{elapsed/60:.1f}m",
            })

            if (batch_id + 1) % 50 == 0:
                print(
                    f"[Index] batch {batch_id+1}/{len(dataloader)} | "
                    f"success={num_success} failed={num_failed} "
                    f"elapsed={elapsed/60:.1f} min"
                )

    if len(all_embs) == 0:
        raise RuntimeError("No valid LoRA embeddings were built.")

    all_embs = torch.cat(all_embs, dim=0)   # [N, D]

    torch.save({
        "model_files": all_paths,
        "embeddings": all_embs,
    }, save_path)

    total_time = time.time() - start_time
    print(f"\n[Index] saved index to: {save_path}")
    print(f"[Index] num success: {num_success}")
    print(f"[Index] num failed: {num_failed}")
    print(f"[Index] embedding shape: {all_embs.shape}")
    print(f"[Index] total time: {total_time/60:.2f} min")
    print(f"[Index] avg speed: {num_success/max(total_time, 1e-6):.2f} lora/s")



def lora_pool_collate_fn(batch):
    batch = [x for x in batch if x is not None]

    if len(batch) == 0:
        return None

    return {
        "model_file": [x["model_file"] for x in batch],
        "lora": [x["lora"] for x in batch],
    }


class LoRARetriever:
    def __init__(self, index_path, device="cuda"):
        self.device = device
        self.clip_encoder = TextImageEncoder().to(device)
        self.clip_encoder.eval()

        index_data = torch.load(index_path, map_location="cpu")
        self.model_files = index_data["model_files"]
        self.lora_embs = index_data["embeddings"].float()   # [N, D]
        print(f"[Index] loaded index: {self.lora_embs.shape}")

        # normalize once
        self.lora_embs = F.normalize(self.lora_embs, dim=-1).to(device)

    @torch.no_grad()
    def retrieve(self, query_text, top_k=5):
        text_emb = self.clip_encoder.encoding_text([query_text])   # [1, D]
        text_emb = F.normalize(text_emb, dim=-1)

        scores = text_emb @ self.lora_embs.t()   # [1, N]
        top_scores, top_indices = torch.topk(scores, k=top_k, dim=-1)

        results = []
        for score, idx in zip(top_scores[0].tolist(), top_indices[0].tolist()):
            results.append({
                "model_file": self.model_files[idx],
                "score": score,
            })
        return results





def call_lora(index_path="lora_index_train.pt",test_data_path="retrieval_testdata_100.jsonl",output_path="test_data/call_lora_res.jsonl",device='cuda'):

    retriever = LoRARetriever(index_path=index_path, device=device)

    # test_data_path = '/shark/zhiwen/LoRAHunter/DiffSynth-Studio/z_retrieval_res/retrieval_testdata_100_extract.jsonl'
    with open(test_data_path, 'r') as f:
        test_datas = [json.loads(line) for line in f.readlines()]

    for i,data in enumerate(test_datas):
        print(i)
        extract_concept = data['extract_concept']
        data['retrieval_results'] = {}
        for ec in extract_concept:
            keyword = ec['keyword']
            print(keyword)
            retrieval_description = ec['retrieval_description']

            # text_emb 和 lora_pool_1 里面的所有向量计算余弦相似度，选出top 5 个最相似的
            query = f"a '{keyword}' adapter. "+retrieval_description


            top5 = retriever.retrieve(query_text=query,top_k=10)
            data['retrieval_results'][keyword] = top5
            # res_datas.append(data)
        with open(output_path, 'a') as f:
            f.write(json.dumps(data)+'\n')
    
def llm_call(lora_list,prompt,keyword):
    text = (
        f"以下是按相关性排序的适配器描述列表：\n\n"
        f"{lora_list}"
        f"\n\n"
        f"现在有一个用来生成图片的prompt ‘{prompt}’ 和关键词 ‘{keyword}’。确定与该关键词及其prompt上下文直接匹配的相关适配器,找到能够提高图像质量并更严格地遵循提示的 LoRA，返回每个适配器对应的分数，分数从 0-10，分数越高越相关，只能是整数。\n"
        'Stick to the specified format: The output must follow the JSON format provided below within <output format> and </output format> tags. \n'
        '''<output format> 
        [ 
            {  "lora": ”[lora id]”, "score": ”[Correlation score]” }, {"lora": ”[lora name]”,"score": ”[Correlation score]”}, ... 
        ]  
        </output format>'''
        f"\n\n"
        f"请以上面指定的 JSON 格式提供您的回复，不要包含 <output format> 和“json” 标签。"
    )

    dashscope.base_http_api_url = "https://dashscope.aliyuncs.com/api/v1"
    # dashscope.base_http_api_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    messages = [
        {"role": "system", "content": "You are a recommender system that recommends adapters from a database of millions of model adapters for popular base models such as Stable Diffusion, SDXL, and LLama."},
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
        print(f"查询的lora: {lora_list}")
        return None

def llm_rerank(test_data_path="",lora_metadata_path='/shark/zhiwen/LoRAHunter/SD_adapter_metadata/sd_lora_1/exist_file_adapters.jsonl'):

    black_list = ['255148','245088','130197','47837','128327','257194','234309']

    with open(test_data_path, 'r') as f:
        test_datas = [json.loads(line) for line in f.readlines()]
    
    # lora_metadata_path = '/shark/zhiwen/LoRAHunter/SD_adapter_metadata/exist_file_adapters.jsonl'
    with open(lora_metadata_path, 'r') as f:
        lora_metadatas = [json.loads(line) for line in f.readlines()]

    lora_metadatas_map = {one['model_file']: one for one in lora_metadatas}

    output_path = test_data_path.replace('.jsonl', '_llm-rerank.jsonl')

    for data in test_datas[:]:
        prompt = data['prompt']
        retrieval_results = data['retrieval_results']
        data['rerank_results'] = {}
        for keyword, top5 in retrieval_results.items():
            print(keyword)
            lora_list_info=""
            for one in top5[:20]:
                model_id = one['model_file']
                if model_id in black_list:
                    continue
                lora_info = lora_metadatas_map[model_id]
                short_description = lora_info['llm_description']
                mid = lora_info['adapter_id']
                # lora_list_info += f"# {mid}: {short_description}\n"
                lora_list_info += f"id: {mid} \n Title: {lora_info['title']} \n Tags:{lora_info['tags']} \n description: {short_description}\n\n"
            res = llm_call(lora_list_info, prompt, keyword)
            if res is None:
                continue
            res = json.loads(res)
            data['rerank_results'][keyword] = res
        with open(output_path, 'a') as f:
            f.write(json.dumps(data)+'\n')


if __name__ == "__main__":

    lora_pool_metadata_file = "/shark/zhiwen/LoRAHunter/SD_adapter_metadata/exist_file_adapters.jsonl"
    # lora_pool_metadata_file = "/shark/zhiwen/LoRAHunter/sd_encoder/train_sd_lora_dataset.jsonl"
    lora_base_path = "/shark/zhiwen/LoRAHunter/sd_lora/lzwecnu"
    encoder_path = "/shark/zhiwen/LoRAHunter/sd_encoder/models/lora_encoder/train_sd_lora_dataset/3/lora_encoder-99.safetensors"
    encoder_path = "/shark/zhiwen/LoRAHunter/sd_encoder/models/lora_encoder/clipemb/train_sd_lora_dataset_2/7/lora_encoder-104.safetensors"


    device='cuda:4'
    index_path = 'lora_index_dataset2_all.pt'
    test_data_path = '../test_data/retrieval_testdata_250_extract.jsonl'
    output_path = 'test_data/data_500_qwenemb_4-28.jsonl'
    # call_lora(index_path,test_data_path,output_path,device)
    llm_rerank(test_data_path=output_path)

    # retriever = LoRARetriever(index_path='lora_index.pt', device="cuda")
    # query = "Cyberpunk style"
    # top5 = retriever.retrieve(query_text=query)
    # print(top5)

# nohup python test_retrieval.py > zlog/build_lora_index_train.log 2>&1 &