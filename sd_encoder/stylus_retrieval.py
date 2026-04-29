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



def build_lora_index_clip(
    lora_pool_metadata_file,
    save_path="lora_index_clip_1.pt",
    batch_size=64,
    device="cuda:4",
):
    # load metadata
    with open(lora_pool_metadata_file, "r", encoding="utf-8") as f:
        datas = [json.loads(line) for line in f]

    print(f"[CLIP Index] loaded {len(datas)} items from {lora_pool_metadata_file}")

    # load text encoder
    clip_encoder = TextImageEncoder().to(device=device)
    clip_encoder.eval()

    model_files = []
    texts = []

    # build text for each LoRA
    for data in datas:
        model_file = data.get("model_file", "")
        title = data.get("title", "")
        description = data.get("llm_description", "")
        tags = data.get("tags", [])

        if isinstance(tags, list):
            tag_str = ", ".join(tags)
        else:
            tag_str = str(tags)

        text = (
            f"Convert Stable Diffusion finetuned adapter description into an embedding for search: "
            f"Title: {title}; Description: {description}; Tags: {tag_str};"
        )

        model_files.append(model_file)
        texts.append(text)

    all_embs = []

    # batch encode
    with torch.no_grad():
        for start in tqdm(range(0, len(texts), batch_size), desc="Building CLIP text index"):
            end = min(start + batch_size, len(texts))
            batch_texts = texts[start:end]

            emb = clip_encoder.encoding_text(batch_texts)   # [B, D]
            emb = F.normalize(emb, dim=-1)
            all_embs.append(emb.cpu())

    all_embs = torch.cat(all_embs, dim=0)   # [N, D]

    # save
    torch.save({
        "model_files": model_files,
        "embeddings": all_embs,
    }, save_path)

    print(f"[CLIP Index] saved to {save_path}")
    print(f"[CLIP Index] num items: {len(model_files)}")
    print(f"[CLIP Index] embedding shape: {all_embs.shape}")




NEGATIVE_PROMPTS = [
    "realisticvision-negative-embedding",
    "ng_deepnegative_v1_75t",
    "bad anatomy",
    "bad proportions",
    "blurry",
    "cloned face",
    "cropped",
    "deformed",
    "dehydrated",
    "disfigured",
    "duplicate",
    "error",
    "extra arms",
    "extra fingers",
    "extra legs",
    "extra limbs",
    "fused fingers",
    "gross proportions",
    "jpeg artifacts",
    "long neck",
    "(low quality: 2)",
    "lowres",
    "malformed limbs",
    "missing arms",
    "missing legs",
    "morbid",
    "mutated hands",
    "mutation",
    "mutilated",
    "out of frame",
    "poorly drawn face",
    "poorly drawn hands",
    "signature",
    "text",
    "too many fingers",
    "ugly",
    "username",
    "watermark",
    "(worst quality:2)",
]
NEGATIVE_PROMPT_STR = ", ".join(NEGATIVE_PROMPTS)



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





def call_lora():
    index_path = "lora_index/lora_index_clip_1.pt"
    retriever = LoRARetriever(index_path=index_path, device="cuda:4")

    test_data_path = 'test_data/retrieval_testdata_500.jsonl'
    with open(test_data_path, 'r') as f:
        test_datas = [json.loads(line) for line in f.readlines()]

    for i,data in enumerate(test_datas):
        print(i)
        # extract_concept = data['extract_concept']
        data['retrieval_results'] = {}
        for ec in [1]:
            keyword = data['prompt']
            print(keyword)
            # retrieval_description = ec['retrieval_description']

            
            top5 = retriever.retrieve(query_text=keyword,top_k=150)
            data['retrieval_results'][keyword] = top5
            # res_datas.append(data)
        with open("test_data/retrieval_testdata_500_calllora_totalpool_clip.jsonl", 'a') as f:
            f.write(json.dumps(data)+'\n')


from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import gc
lora_base_path = "/shark/zhiwen/LoRAHunter/sd_lora/sd_lora_1"

def generate_images(num=0,start=0,end=250,model_type='original'):
    test_data_path = 'test_data/testdata_500_totalpool_stylus.jsonl'
    with open(test_data_path, 'r') as f:
        test_datas = [json.loads(line) for line in f.readlines()]

    lora_metadata_path = '/shark/zhiwen/LoRAHunter/SD_adapter_metadata/sd_lora_1/exist_file_adapters.jsonl'
    with open(lora_metadata_path, 'r') as f:
        lora_metadatas = [json.loads(line) for line in f.readlines()]

    lora_metadatas_map = {str(one['adapter_id']): one for one in lora_metadatas}

    torch_dtype=torch.bfloat16,

    # num = 0
    device = f"cuda:{num}"



    seed_num = num
    seeds = [42,6734,3252,23498,62991,4324,54894,12047592,163884,63485,927429,238451]
    seeds = seeds[:10]

    save_path = f"outputs2/stylus_{test_data_path.split('/')[-1].split('.')[0]}/{model_type}"
    os.makedirs(save_path, exist_ok=True)

    # load sd 1.5 pipe
    if model_type == 'original':

        pipe = StableDiffusionPipeline.from_pretrained(
            "/shark/zhiwen/LoRAHunter/DiffSynth-Studio/models/AI-ModelScope/stable-diffusion-v1-5", 
            torch_dtype=torch.bfloat16
        )
        prompt_bias = ''
    elif model_type == 'realistic':
        prompt_bias = ' realistic, high quality'
        model_path = '/shark/zhiwen/LoRAHunter/DiffSynth-Studio/models/lzwecnu/SDv1-5-model/realisticVisionV60B1_v51VAE.safetensors'
        print(f"load from {model_path}")
        pipe = StableDiffusionPipeline.from_single_file(model_path, torch_dtype=torch.bfloat16, local_files_only=True)
    else:
        raise ValueError

    pipe.safety_checker = None
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config,algorithm_type="dpmsolver++")
    pipe = pipe.to(device)

    for i,data in enumerate(test_datas[start:end]):

        if os.path.exists(f"{save_path}/rerank_{i+start}_{seed_num}.png"):
            continue

        rerank_results = data['rerank_results']
        adapter_names = []
        # alpha = max(1/len(rerank_results),0.5)
        alpha = 0.8



        prompt = data['prompt']
        prompts = [prompt + prompt_bias] * len(seeds)
        generators = [
            torch.Generator(device=device).manual_seed(seed)
            for seed in seeds
        ]

        if len(rerank_results) == 0:


            images = pipe(
                    prompts, 
                    negative_prompt=[NEGATIVE_PROMPT_STR] * len(seeds),
                    num_inference_steps=35,
                    num_images_per_prompt=1,
                    generator=generators,
                    guidance_scale=7
                ).images
            for seed_num, (seed, image) in enumerate(zip(seeds, images)):
                image.save(f"{save_path}/rerank_{i+start}_{seed_num}.png")
            # continue

        for keyword,loras in rerank_results.items():
            # lora_list = [x["lora"] for x in sorted(loras, key=lambda x: int(x["score"]), reverse=True)]
            if len(loras) == 0:
                continue
            model_ids = list(loras.keys())
            try:
                for j in range(1):
                    topk = model_ids[j]
                    model_file = lora_metadatas_map[topk]['model_file']
                    lora_path = f"{lora_base_path}/{model_file}"
                    # pipe.load_lora(pipe.dit, lora_path, alpha=alpha)
                    if str(topk) not in adapter_names:
                        pipe.load_lora_weights(lora_path, adapter_name=str(topk))
                        adapter_names.append(str(topk))
            except Exception as e:
                print(e)
        if len(adapter_names) > 0:
            alpha = max(1/len(adapter_names),0.45)
            pipe.set_adapters(adapter_names,adapter_weights=[alpha]*len(adapter_names))

        
        
        images = pipe(
                prompts, 
                negative_prompt=[NEGATIVE_PROMPT_STR] * len(seeds), 
                num_inference_steps=35,
                num_images_per_prompt=1,
                generator=generators,
                guidance_scale=7
            ).images
        for seed_num, (seed, image) in enumerate(zip(seeds, images)):
            image.save(f"{save_path}/rerank_{i+start}_{seed_num}.png")

        pipe.unload_lora_weights()
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # build_lora_index_clip(lora_pool_metadata_file="/shark/zhiwen/LoRAHunter/SD_adapter_metadata/sd_lora_1/exist_file_adapters.jsonl")
    # call_lora()
    # realistic original
    generate_images(num=3,start=250,end=500,model_type='realistic')

# nohup python stylus_retrieval.py > zlog/test_log/generate_stylus_realistic_1.log 2>&1 &