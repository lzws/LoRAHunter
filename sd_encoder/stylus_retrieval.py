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

def build_lora_index_clip(
    lora_pool_metadata_file='/shark/zhiwen/LoRAHunter/SD_adapter_metadata/exist_file_adapters.jsonl',
    ref_index_path="lora_index.pt",
    save_path="lora_index_clip.pt",
    batch_size=64,
    device="cuda",
):
    ref_index = torch.load(ref_index_path, map_location="cpu")
    ref_model_files = ref_index["model_files"]

    print(f"[CLIP Index] loaded reference index: {ref_index_path}")
    print(f"[CLIP Index] reference model files: {len(ref_model_files)}")

    with open(lora_pool_metadata_file, "r", encoding="utf-8") as f:
        all_datas = [json.loads(line) for line in f]

    metadata_map = {}
    for data in all_datas:
        model_file = data.get("model_file", "")
        metadata_map[model_file] = data

    matched_model_files = []
    texts = []

    for model_file in ref_model_files:
        if model_file not in metadata_map:
            continue

        data = metadata_map[model_file]
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

        matched_model_files.append(model_file)
        texts.append(text)

    print(f"[CLIP Index] matched items: {len(matched_model_files)}")

    clip_encoder = TextImageEncoder().to(device=device)
    clip_encoder.eval()

    all_embs = []
    with torch.no_grad():
        for start in tqdm(range(0, len(texts), batch_size), desc="Building CLIP text index"):
            end = min(start + batch_size, len(texts))
            batch_texts = texts[start:end]

            emb = clip_encoder.encoding_text(batch_texts)   # [B, D]
            emb = F.normalize(emb, dim=-1)
            all_embs.append(emb.cpu())

    if len(all_embs) == 0:
        raise RuntimeError("No embeddings were built for CLIP index.")

    all_embs = torch.cat(all_embs, dim=0)

    torch.save({
        "model_files": matched_model_files,
        "embeddings": all_embs,
    }, save_path)

    print(f"[CLIP Index] saved to {save_path}")
    print(f"[CLIP Index] num items: {len(matched_model_files)}")
    print(f"[CLIP Index] embedding shape: {all_embs.shape}")



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
    index_path = "lora_index/lora_index_clip.pt"
    retriever = LoRARetriever(index_path=index_path, device="cuda:4")

    test_data_path = 'test_data/retrieval_testdata_250.jsonl'
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

            
            top5 = retriever.retrieve(query_text=keyword,top_k=75)
            data['retrieval_results'][keyword] = top5
            # res_datas.append(data)
        with open("retrieval_testdata_100_calllora_totalpool_clip.jsonl", 'a') as f:
            f.write(json.dumps(data)+'\n')


from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import gc
lora_base_path = "/shark/zhiwen/LoRAHunter/sd_lora/lzwecnu"

def generate_images(num=0):
    test_data_path = '/shark/zhiwen/LoRAHunter/baseline/stylus/stylus/composer/testdata_250_totalpool_stylus.jsonl'
    with open(test_data_path, 'r') as f:
        test_datas = [json.loads(line) for line in f.readlines()]

    lora_metadata_path = '/shark/zhiwen/LoRAHunter/SD_adapter_metadata/exist_file_adapters.jsonl'
    with open(lora_metadata_path, 'r') as f:
        lora_metadatas = [json.loads(line) for line in f.readlines()]

    lora_metadatas_map = {str(one['adapter_id']): one for one in lora_metadatas}

    torch_dtype=torch.bfloat16,

    # num = 0
    device = f"cuda:{num}"

    start = 0 + num*25
    end = start + 25

    start, end = 0, 175

    seed_num = num
    seeds = [42,6734,3252,23498,62991]
    save_path = f"outputs/{test_data_path.split('/')[-1].split('.')[0]}"
    os.makedirs(save_path, exist_ok=True)

    # load sd 1.5 pipe
    # pipe = StableDiffusionPipeline.from_pretrained(
    #     "/shark/zhiwen/LoRAHunter/DiffSynth-Studio/models/AI-ModelScope/stable-diffusion-v1-5", 
    #     torch_dtype=torch_dtype
    # )
    model_path = '/shark/zhiwen/LoRAHunter/DiffSynth-Studio/models/lzwecnu/SDv1-5-model/realisticVisionV60B1_v51VAE.safetensors'
    pipe = StableDiffusionPipeline.from_single_file(model_path, torch_dtype=torch_dtype, local_files_only=True)

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
        if len(rerank_results) == 0:
            prompt = data['prompt']
            seed = seeds[seed_num]
            image = pipe(
                        prompt, 
                        negative_prompt="low quality, bad quality, worst quality, blurry, out of focus, bad hands, missing fingers, extra limbs, deformed, distorted,", 
                        num_inference_steps=40,
                        num_images_per_prompt=1,
                        generator=torch.Generator(device=device).manual_seed(seed),
                        guidance_scale=7.5
                    ).images[0]
            image.save(f"outputs/retrieval_testdata_100_stylus/rerank_{i+start}.png")
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

        
        prompt = data['prompt']
        seed = seeds[seed_num]
        image = pipe(
                    prompt + " realistic, high quality", 
                    negative_prompt=NEGATIVE_PROMPT_STR, 
                    num_inference_steps=35,
                    num_images_per_prompt=1,
                    generator=torch.Generator(device=device).manual_seed(seed),
                    guidance_scale=7
                ).images[0]
        image.save(f"{save_path}/rerank_{i+start}_{seed_num}.png")

        pipe.unload_lora_weights()
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # build_lora_index_clip()
    # call_lora()
    generate_images(4)

# nohup python stylus_retrieval.py > zlog/generate_images_sty_4.log 2>&1 &