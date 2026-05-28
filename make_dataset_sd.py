import sys
import os,json
import pandas as pd
from utils import load_adapter_metadata, check_file_size
from safetensors.torch import load_file
from CARLoS_prompt import prompts_for_indexing
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import random
import gc

adapter_metadata_path = "/shark/zhiwen/LoRAHunter/baseline/dataset/Stylusdocsv2/cache/sd_adapters.pkl"
root_lora_dir = "/shark/zhiwen/LoRAHunter/sd_lora/sd_lora_1"


def convert_pkl_to_csv():
    adapters = load_adapter_metadata(adapter_metadata_path)
    datas = []
    for i, adapter in enumerate(adapters):
        data = {
            "index":i,
            "adapter_id": adapter.adapter_id,
            "download_url": adapter.download_url,
            "alias": adapter.alias,
            "title": adapter.title,
            "tags": adapter.tags,
            "description": adapter.description,
            "llm_description": adapter.llm_description,
        }
        datas.append(data)
    

def get_file_path(adapter_id):
    model_dir_list = os.listdir(root_lora_dir)
    id_prefix = str(adapter_id)[:2]

    model_files = []
    for model_dir in model_dir_list:
        model_file = f"{root_lora_dir}/{model_dir}/{id_prefix}/{adapter_id}.safetensors"
        if os.path.exists(model_file):
            model_files.append(model_file)
    if len(model_files) > 0:
        for model_file in model_files:
            if not check_file_size(model_file):
                return model_file.replace(f"{root_lora_dir}/","")
        return model_files[0].replace(f"{root_lora_dir}/","")
    else:
        return None

def find_adapter_file():
    '''
    1 遍历metadata文件，在lora 目录找到权重文件
    2 分成 3 个文件存储：1.正常找到的文件，2.本地没有，下载失败 3.文件小于1MB的，下载的文件有问题，重新下载
    '''
    metadata_file = "SD_adapter_metadata/sd_adapter_metadata.csv"
    metadata = pd.read_csv(metadata_file)
    no_file_list = []
    bad_file_list = []
    exist_file_list = []
    for i, row in metadata.iterrows():
        adapter_id = row["adapter_id"]
        file_path = get_file_path(adapter_id)
        if file_path is None:
            no_file_list.append({'meta_index':row['index'],'adapter_id':adapter_id,'download_url':row['download_url']})
        else:
            if check_file_size(f"{root_lora_dir}/{file_path}"):
                bad_file_list.append({'meta_index':row['index'],'adapter_id':adapter_id,'download_url':row['download_url'],'model_file':file_path})
            else:
                data = row.to_dict()
                data['model_file'] = file_path
                exist_file_list.append(data)

    print(f"no_file_list: {len(no_file_list)}")
    print(f"bad_file_list: {len(bad_file_list)}")
    print(f"exist_file_list: {len(exist_file_list)}")
    with open("SD_adapter_metadata/sd_lora_1/no_file_adapters.jsonl","w",encoding="utf-8") as f:
        for item in no_file_list:
            f.write(json.dumps(item,ensure_ascii=False)+"\n")
    with open("SD_adapter_metadata/sd_lora_1/bad_file_adapters.jsonl","w",encoding="utf-8") as f:
        for item in bad_file_list:
            f.write(json.dumps(item,ensure_ascii=False)+"\n")
    with open("SD_adapter_metadata/sd_lora_1/exist_file_adapters.jsonl","w",encoding="utf-8") as f:
        for item in exist_file_list:
            f.write(json.dumps(item,ensure_ascii=False)+"\n")


# 对bad file lis中的文件重新下载


# 用SD加载lora，检查是否可用
# 生成图片 每个LoRA 280个prompt，每个prompt 5 张图片

def select_train_set():
    input_file = 'SD_adapter_metadata/exist_file_adapters.jsonl'
    output_file = 'SD_adapter_metadata/train_lora_10k.jsonl'
    count = 10000
    # 1. 获取文件总行数 (快速扫描)
    print(f"正在统计总行数: {input_file} ...")
    with open(input_file, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    
    if count >= total_lines:
        print(f"请求数量 ({count}) 大于或等于总行数 ({total_lines})，直接复制原文件。")
        with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
            f_out.write(f_in.read())
        return

    # 2. 随机生成不重复的行号索引
    # random.sample 保证不重复，如果你允许重复可以使用 random.choices
    selected_indices = set(random.sample(range(total_lines), count))
    
    print(f"正在从 {total_lines} 行中抽取 {count} 行...")
    
    # 3. 第二次扫描，只写入命中的行
    with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
        for i, line in enumerate(f_in):
            if i in selected_indices:
                f_out.write(line)
                # 每抽取 1000 行打印一次进度
                if len(selected_indices) % 1000 == 0: 
                    pass # 这里可以加进度条逻辑，简单起见省略

    print(f"✅ 完成！已保存至: {output_file}")

def generate_diff_image(start,end,device='cuda'):
    # load prompts
    prompts = prompts_for_indexing()
    categories = ['Portraits', 'Landscapes', 'Artistic_Styles', 'Conceptual_Arts', 'Animals', 'Fashion', 'Vehicles', 'Food', 'Cinematic', 'Logos']
    seeds = [42, 984753, 237462, 81724, 5418346, 6562832, 12844, 67544, 43653, 35284]
    save_path = "DiffImage_SD"

    # lora metadata
    metadata_path = "SD_adapter_metadata/train_lora_10k_2.jsonl"
    with open(metadata_path, 'r') as f:
        metadatas = [json.loads(line) for line in f]
    
    # load sd 1.5 pipe
    pipe = StableDiffusionPipeline.from_pretrained(
        "/shark/zhiwen/LoRAHunter/DiffSynth-Studio/models/AI-ModelScope/stable-diffusion-v1-5", 
        torch_dtype=torch.float16
    )
    pipe.safety_checker = None
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)

    print("##"*10)
    print(f" task： {start} - {end}, device: {device}")
    print("##"*10)
    bs = 5
    # generate image
    for i, data in enumerate(metadatas[start:end]):
        model_id = data['adapter_id']
        model_file = data['model_file']

        print(f"############## 正在处理 {model_id} ##############")
        try:
            pipe.load_lora_weights(f"{root_lora_dir}/{model_file}")

            for c_tag in categories:
                sub_categories = prompts[c_tag]
                for k,v in sub_categories.items():
                    for i in range(len(v)):
                        prompt = v[i]
                        img_save_path = f"{save_path}/{model_id}/{c_tag}/{k}/{i}"
                        os.makedirs(img_save_path, exist_ok=True)
                        
                        generators = [torch.Generator(device=device).manual_seed(seed) for seed in seeds[:bs]]
                        images = pipe(
                            prompt, 
                            negative_prompt="low quality, bad quality, worst quality, blurry, out of focus, bad hands, missing fingers, extra limbs, deformed, distorted,", 
                            num_inference_steps=30,
                            num_images_per_prompt=bs,
                            generator=generators,
                            guidance_scale=7.5
                        ).images
                        for seed, image in zip(seeds[:bs], images):
                            image.save(f"{img_save_path}/{seed}.jpg")
        except Exception as e:
            print("++"*10)
            print(f"model_id: {model_id}, Error: {e}")
            print("++"*10)
            with open('error_log.jsonl', 'a', encoding='utf-8') as f:
            # json.dumps 将字典转为 JSON 字符串
                f.write(json.dumps({'model_id':model_id,'msg':str(e)}, ensure_ascii=False) + '\n')

        pipe.unload_lora_weights()
        gc.collect()
        torch.cuda.empty_cache()


def generate_sd_img():
    device = 'cuda:4'
    # load prompts
    prompts = prompts_for_indexing()
    categories = ['Portraits', 'Landscapes', 'Artistic_Styles', 'Conceptual_Arts', 'Animals', 'Fashion', 'Vehicles', 'Food', 'Cinematic', 'Logos']
    seeds = [42, 984753, 237462, 81724, 5418346, 6562832, 12844, 67544, 43653, 35284]
    save_path = "DiffImage_SD"


    
    # load sd 1.5 pipe
    pipe = StableDiffusionPipeline.from_pretrained(
        "/shark/zhiwen/LoRAHunter/DiffSynth-Studio/models/AI-ModelScope/stable-diffusion-v1-5", 
        torch_dtype=torch.bfloat16
    )
    pipe.safety_checker = None
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)

    print("##"*10)
    print(f" task： {start} - {end}, device: {device}")
    print("##"*10)
    bs = 5
    model_id = 'SDv1-5-bf'

    print(f"############## 正在处理 {model_id} ##############")


    for c_tag in categories:
        sub_categories = prompts[c_tag]
        for k,v in sub_categories.items():
            for i in range(len(v)):
                prompt = v[i]
                img_save_path = f"{save_path}/{model_id}/{c_tag}/{k}/{i}"
                os.makedirs(img_save_path, exist_ok=True)
                
                generators = [torch.Generator(device=device).manual_seed(seed) for seed in seeds[:bs]]
                images = pipe(
                    prompt, 
                    negative_prompt="low quality, bad quality, worst quality, blurry, out of focus, bad hands, missing fingers, extra limbs, deformed, distorted,", 
                    num_inference_steps=30,
                    num_images_per_prompt=bs,
                    generator=generators,
                    guidance_scale=7.5
                ).images
                for seed, image in zip(seeds[:bs], images):
                    image.save(f"{img_save_path}/{seed}.jpg")




if __name__ == "__main__":
    metadata_file = "SD_adapter_metadata/sd_adapter_metadata.csv"

                    # 76M.   18M
    lora_files = ["/shark/zhiwen/LoRAHunter/sd_lora/lzwecnu/civitai-lora-66k-70k/28/280276.safetensors","/shark/zhiwen/LoRAHunter/sd_lora/lzwecnu/civitai-lora-66k-70k/28/289562.safetensors",]
    find_adapter_file()
    # select_train_set()
    # 8000-8800 dev 2
    # 8800 - 9200 dev3
    # 0-800 123
    # 800 - 1100 shared-3 
    # 1100 - 1300 A100-1
    # 1300 - 1500 A100-3
    # 1500 - 1900 A100-4
    # 1900 - 2300 A100-4-2
    # 2300 - 3100 h20
    # 3100- 3900 a100-8
    # 3900 - 6300 123
    # 6300 - 7900 a100-8
    # 7900 - 8000 shared-3
    # 9200 - 9400 shared-3
    # 9400 - 9800 dev 2
    # 9800 - 10000 dev 2

    # train_2
    # 0-2400 123
    # 2400 - 3300 shared-3
    # 3300 - 3600 A100-1
    # 3600 - 4200 A100-4
    # 4200 - 4500 A100-3
    # 4500 - 5100 A100-8
    # 5100 - 5500 dev3
    # 5500 - 5600 A100-4-2
    # 5600 - 6400 A100-8
    # 6400 - 8800 123
    num = 3
    start = 5100 + num * 100
    end = start + 100
    device = f"cuda:{num}"
    # generate_diff_image(start,end,device)
    # generate_sd_img()


# nohup python make_dataset.py > zlog/trainset_2_diffimg_5100-5500-3.log 2>&1 &
