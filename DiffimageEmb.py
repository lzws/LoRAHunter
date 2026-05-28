import os,json
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# from ..encoder import TextImageEncoder
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
root_image_path = 'DiffImage_SD'

from torch.utils.data import Dataset, DataLoader

from encoder import QwenVLEncoder
import torch.nn.functional as F
from tqdm import tqdm

# "openai/clip-vit-large-patch14"
class TextImageEncoder(torch.nn.Module):
    def __init__(self, model_name="DiffSynth-Studio/models/AI-ModelScope/clip-vit-large-patch14", dtype=torch.float,device="cuda"):
        super().__init__()
        self.device = device
        self.dtype = dtype
        # self.model, self.preprocess = clip.load('ViT-L/14@336px', download_root=model_name, device=device)
        print(f"load clip model from: {model_name}")
        self.model = CLIPModel.from_pretrained(model_name).to(dtype=torch.float)
        self.processor = CLIPProcessor.from_pretrained(model_name)



    @torch.no_grad()
    def encoding_image(self,image_path=''):
        image = Image.open(image_path)
        image_input = self.processor(images=image, return_tensors="pt").to(self.model.device)
        image_features = self.model.get_image_features(**image_input)
        return image_features.pooler_output #[1,768]

    @torch.no_grad()
    def encoding_images(self,image_path=[]):
        if isinstance(image_path,str):
            image_path = [image_path]
        images = []
        for img in image_path:
            images.append(Image.open(img).convert('RGB'))
        image_input = self.processor(images=images, return_tensors="pt").to(self.model.device)
        image_features = self.model.get_image_features(**image_input)
        return image_features.pooler_output #[N,768]
        

    @torch.no_grad()
    def encoding_text(self,text=''):
        text_input = self.processor(text=text, return_tensors="pt", max_length=77, truncation=True, padding=True).to(self.model.device)
        text_features = self.model.get_text_features(**text_input)
        return text_features.pooler_output #[1, 768]


class EmbSaver():
    def __init__(self,emb_path='Diffimage-SD-emb',root_image_path='Diffimage-SD',prompt_emb_path="",gcl_emb_path="") -> None:
        self.emb_path = emb_path
        self.root_image_path = root_image_path
        self.prompt_emb_path = prompt_emb_path
        self.gcl_emb_path = gcl_emb_path
    
    def get_dir(self, model_id):
        dir_1 = str(model_id)[:2]
        dir_2 = str(model_id)[2:4]
        return f'{self.emb_path}/{dir_1}/{dir_2}'
    
    def get_emb_path(self, model_id):
        return f'{self.get_dir(model_id)}/{str(model_id).replace("/", "__")}.pth'
    def get_vec_path(self, model_id):
        return f'{self.get_dir(model_id)}/{str(model_id).replace("/", "__")}_diffvec.pth'
    
    def get_prompt_emb_path(self, model_id, pid):
        dir_1 = str(model_id)[:2]
        dir_2 = str(model_id)[2:4]
        return f'{self.prompt_emb_path}/{dir_1}/{dir_2}/{str(model_id)}/{pid}.pth'

    def get_gcl_emb_path(self, iid):
        dir_1 = str(iid)[:2]
        return f'{self.gcl_emb_path}/{dir_1}/{iid}.pth'

    def save_gcl_emb(self, emb_dict, iid, save_path=None):
        if save_path is None:
            save_path = self.get_gcl_emb_path(iid)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(emb_dict,save_path)
        # print(f"save gcl emb to {save_path}")

    def load_gcl_emb(self, iid, device='cpu',file_path=None):
        if file_path is None:
            file_path = self.get_gcl_emb_path(iid)
        emb_dict = torch.load(file_path, map_location=device, weights_only=True)
        return emb_dict

    def save_prompt_emb(self, emb_dict, model_id, pid, save_path=None):
        if save_path is None:
            save_path = self.get_prompt_emb_path(model_id, pid)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(emb_dict,save_path)
        print(f"save prompt emb to {save_path}")
    
    def load_prompt_emb(self, model_id, pid, device='cpu',file_path=None):
        if file_path is None:
            file_path = self.get_prompt_emb_path(model_id, pid)
        emb_dict = torch.load(file_path, map_location=device, weights_only=True)
        return emb_dict

    def save_emb_dict(self, emb_dict, model_id, save_path=None):
        if save_path is None:
            save_path = self.get_emb_path(model_id)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(emb_dict,save_path)
        print(f"save emb dict to {save_path}")
    
    def save_vec(self, vec, model_id, save_path=None):
        if save_path is None:
            save_path = self.get_vec_path(model_id)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        vec = vec.cpu()
        torch.save(vec,save_path)
        # print(f"save diff vec to {save_path}")
    
    def load_emb_dict(self, model_id, device='cpu',file_path=None):
        if file_path is None:
            file_path = self.get_emb_path(model_id)
        emb_dict = torch.load(file_path, map_location=device, weights_only=True)
        # print(f"load emb dict from {file_path}")
        return emb_dict

    def load_vec(self, model_id, device='cpu',file_path=None):
        if file_path is None:
            file_path = self.get_vec_path(model_id)
        vec = torch.load(file_path, map_location=device, weights_only=True)
        # print(f"load diff vec from {file_path}")
        return vec


# 遍历每张图片，都变成emb存在一个pth文件里

def get_image_emb(model_id = 'Qwen',text_image_encoder=None):
    
    images_dir = f'{root_image_path}/{model_id}'
    emb_dict = {}
    category = os.listdir(images_dir)
    for c in category[:]:
        print(f'======= {c} =======')
        category_dir = f'{images_dir}/{c}'
        sub_category = os.listdir(category_dir)
        for s in sub_category[:]:
            sub_category_dir = f'{category_dir}/{s}'
            prompts = os.listdir(sub_category_dir)
            for p in prompts:
                images_path = f'{sub_category_dir}/{p}'
                seed_images = os.listdir(images_path)
                for seed_image in seed_images:
                    image_path = f'{images_path}/{seed_image}'
                    name = f'{c}.{s}.{p}.{seed_image.split(".")[0]}'
                    with torch.no_grad():
                        image_emb = text_image_encoder.encoding_image(image_path=image_path).cpu()
                    emb_dict[name] = image_emb
    EmbSaver().save_emb_dict(emb_dict, model_id)

def get_diff_vec(model_id, emb_saver, base_embs):
    lora_embs = emb_saver.load_emb_dict(model_id)

    mean_diff = torch.stack(
        [lora_embs[k] - base_embs[k] for k in lora_embs.keys()],
        dim=0
    ).mean(dim=0)

    emb_saver.save_vec(mean_diff, model_id)



def test_diff_vec(lora_id):
    lora_vec = EmbSaver().load_vec(lora_id,device='cuda:4')
    print(lora_vec.shape)
    text_image_encoder = TextImageEncoder(device='cuda:4')
    text = 'watercolor style'
    text = "cyberpunk robotic rabbit with metallic limbs and neon-lit organs, sleek futuristic design, vibrant color palette of blues and pinks, high-tech research facility interior, sharp focus on subject, cinematic sci-fi composition"
    text = "A young woman with long hair in an elegant gothic maid outfit sits poised before a towering sci-fi mecha with glowing blue eyes, set in a dark atmospheric hangar illuminated by cool cyan lights, detailed 2D illustration style."
    text = "A dynamic pose of a dancer wearing 唐襦裙_第二种唐风 with vibrant golden accents and warm orange hues, set against a backdrop of glowing temple eaves at night, attention to the texture and movement of the fabric, traditional hair with ornate pins, clear facial features, serene and elegant, modern interpretation of classic painting style --ar 2:3"
    with torch.no_grad():
        text_emb = text_image_encoder.encoding_text(text)
    print(text_emb.shape)

    cos_sim = torch.sum(lora_vec * text_emb, dim=1) / (torch.norm(lora_vec, dim=1) * torch.norm(text_emb, dim=1))
    print(cos_sim.item())



def load_emb(model_id = 'Qwen'):

    emb_dict = EmbSaver().load_emb_dict(model_id)
    # print(emb_dict.keys())
    for k,v in emb_dict.items():
        print(k,v.shape)


def coco_prompts():
    import pandas as pd
    coco_30k_file = "/shark/zhiwen/benchmark/EraseBenchmark/dataset/dataset/coco_30k.csv"
    df = pd.read_csv(coco_30k_file)
    # 随机抽取 1000 条
    sample_df = df.sample(n=1000, random_state=42)

    # 保存为新的 csv
    sample_df.to_csv("coco-1k.csv", index=False)


def get_image_embs_all(model_id='Qwen', text_image_encoder=None, batch_size=64):
    """
    通过先收集所有路径再统一 Batch 计算来加速
    """
    images_dir = f'{root_image_path}/{model_id}'
    emb_dict = {}
    
    # 用于收集所有图片的路径和对应的名称
    all_image_paths = []
    all_image_names = []
    
    print(f"正在扫描目录结构并收集图片路径...")
    
    # --- 第一阶段：纯循环收集路径和名称 (不进行推理) ---
    category = os.listdir(images_dir)
    for c in category[:]:
        category_dir = f'{images_dir}/{c}'
        sub_category = os.listdir(category_dir)
        
        for s in sub_category[:]:
            sub_category_dir = f'{category_dir}/{s}'
            prompts = os.listdir(sub_category_dir)
            
            for p in prompts:
                images_path = f'{sub_category_dir}/{p}'
                seed_images = os.listdir(images_path)
                
                # 这里只有 5 张图，非常快，直接全部加入大列表
                for seed_image in seed_images:
                    image_path = f'{images_path}/{seed_image}'
                    # 生成对应的 key
                    name = f'{c}.{s}.{p}.{seed_image.split(".")[0]}'
                    
                    all_image_paths.append(image_path)
                    all_image_names.append(name)

    print(f"收集完成，共 {len(all_image_paths)} 张图片。开始批量推理...")
    
    # --- 第二阶段：使用大 Batch 进行统一编码 ---
    total_count = len(all_image_paths)
    
    # 确保模型在评估模式
    text_image_encoder.model.eval()
    
    for i in range(0, total_count, batch_size):
        # 切片获取当前批次的路径和名称
        batch_paths = all_image_paths[i : i + batch_size]
        batch_names = all_image_names[i : i + batch_size]
        
        # 调用你的 encoding_images 方法
        # 此时 batch_paths 的长度是 batch_size (最后一批可能小于 batch_size)
        with torch.no_grad():
            batch_embeddings = text_image_encoder.encoding_images(batch_paths)
        
        # 将结果存入字典
        # 将 Tensor 移到 CPU
        batch_embeddings_cpu = batch_embeddings.cpu()
        
        for idx, name in enumerate(batch_names):
            emb_dict[name] = batch_embeddings_cpu[idx]
            
        # 简单的进度打印
        if (i + batch_size) % 1000 == 0 or i + batch_size >= total_count:
            print(f"进度: {min(i + batch_size, total_count)} / {total_count}")

    EmbSaver().save_emb_dict(emb_dict, model_id)


# 3. 定义 Dataset 类 (负责单张图片的读取和解码)
class ImageDataset(Dataset):
    def __init__(self, paths, names):
        self.paths = paths
        self.names = names

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # 这个函数会在多进程中被并行执行
        # 这里的异常处理防止某张坏图导致整个进程崩溃
        try:
            img = Image.open(self.paths[idx]).convert('RGB')
            return img, self.names[idx]
        except Exception as e:
            print(f"读取图片失败 {self.paths[idx]}: {e}")
            # 返回 None 占位，需要在 collate_fn 中处理
            return None, self.names[idx]



def get_image_embs(model_id='Qwen', text_image_encoder=None, emb_saver=None, task='clipemb', batch_size=64, num_workers=8):
    """
    高性能图片 Embedding 提取函数
    使用 DataLoader 多进程加载 + 批量推理加速
    """
    # 1. 基础路径设置
    root_image_path = "DiffImage_SD" # 请确保这个路径变量在你的环境中已定义
    images_dir = f'{root_image_path}/{model_id}'
    
    if not os.path.exists(images_dir):
        print(f"错误: 目录不存在 - {images_dir}")
        return

    emb_dict = {}
    device = text_image_encoder.device
    # text_image_encoder.model.eval()  # 切换到评估模式
    # emb_saver = EmbSaver(emb_path='Diffimage-SD-emb-qwen', root_image_path='Diffimage-SD')

    print(f"正在扫描目录结构并收集路径...")
    
    # 2. 第一阶段：快速扫描收集所有路径 (不涉及耗时的图片读取)
    all_image_paths = []
    all_image_names = []
    
    category = os.listdir(images_dir)
    for c in category[:]:
        category_dir = f'{images_dir}/{c}'
        if not os.path.isdir(category_dir): continue
            
        sub_category = os.listdir(category_dir)
        for s in sub_category[:]:
            sub_category_dir = f'{category_dir}/{s}'
            if not os.path.isdir(sub_category_dir): continue
                
            prompts = os.listdir(sub_category_dir)
            for p in prompts:
                images_path = f'{sub_category_dir}/{p}'
                if not os.path.isdir(images_path): continue
                    
                seed_images = os.listdir(images_path)
                for seed_image in seed_images:
                    # 简单的文件过滤，防止读取非图片文件
                    if not seed_image.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                        continue
                        
                    image_path = f'{images_path}/{seed_image}'
                    name = f'{c}.{s}.{p}.{seed_image.split(".")[0]}'
                    
                    all_image_paths.append(image_path)
                    all_image_names.append(name)

    print(f"收集完成，共 {len(all_image_paths)} 张图片。开始构建数据加载器...")



    # 4. 定义 collate_fn (负责将 PIL 图片转为 Tensor Batch)
    def collate_fn(batch):
        # 过滤掉读取失败的 None
        batch = [b for b in batch if b[0] is not None]
        if not batch:
            return None, None
            
        images, names = zip(*batch)
        
        # 使用你的 encoder 里的 processor 进行批量预处理
        # 这一步也在 CPU 进程中完成，利用 Batch 优势
        inputs = text_image_encoder.processor(images=list(images), return_tensors="pt", padding=True)
        # inputs = list(images)
        return inputs, names

    # 4. 定义 collate_fn (负责将 PIL 图片转为 Tensor Batch)
    def collate_fn_qwenvl(batch):
        # 过滤掉读取失败的 None
        batch = [b for b in batch if b[0] is not None]
        if not batch:
            return None, None
            
        images, names = zip(*batch)
        
        # 使用你的 encoder 里的 processor 进行批量预处理
        # 这一步也在 CPU 进程中完成，利用 Batch 优势

        images = list(images) #[PIL.Image]
        
        inputs = [{"image": img} for img in images]

        conversations = [text_image_encoder.model.format_model_input(
            text=ele.get('text'),
            image=ele.get('image'),
            video=ele.get('video'),
            instruction=ele.get('instruction'),
            fps=ele.get('fps'),
            max_frames=ele.get('max_frames')
        ) for ele in inputs]

        processed_inputs = text_image_encoder.model._preprocess_inputs(conversations)
        
        return processed_inputs, names

    collate_fn_mapping = {
        'clipemb': collate_fn,
        'qwenemb': collate_fn_qwenvl
    }

    # 5. 创建 DataLoader (核心加速引擎)
    dataset = ImageDataset(all_image_paths, all_image_names)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,  # 关键：开启多进程读取
        pin_memory=True,          # 关键：加速 CPU->GPU 传输
        prefetch_factor=2,        # 每个 worker 预先加载 2 个 batch
        drop_last=False,
        collate_fn=collate_fn_mapping['task']
    )

    print(f"开始批量推理 (Batch Size: {batch_size}, Workers: {num_workers})...")
    
    # 6. 推理循环
    total_steps = len(dataloader)
    
    for step, (inputs, names) in enumerate(dataloader):
        if inputs is None: continue

        
        with torch.no_grad():
            if task == 'clipemb':
                inputs = inputs.to(device, non_blocking=True)
                image_features = text_image_encoder.model.get_image_features(**inputs).pooler_output
            else:
                image_features = text_image_encoder.encoding_images(inputs)
            # 获取 pooler_output (即最终的 embedding 向量)
            embeddings = image_features.cpu() # 移回 CPU 以便存入字典 [N, 2048]

        # 存入结果字典
        for idx, name in enumerate(names):
            emb_dict[name] = embeddings[idx]
            
        # 进度打印
        if (step + 1) % 10 == 0 or (step + 1) == total_steps:
            print(f"进度: {min((step + 1) * batch_size, len(all_image_paths))} / {len(all_image_paths)}")

    print("推理完成，正在保存...")
    emb_saver.save_emb_dict(emb_dict, model_id)


def make_train_text_embs(metadapath="", batch_size=16):
    device = 'cuda'

    # 1. 加载数据
    with open(metadapath, 'r') as f:
        datas = [json.loads(line) for line in f.readlines()]
    print(f'load {len(datas)} datas')

    # 2. 加载模型
    encoder = QwenVLEncoder(device=device)
    emb_saver = EmbSaver(emb_path='train_gcl_promptemb', root_image_path='Diffimage-SD')

    # 3. 开始分批处理
    for i in range(0, len(datas), batch_size):
        # 切片获取当前批次
        batch_datas = datas[i : i + batch_size]
        
        batch_texts = []
        batch_ids = []

        # 构造当前批次的文本列表和 ID 列表
        for data in batch_datas:
            title = data['title']
            tags = data['tags']
            des = data['llm_description']
            # model_id = data['adapter_id']
            model

            query_text = (
                f"Convert Stable Diffusion finetuned adapter description into an embedding for search: "
                f"Title: {title}; Description: {des}; Tags: {tags};"
            )
            batch_texts.append(query_text)
            batch_ids.append(model_id)

        # 4. 批量推理 [batch_size, 2048]
        # 确保你的 encoding_text 方法支持传入 List[str]
        batch_embs = encoder.encoding_text(batch_texts) 

        # 5. 逐个保存结果
        for j, model_id in enumerate(batch_ids):
            # 取出对应的 embedding 行，并增加一个维度 [1, 2048] 以符合 save_emb_dict 的预期
            single_emb = batch_embs[j:j+1] 
            single_emb = single_emb.cpu().detach()
            emb_saver.save_emb_dict(single_emb, model_id)

        print(f"Processed batch {i//batch_size + 1}/{(len(datas)-1)//batch_size + 1}")





def main(num=0):
    # 8000 - 9200
    # 9400 - 10000
    # num = 
    task = 'qwenemb'
    device = f'cuda:{num}'
    start = 5100 + num * 100
    end = start + 100
    start, end = 5300, 5500
    print(f'start: {start}, end: {end}')

    emb_saver = EmbSaver(emb_path='Diffimage-SD-emb-qwen', root_image_path='Diffimage-SD')

    # 
    # text_image_encoder = TextImageEncoder(device=device).to(device=device)
    # print('text_image_encoder loaded')

    text_image_encoder = QwenVLEncoder(device=device)

    # base_embs = emb_saver.load_emb_dict('SDv1-5')

    # 1800 - 2200, 2500 - 3114
    metadata_path = 'SD_adapter_metadata/train_lora_10k_2.jsonl'
    with open(metadata_path, 'r') as f:
        datas = [json.loads(line) for line in f.readlines()]
    print(f'load {len(datas)} datas')

    n = 0
    for data in datas[start:end]:
        if n % 100 == 0:
            print(f'n: {n}')
        n+=1
        model_id = data['adapter_id']
        val_path = emb_saver.get_emb_path(model_id)
        vec_path = emb_saver.get_vec_path(model_id)

        if os.path.exists(val_path) or not os.path.exists(f'{root_image_path}/{model_id}'):
            print(f"{model_id} exists")
            continue
        get_image_embs(model_id = model_id, text_image_encoder=text_image_encoder, emb_saver=emb_saver, task=task)

        # if os.path.exists(val_path) and not os.path.exists(vec_path): 
        #     # print(f"############### encoding {model_id} ##################")
            
        #     get_diff_vec(model_id=model_id,emb_saver=emb_saver,base_embs=base_embs)

def make_diff_vec():

    emb_saver = EmbSaver(emb_path='Diffimage-SD-emb-qwen', root_image_path='Diffimage-SD')

    metadata_path = 'SD_adapter_metadata/train_lora_10k.jsonl'
    with open(metadata_path, 'r') as f:
        datas = [json.loads(line) for line in f.readlines()]
    print(f'load {len(datas)} datas')

    # load base only once
    base_embs = emb_saver.load_emb_dict('SDv1-5')
    
    todo_model_ids = []
    for data in datas:
        model_id = data['adapter_id']
        val_path = emb_saver.get_emb_path(model_id)
        vec_path = emb_saver.get_vec_path(model_id)

        if os.path.exists(val_path) and not os.path.exists(vec_path):
            todo_model_ids.append(model_id)

    print(f"need to process {len(todo_model_ids)} model ids")

    for n, model_id in enumerate(todo_model_ids):
        if n % 100 == 0:
            print(f"progress: {n}/{len(todo_model_ids)}")

        try:
            get_diff_vec(model_id=model_id, emb_saver=emb_saver, base_embs=base_embs)
        except Exception as e:
            print(f"failed: {model_id}, error: {e}")



from concurrent.futures import ProcessPoolExecutor, as_completed


_GLOBAL_EMB_SAVER = None
_GLOBAL_BASE_EMBS = None


def init_worker(emb_path, root_image_path):
    global _GLOBAL_EMB_SAVER, _GLOBAL_BASE_EMBS
    _GLOBAL_EMB_SAVER = EmbSaver(emb_path=emb_path, root_image_path=root_image_path)
    _GLOBAL_BASE_EMBS = _GLOBAL_EMB_SAVER.load_emb_dict('SDv1-5')


def process_one_model(model_id):
    global _GLOBAL_EMB_SAVER, _GLOBAL_BASE_EMBS
    try:
        lora_embs = _GLOBAL_EMB_SAVER.load_emb_dict(model_id)

        mean_diff = torch.stack(
            [lora_embs[k] - _GLOBAL_BASE_EMBS[k] for k in lora_embs.keys()],
            dim=0
        ).mean(dim=0)

        _GLOBAL_EMB_SAVER.save_vec(mean_diff, model_id)
        return model_id, True, None
    except Exception as e:
        return model_id, False, str(e)


def make_diff_vec_parallel(max_workers=8):

    emb_saver = EmbSaver(emb_path='Diffimage-SD-emb-qwen', root_image_path='Diffimage-SD')

    metadata_path = 'SD_adapter_metadata/train_lora_10k_2.jsonl'
    with open(metadata_path, 'r') as f:
        datas = [json.loads(line) for line in f.readlines()]
    print(f'load {len(datas)} datas')

    # load base only once
    base_embs = emb_saver.load_emb_dict('SDv1-5')

    todo_model_ids = []
    for data in datas:
        model_id = data['adapter_id']
        val_path = emb_saver.get_emb_path(model_id)
        vec_path = emb_saver.get_vec_path(model_id)

        if os.path.exists(val_path) and not os.path.exists(vec_path):
            todo_model_ids.append(model_id)

    print(f"need to process {len(todo_model_ids)} model ids with {max_workers} workers")

    success = 0
    failed = 0

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=init_worker,
        initargs=(emb_saver.emb_path, emb_saver.root_image_path),
    ) as executor:
        futures = [executor.submit(process_one_model, model_id) for model_id in todo_model_ids]

        for i, future in enumerate(as_completed(futures), 1):
            model_id, ok, err = future.result()

            if ok:
                success += 1
            else:
                failed += 1
                print(f"failed: {model_id}, error: {err}")

            if i % 100 == 0:
                print(f"progress: {i}/{len(todo_model_ids)}, success={success}, failed={failed}")

    print(f"done. success={success}, failed={failed}")



def make_carlos_difftxt_emb(device="cuda", save_path="carlos_difftxt_qwenemb.pt", task='qwenemb'):
    from CARLoS_prompt import prompts_for_indexing

    prompts = prompts_for_indexing()
    categories = [
        'Portraits', 'Landscapes', 'Artistic_Styles', 'Conceptual_Arts',
        'Animals', 'Fashion', 'Vehicles', 'Food', 'Cinematic', 'Logos'
    ]

    all_prompts = []
    for c_tag in categories:
        sub_categories = prompts[c_tag]
        for k, v in sub_categories.items():
            for prompt in v:
                all_prompts.append(prompt)

    # 去重
    all_prompts = list(dict.fromkeys(all_prompts))
    print(f"[Prompt] num prompts after dedup: {len(all_prompts)}")

    # encoder
    if task == "clipemb":
        encoder = TextImageEncoder(device=device).to(device)
        encoder.eval()
    else:
        encoder = QwenVLEncoder(device=device)
        encoder.eval()

    batch_size = 64
    all_embs = []

    for i in range(0, len(all_prompts), batch_size):
        batch_prompts = all_prompts[i:i+batch_size]
        embs = encoder.encoding_text(batch_prompts)   # [B, D]
        embs = F.normalize(embs, dim=-1)
        all_embs.append(embs.detach().cpu())

        print(f"[Prompt] encoded {i + len(batch_prompts)}/{len(all_prompts)}")

    prompt_embs = torch.cat(all_embs, dim=0)   # [M, D]
    prompt_emb_mean = F.normalize(prompt_embs.mean(dim=0, keepdim=True), dim=-1)   # [1, D]

    save_data = {
        "prompts": all_prompts,
        "prompt_embeddings": prompt_embs,       # [M, D]
        "prompt_embedding_mean": prompt_emb_mean,  # [1, D]
    }
    torch.save(save_data, save_path)
    print(f"[Done] saved prompt embeddings to {save_path}")



def make_train_prompts_embs(metadapath="",start=0,end=100,device='cuda'):
    # device = 'cuda'

    # 1. 加载数据
    with open(metadapath, 'r') as f:
        datas = [json.loads(line) for line in f.readlines()]
    print(f'load {len(datas)} datas, start: {start}, end: {end}')
    datas = datas[start:end]

    # 2. 加载模型
    encoder = QwenVLEncoder(device=device)
    emb_saver = EmbSaver(prompt_emb_path='train_set_prompt_emb_20k_2', root_image_path='Diffimage-SD')

    # 3. 开始分批处理
    for i in tqdm(range(0, len(datas))):
        # 切片获取当前批次
        # batch_datas = datas[i : i + batch_size]

        data = datas[i]
        prompts = data['prompts']

        model_id = data['adapter_id']

        batch_embs = encoder.encoding_text(prompts) 

        # 5. 逐个保存结果
        for j in range(0, len(prompts)):
            # 取出对应的 embedding 行，并增加一个维度 [1, 2048] 以符合 save_emb_dict 的预期
            single_emb = batch_embs[j:j+1]
            single_emb = single_emb.detach().cpu()
            emb_saver.save_prompt_emb(single_emb, model_id, j)


def make_train_gcl_embs(metadapath="", batch_size=16):
    device = 'cuda'

    # 1. 加载数据
    with open(metadapath, 'r') as f:
        datas = [json.loads(line) for line in f.readlines()]
    print(f'load {len(datas)} datas')

    # 2. 加载模型
    encoder = QwenVLEncoder(device=device)
    emb_saver = EmbSaver(gcl_emb_path='train_gcl_promptemb', root_image_path='Diffimage-SD')

    # 3. 开始分批处理
    for i in range(0, len(datas), batch_size):
        # 切片获取当前批次
        batch_datas = datas[i : i + batch_size]
        
        batch_texts = []
        batch_ids = []

        # 构造当前批次的文本列表和 ID 列表
        for data in batch_datas:
            prompt = str(data['prompt'])
            iid = data['iid']


            batch_texts.append(prompt)
            batch_ids.append(iid)

        # 4. 批量推理 [batch_size, 2048]
        # 确保你的 encoding_text 方法支持传入 List[str]
        batch_embs = encoder.encoding_text(batch_texts) 

        # 5. 逐个保存结果
        for j, model_id in enumerate(batch_ids):
            # 取出对应的 embedding 行，并增加一个维度 [1, 2048] 以符合 save_emb_dict 的预期
            single_emb = batch_embs[j:j+1] 
            single_emb = single_emb.cpu().detach()
            emb_saver.save_gcl_emb(single_emb, model_id)

        print(f"Processed batch {i//batch_size + 1}/{(len(datas)-1)//batch_size + 1}")




if __name__ == '__main__':
    # main(1)
    # make_train_text_embs(metadapath="SD_adapter_metadata/train_lora_10k.jsonl", batch_size=32)
    # make_diff_vec_parallel()
    # make_carlos_difftxt_emb()
    num = 7
    device = f"cuda:{num}"
    start = 0 + num * 2500
    end =  start + 2500
    # make_train_prompts_embs(metadapath="sd_encoder/train_sd_lora_dataset_20k_prompt2.jsonl", start=start, end=end, device=device)
    make_train_gcl_embs(metadapath="/shark/zhiwen/LoRAHunter/rank_encoder/train_dataset/diffusion_db2_candidates.jsonl", batch_size=32)
    
# nohup python DiffimageEmb.py > zlog/diffimage_prompt_emb_7.log 2>&1 &


