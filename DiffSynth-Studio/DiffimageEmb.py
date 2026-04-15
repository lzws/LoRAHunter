import os,json

from encoder import TextImageEncoder
import torch
root_image_path = '/shark/zhiwen/LoRAHunter/DiffSynth-Studio/DiffImage-2'

class EmbSaver():
    def __init__(self) -> None:
        self.emb_path = '/shark/zhiwen/LoRAHunter/DiffSynth-Studio/Diffimage-2-emb'
        self.root_image_path = '/shark/zhiwen/LoRAHunter/DiffSynth-Studio/DiffImage-2'
    
    def save_emb_dict(self, emb_dict, model_id, save_path=None):
        if save_path is None:
            save_path = f"{self.emb_path}/{model_id.replace('/', '__')}.pth"
        torch.save(emb_dict,save_path)
        print(f"save emb dict to {save_path}")
    
    def save_vec(self, vec, model_id, save_path=None):
        if save_path is None:
            save_path = f"{self.emb_path}/{model_id.replace('/', '__')}_diffvec.pth"
        vec = vec.cpu()
        torch.save(vec,save_path)
        print(f"save diff vec to {save_path}")
    
    def load_emb_dict(self, model_id, device='cpu',file_path=None):
        if file_path is None:
            file_path = f"{self.emb_path}/{model_id.replace('/', '__')}.pth"
        emb_dict = torch.load(file_path, map_location=device, weights_only=True)
        # print(f"load emb dict from {file_path}")
        return emb_dict

    def load_vec(self, model_id, device='cpu',file_path=None):
        if file_path is None:
            file_path = f"{self.emb_path}/{model_id.replace('/', '__')}_diffvec.pth"
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

def get_diff_vec(model_id):
    # 计算差分向量
    lora_embs = EmbSaver().load_emb_dict(model_id)
    base_embs = EmbSaver().load_emb_dict('Qwen')

    mean_diff = torch.stack([lora_embs[k] - base_embs[k] for k in lora_embs.keys()], dim=0).mean(dim=0)
    EmbSaver().save_vec(mean_diff, model_id)
    # print(mean_diff.shape)



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



if __name__ == '__main__':
    # text_image_encoder = TextImageEncoder(device='cuda:4')
    # print('text_image_encoder loaded')
    lora_id = 'qiyuanai/Qwen-Image_Cyberpunk-Style_Style-Material-Master-Series'
    with open('/shark/zhiwen/LoRAHunter/train_mini_datas.jsonl','r') as f:
        datas = [json.loads(line) for line in f.readlines()]

    text_image_encoder = TextImageEncoder(device='cuda:4')
    print('text_image_encoder loaded')
    for data in datas:
        get_diff_vec(model_id=data['model_id'])
    # load_emb(model_id='Qwen')
    # test_diff_vec(lora_id)