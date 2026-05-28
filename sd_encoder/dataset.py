import torch
# from DiffimageEmb import EmbSaver
from torch.utils.data import DataLoader
import json, os
# from diffsynth.core import load_state_dict
# from diffsynth.utils.lora import GeneralLoRALoader
# from diffsynth.core.loader import hash_model_file,convert_keys_dict_to_single_str,load_keys_dict
from diffusers.loaders import StableDiffusionLoraLoaderMixin
import torch.distributed as dist


class EmbSaver():
    def __init__(self,emb_path='/shark/zhiwen/LoRAHunter/Diffimage-SD-emb', root_image_path='/shark/zhiwen/LoRAHunter/DiffImage_SD', txt_emb_path='/shark/zhiwen/LoRAHunter/train_set_txtemb_10k',prompt_emb_path="/shark/zhiwen/LoRAHunter/train_set_prompt_emb_20k") -> None:
        self.emb_path = emb_path
        self.root_image_path = root_image_path
        self.txt_emb_path = txt_emb_path
        self.prompt_emb_path = prompt_emb_path
    
    def get_dir(self, model_id):
        dir_1 = str(model_id)[:2]
        dir_2 = str(model_id)[2:4]
        return f'{self.emb_path}/{dir_1}/{dir_2}'
    
    def get_emb_path(self, model_id):
        return f'{self.get_dir(model_id)}/{str(model_id).replace("/", "__")}.pth'
    def get_vec_path(self, model_id):
        return f'{self.get_dir(model_id)}/{str(model_id).replace("/", "__")}_diffvec.pth'
    def get_txtemb_path(self, model_id):
        dir_1 = str(model_id)[:2]
        dir_2 = str(model_id)[2:4]
        return f'{self.txt_emb_path}/{dir_1}/{dir_2}/{str(model_id).replace("/", "__")}.pth'

    def get_prompt_emb_path(self, model_id, pid):
        dir_1 = str(model_id)[:2]
        dir_2 = str(model_id)[2:4]
        return f'{self.prompt_emb_path}/{dir_1}/{dir_2}/{str(model_id)}/{pid}.pth'

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
        torch.save(emb_dict,save_path)
        print(f"save emb dict to {save_path}")
    
    def save_vec(self, vec, model_id, save_path=None):
        if save_path is None:
            save_path = self.get_vec_path(model_id)
        vec = vec.cpu()
        torch.save(vec,save_path)
        print(f"save diff vec to {save_path}")
    
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
    
    def load_txtemb(self, model_id, device='cpu',file_path=None):
        if file_path is None:
            file_path = self.get_txtemb_path(model_id)
        vec = torch.load(file_path, map_location=device, weights_only=True)
        # print(f"load diff vec from {file_path}")
        return vec
    

class LoRADataset1(torch.utils.data.Dataset):
    def __init__(self, base_path='/shark/zhiwen/LoRAHunter/sd_lora/sd_lora_1', metadata_path='/shark/zhiwen/LoRAHunter/sd_encoder/train_sd_lora_dataset.jsonl', steps_per_epoch=1000, loras_per_item=1):
        self.base_path = base_path
        with open(metadata_path, 'r') as f:
            self.datas=[json.loads(line) for line in f.readlines()]

    def get_data(self, data_id):

        # prompts = self.datas[data_id]['prompts']
        # 只取前4条，留下1条测试
        # prompt_id = torch.randint(0, len(prompts)-1, (1,))[0]
        # prompt = prompts[prompt_id]

        model_file = self.datas[data_id]['model_file']
        model_id = self.datas[data_id]['model_id']
        diff_vec = EmbSaver().load_vec(model_id)
        # diff_vec = torch.randn(1, 768)

        text = self.datas[data_id]['llm_description']
        

        data = {
            "model_file": f'{self.base_path}/{model_file}', 
            "diff_vec": diff_vec, # [B,1,768]
            "lora_text": text, # [p1, p2 ...]
        }
        return data


    def __getitem__(self, index):
        return self.get_data(index)

    def __len__(self):
        return len(self.datas)


def default_lora_patterns():
    lora_patterns = []


    down_block_dict = {
        "attentions.0.proj_in.lora": [(320, 320),(640, 640), (1280, 1280)], "attentions.0.proj_out.lora": [(320, 320),(640, 640), (1280, 1280)],
        "attentions.0.transformer_blocks.0.attn1.processor.to_k_lora":[(320, 320),(640, 640),(1280, 1280)], "attentions.0.transformer_blocks.0.attn1.processor.to_v_lora":[(320, 320),(640, 640), (1280, 1280)],
        "attentions.0.transformer_blocks.0.attn1.processor.to_q_lora":[(320, 320),(640, 640),(1280, 1280)], "attentions.0.transformer_blocks.0.attn1.processor.to_out_lora":[(320, 320),(640, 640), (1280, 1280)],
        "attentions.0.transformer_blocks.0.attn2.processor.to_k_lora":[(768, 320),(768, 640),(768, 1280)], "attentions.0.transformer_blocks.0.attn2.processor.to_v_lora":[(768, 320),(768, 640),(768, 1280)],
        "attentions.0.transformer_blocks.0.attn2.processor.to_q_lora":[(320, 320),(640, 640),(1280, 1280)], "attentions.0.transformer_blocks.0.attn2.processor.to_out_lora":[(320, 320),(640, 640), (1280, 1280)],
        "attentions.0.transformer_blocks.0.ff.net.0.proj.lora":[(320, 2560),(640, 5120),(1280,10240)], "attentions.0.transformer_blocks.0.ff.net.2.lora":[(1280, 320),(2560, 640),(5120,1280)],
        "attentions.1.proj_in.lora": [(320, 320),(640, 640), (1280, 1280)], "attentions.1.proj_in.lora": [(320, 320),(640, 640), (1280, 1280)],
        "attentions.1.transformer_blocks.0.attn1.processor.to_k_lora":[(320, 320),(640, 640),(1280, 1280)], "attentions.1.transformer_blocks.0.attn1.processor.to_v_lora":[(320, 320),(640, 640),(1280, 1280)],
        "attentions.1.transformer_blocks.0.attn1.processor.to_q_lora":[(320, 320),(640, 640),(1280, 1280)], "attentions.1.transformer_blocks.0.attn1.processor.to_out_lora":[(320, 320),(640, 640),(1280, 1280)],
        "attentions.1.transformer_blocks.0.attn2.processor.to_k_lora":[(768, 320),(768, 640),(768, 1280)], "attentions.1.transformer_blocks.0.attn2.processor.to_v_lora":[(768, 320),(768, 640),(768, 1280)],
        "attentions.1.transformer_blocks.0.attn2.processor.to_q_lora":[(320, 320),(640, 640),(1280, 1280)], "attentions.1.transformer_blocks.0.attn2.processor.to_out_lora":[(320, 320),(640, 640),(1280, 1280)],
        "attentions.1.transformer_blocks.0.ff.net.0.proj.lora":[(320, 2560),(640, 5120),(1280,10240)], "attentions.1.transformer_blocks.0.ff.net.2.lora":[(1280, 320),(2560, 640),(5120,1280)],
    }
    for i in range(3):
        for suffix in down_block_dict:
            lora_patterns.append({
                "name": f"unet.down_blocks.{i}.{suffix}",
                "dim": down_block_dict[suffix][i],
                "type": suffix+'_'+str(i),
            })
    
    mid_block_dict = {
        "attentions.0.proj_in.lora": [(320, 320),(640, 640), (1280, 1280)], "attentions.0.proj_out.lora": [(320, 320),(640, 640), (1280, 1280)],
        "attentions.0.transformer_blocks.0.attn1.processor.to_k_lora":[(320, 320),(640, 640),(1280, 1280)], "attentions.0.transformer_blocks.0.attn1.processor.to_v_lora":[(320, 320),(640, 640), (1280, 1280)],
        "attentions.0.transformer_blocks.0.attn1.processor.to_q_lora":[(320, 320),(640, 640),(1280, 1280)], "attentions.0.transformer_blocks.0.attn1.processor.to_out_lora":[(320, 320),(640, 640), (1280, 1280)],
        "attentions.0.transformer_blocks.0.attn2.processor.to_k_lora":[(768, 320),(768, 640),(768, 1280)], "attentions.0.transformer_blocks.0.attn2.processor.to_v_lora":[(768, 320),(768, 640),(768, 1280)],
        "attentions.0.transformer_blocks.0.attn2.processor.to_q_lora":[(320, 320),(640, 640),(1280, 1280)], "attentions.0.transformer_blocks.0.attn2.processor.to_out_lora":[(320, 320),(640, 640), (1280, 1280)],
        "attentions.0.transformer_blocks.0.ff.net.0.proj.lora":[(320, 2560),(640, 5120),(1280,10240)], "attentions.0.transformer_blocks.0.ff.net.2.lora":[(1280, 320),(2560, 640),(5120,1280)],
    }

    for i in range(1):
        for suffix in mid_block_dict:
            lora_patterns.append({
                "name": f"unet.mid_block.{suffix}",
                "dim": mid_block_dict[suffix][2],
                "type": suffix + '_2',
            })
    
    for i in range(3):
        dim_idx = 3 - 1 - i
        for suffix in down_block_dict:
            lora_patterns.append({
                "name": f"unet.up_blocks.{i+1}.{suffix}",
                "dim": down_block_dict[suffix][dim_idx],
                "type": suffix + '_' + str(dim_idx),
            })


    return lora_patterns


class LoRADataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path='/shark/zhiwen/LoRAHunter/sd_lora/lzwecnu',
        task='clipemb',
        metadata_path='/shark/zhiwen/LoRAHunter/sd_encoder/train_sd_lora_dataset.jsonl',
        emb_path='',
        txt_emb_path='',
        prompt_emb_path='',
    ):
        self.base_path = base_path
        self.task = task
        self.cache_path = '/shark/zhiwen/LoRAHunter/sd_lora/lora_cache_2'

        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.datas = [json.loads(line) for line in f.readlines()]

        self.emb_saver = EmbSaver(emb_path=emb_path, txt_emb_path=txt_emb_path,prompt_emb_path=prompt_emb_path)
        self.patterns = default_lora_patterns()

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        item = self.datas[index]
        # print(f"################### {item.keys()}")
        model_file = item['model_file']
        model_id = item['adapter_id']
        text = item['llm_description']


        # full_model_path = f'{self.base_path}/{model_file}'
        cache_model_path = f'{self.cache_path}/{model_file.replace(".safetensors", ".pt")}'
        # load LoRA in dataloader stage
        # lora = StableDiffusionLoraLoaderMixin.lora_state_dict(full_model_path)[0]
        try:
            lora = torch.load(cache_model_path, map_location="cpu", weights_only=True)
        except:
            print(f"load lora from {cache_model_path} failed, model_file:{model_file}")
            raise Exception("load lora from cache failed")
        # lora = torch.load(cache_model_path, map_location="cpu", weights_only=True)


        # needed_keys = set(p["name"] + ".down.weight" for p in self.patterns) | \
        #        set(p["name"] + ".up.weight" for p in self.patterns)
        # lora = {k: v for k, v in lora.items() if k in needed_keys}

        # precomputed image diff vector
        diff_vec = self.emb_saver.load_vec(model_id)
        if diff_vec.dim() == 2 and diff_vec.size(0) == 1:
            diff_vec = diff_vec.squeeze(0)

        # load qwenvl emb
        if self.task == 'qwenemb':
            txtemb = self.emb_saver.load_txtemb(model_id)
            if txtemb.dim() == 2 and txtemb.size(0) == 1:
                txtemb = txtemb.squeeze(0)
        elif self.task == 'promptemb':
            prompts = item['prompts']
            pid = torch.randint(0, len(prompts), (1,)).item()
            text = prompts[pid]
            txtemb = self.emb_saver.load_prompt_emb(model_id=model_id,pid=pid)
            if txtemb.dim() == 2 and txtemb.size(0) == 1:
                txtemb = txtemb.squeeze(0)
        else:
            txtemb = torch.zeros(768)
        
        return {
            "model_file": cache_model_path,
            "lora": lora,           # dict[str, tensor]
            "diff_vec": diff_vec,   # e.g. [1, 768]
            "lora_text": text,      # str
            "txtemb": txtemb,
        }


def lora_collate_fn(batch):
    return {
        "model_file": [x["model_file"] for x in batch],
        "lora": [x["lora"] for x in batch],  # keep as list[dict]
        "diff_vec": torch.stack([x["diff_vec"] for x in batch], dim=0),
        "lora_text": [x["lora_text"] for x in batch],
        "txtemb": torch.stack([x["txtemb"] for x in batch], dim=0),
    }


def make_metadata():
    source_file = '/shark/zhiwen/LoRAHunter/Available_LoRA_carlos_tags2_prompt.jsonl'
    with open(source_file, 'r') as f:
        datas = [json.loads(line) for line in f.readlines()]
    train_datas = []
    for data in datas:
        model_id = data['model_id']
        vec_path = f'/shark/zhiwen/LoRAHunter/DiffSynth-Studio/Diffimage-2-emb/{model_id.replace("/", "__")}_diffvec.pth'
        if os.path.exists(vec_path):
            train_datas.append(data)

    print(f'一共 {len(train_datas)} 训练数据')
    with open('train_available_lora_dataset.jsonl', 'w', encoding='utf-8') as f:
        for item in train_datas:
            # 关键步骤：对每个字典单独使用 json.dumps，然后写入一行
            # ensure_ascii=False 保证中文正常显示，而不是变成 \uXXXX
            json_str = json.dumps(item, ensure_ascii=False)
            f.write(json_str + '\n')


def check_lora():
    metadata_path = 'train_available_lora_dataset.jsonl'
    root_lora_path = "/shark/zhiwen/LoRAHunter/Qwen_LoRA"
    with open(metadata_path, 'r') as f:
        datas = [json.loads(line) for line in f.readlines()]
    num = 5
    start = 0 + num * 500
    end = start + 500
    lora_loader = GeneralLoRALoader()
    for i, data in enumerate(datas[:]):
        if i % 100 == 0:
            print(i)
        model_file = data['model_file']
        lora_path = os.path.join(root_lora_path, model_file)
        # lora = load_state_dict(lora_path, torch_dtype=torch.bfloat16, device='cpu')
        # lora = lora_loader.convert_state_dict(lora)
        model_hash = hash_model_file(lora_path, with_shape=False)
        if model_hash != 'fe6f3f7580702c8a4b959604265a603e':
            model_id = data['model_id']
            lora = load_state_dict(lora_path, torch_dtype=torch.bfloat16, device='cpu')
            lora = lora_loader.convert_state_dict(lora)
            keys = list(lora.keys())
            edata = {'model_id': model_id, 'model_file':data['model_file'], 'model_hash': model_hash,'key_format': keys[0], "len": len(keys)}
            with open(f'zchecklora/check_lora3_{num}.jsonl', 'a', encoding='utf-8') as f:
                json_str = json.dumps(edata, ensure_ascii=False)
                f.write(json_str + '\n')

def check_one_lora():
    model_id = "m40740/Qwen-Image-2512-Lightning-8steps-V1.0-bf16"
    metadata_path = 'train_available_lora_dataset.jsonl'
    model_file = 'm40740/Qwen-Image-2512-Lightning-8steps-V1.0-bf16/Qwen-Image-2512-Lightning-8steps-V1.0-bf16.safetensors'
    model_file = 'qwerddd678/RoleScene/角色置景_RoleScene Blend.safetensors'
    model_file = 'FreshIdeas/charcoalsketch2025/20.safetensors'
    model_file = 'ZhiluAI/Mecha/Mecha_20.safetensors'
    # with open(metadata_path, 'r') as f:
    #     datas = [json.loads(line) for line in f.readlines()]
    # data_map = {data['model_id']: data for data in datas}
    # data = data_map[model_id]
    lora_path = f"/shark/zhiwen/LoRAHunter/Qwen_LoRA/{model_file}"
    # lora = load_state_dict(lora_path, torch_dtype=torch.bfloat16, device='cuda:0')
    model_hash = hash_model_file(lora_path, with_shape=False)
    print(model_hash)
    # keys = list(lora.keys())
    # print(keys[0:5])
    # print('\n\n')
    # lora_loader = GeneralLoRALoader()
    # lora = lora_loader.convert_state_dict(lora)
    # keys = list(lora.keys())
    # print(keys[0:5])


if __name__ == '__main__':
    # check_one_lora()
    # check_lora()
    dataset = LoRADataset()
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True,num_workers=4,collate_fn=lora_collate_fn,pin_memory=True,persistent_workers=True)

    batch = next(iter(dataloader))
    # for k, v in batch.items():
    #     if k == 'diff_vec':
    #         print(k, v.shape)
    #     else:
    #         print(k, v)
    diff_vecs = batch['diff_vec']
    txt = batch['lora_text']
    print(diff_vecs.shape)
    print(diff_vecs[0].shape)

    