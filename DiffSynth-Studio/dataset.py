import torch
from DiffimageEmb import EmbSaver
from torch.utils.data import DataLoader
import json, os
from diffsynth.core import load_state_dict
from diffsynth.utils.lora import GeneralLoRALoader
from diffsynth.core.loader import hash_model_file,convert_keys_dict_to_single_str,load_keys_dict
# 数据 按LoRA划分，每个LoRA一条数据
# 每条数据包括 LoRA 权重、short_llm_description, diff_vec, prompt(随机5选1)
# 

class LoRADataset(torch.utils.data.Dataset):
    def __init__(self, base_path='/shark/zhiwen/LoRAHunter/Qwen_LoRA', metadata_path='/shark/zhiwen/LoRAHunter/DiffSynth-Studio/train_available_lora_dataset.jsonl', steps_per_epoch=1000, loras_per_item=1):
        self.base_path = base_path
        with open(metadata_path, 'r') as f:
            self.datas=[json.loads(line) for line in f.readlines()]

    def get_data(self, data_id):

        prompts = self.datas[data_id]['prompts']
        # 只取前4条，留下1条测试
        prompt_id = torch.randint(0, len(prompts)-1, (1,))[0]
        prompt = prompts[prompt_id]

        model_file = self.datas[data_id]['model_file']
        model_id = self.datas[data_id]['model_id']
        diff_vec = EmbSaver().load_vec(model_id)
        # diff_vec = torch.randn(1, 768)

        text = self.datas[data_id]['short_description']
        

        data = {
            "model_file": f'{self.base_path}/{model_file}', 
            "diff_vec": diff_vec, # [B,1,768]
            "lora_text": text, # [p1, p2 ...]
            "prompt": prompt

        }
        return data


    def __getitem__(self, index):
        return self.get_data(index)

    def __len__(self):
        return len(self.datas)


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
    check_lora()
    # dataset = LoRADataset()
    # dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # batch = next(iter(dataloader))
    # # for k, v in batch.items():
    # #     if k == 'diff_vec':
    # #         print(k, v.shape)
    # #     else:
    # #         print(k, v)
    # diff_vecs = batch['diff_vec']
    # print(diff_vecs.shape)
    # print(diff_vecs[0].shape)

    