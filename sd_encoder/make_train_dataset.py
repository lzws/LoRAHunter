import os, json
from tqdm import tqdm

emb_path = '/shark/zhiwen/LoRAHunter/Diffimage-SD-emb'

metadata_path = '../SD_adapter_metadata/train_lora_10k.jsonl'
with open(metadata_path, 'r') as f:
    datas = [json.loads(line) for line in f.readlines()]
print(f'load {len(datas)} datas')

metadata_path2 = '../SD_adapter_metadata/train_lora_10k_2.jsonl'
with open(metadata_path2, 'r') as f:
    datas2 = [json.loads(line) for line in f.readlines()]

datas = datas + datas2

train_dataset = []

for data in tqdm(datas):
    model_id = data['adapter_id']
    model_id = str(model_id)
    val_path = f'{emb_path}/{model_id[:2]}/{model_id[2:4]}/{model_id.replace("/", "__")}_diffvec.pth'
    model_file = ''
    if os.path.exists(val_path): 
        train_dataset.append(data)

with open('train_sd_lora_dataset_20k.jsonl', 'w', encoding='utf-8') as f:
    for item in train_dataset:
        # 关键步骤：对每个字典单独使用 json.dumps，然后写入一行
        # ensure_ascii=False 保证中文正常显示，而不是变成 \uXXXX
        json_str = json.dumps(item, ensure_ascii=False)
        f.write(json_str + '\n')