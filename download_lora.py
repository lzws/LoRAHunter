
#模型下载
from modelscope import snapshot_download
import os, json
import logging
# model_dir = snapshot_download('axin9470/BXLCH')

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("log/download_lora8k-13k_errors_2.log"),
        logging.StreamHandler()
    ]
)

source_file = 'source_files/LoRA_QWEN_IMAGE_20_B_4k-13k.jsonl'

source_datas = []
with open(source_file, 'r') as f:
    for line in f:
        source_datas.append(json.loads(line))

print(f'共读取{len(source_datas)}条数据')
num = 3
start = 0 + num * 2500
end = start + 2500

for data in source_datas[start:end]:
    model_id = data['model_id']
    owner = data['owner']
    name = data['name']
    try:
        save_path = f'./Qwen_LoRA/{owner}/{name}'
        os.makedirs(save_path, exist_ok=True)
        model_dir = snapshot_download(f'{model_id}',local_dir = save_path)
        metadata_path = os.path.join(save_path, "metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        # 记录失败信息
        error_msg = f"❌ 下载失败: {model_id} | 错误: {str(e)}"
        print(error_msg)
        logging.error(error_msg)
        
# nohup python3 download_lora.py > log/download_lora8k-13k_3.log 2>&1 &