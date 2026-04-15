import pandas as pd
from modelscope.hub.api import HubApi
from modelscope import snapshot_download
import os
import zipfile


def upload(file_path,repo_path):
    YOUR_ACCESS_TOKEN = 'ms-b99bca69-c5d6-48b9-aa6d-58cb49ed57ac'
    api = HubApi()
    api.login(YOUR_ACCESS_TOKEN)
    owner_name = 'lzwecnu'
    model_name = 'Qwen-LoRA-DiffVec'
    # filename = file_path.split('/')[-1]
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=repo_path,
        repo_id=f"{owner_name}/{model_name}",
        repo_type = 'model',
        commit_message='upload model',
    )

def download_model():
    api = HubApi()
    api.login('ms-b99bca69-c5d6-48b9-aa6d-58cb49ed57ac')
    #模型下载
    model_dir = snapshot_download('lzwecnu/Qwen-LoRA-DiffVec', local_dir = 'avail_lora')



def unzip_file(zip_path, extract_to):
    # 确保目标目录存在
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # 获取所有文件名（可选，用于打印日志）
            # print(f"文件包含: {zip_ref.namelist()}")
            
            # 解压所有文件
            zip_ref.extractall(extract_to)
            print(f"成功解压到: {extract_to}")
            
    except zipfile.BadZipFile:
        print("错误：文件不是有效的 ZIP 文件或已损坏。")
    except Exception as e:
        print(f"发生错误: {e}")

from pathlib import Path
import shutil

def copy_pth_files_modern(source_folder, destination_folder):
    source_path = Path(source_folder)
    dest_path = Path(destination_folder)

    # 1. 确保目标文件夹存在
    dest_path.mkdir(parents=True, exist_ok=True)

    # 2. 查找所有 .pth 文件
    # glob("*.pth") 只查找当前层；rglob("*.pth") 会递归查找所有子文件夹
    pth_files = list(source_path.glob("*.pth")) 
    
    if not pth_files:
        print(f"未在 {source_folder} 中找到 .pth 文件。")
        return

    # 3. 遍历复制
    for file_path in pth_files:
        target_file = dest_path / file_path.name
        try:
            shutil.copy2(file_path, target_file)
            print(f"✅ 已复制: {file_path.name}")
        except Exception as e:
            print(f"❌ 复制 {file_path.name} 失败: {e}")

    print(f"操作完成，共处理 {len(pth_files)} 个文件。")

# 使用示例
# copy_pth_files_modern("./models", "./models_backup")

# 使用示例
# unzip_file('data.zip', './output_folder')

if __name__ == "__main__":
    root_emb_path = "/shark/zhiwen/LoRAHunter/DiffSynth-Studio/Diffimage-2-emb"
    model_id = 'Qwen'
    file_path = f"{root_emb_path}/{model_id.replace('/', '__')}.pth"
    repo_path = f"diffimage-emb/{model_id.replace('/', '__')}.pth"
    # upload(file_path,repo_path)
    # download_model()
    file_list = ['Diffimage-2-emb-0-400-1000-1400.zip','Diffimage-2-emb-1800-2200-2500-3114.zip','Diffimage-2-emb-2200-2500.zip']
    for file_name in file_list:
        # unzip_file(f'avail_lora/{file_name}', './output_folder')
        copy_pth_files_modern(f"./output_folder/{file_name.replace('.zip','')}", "/shark/zhiwen/LoRAHunter/DiffSynth-Studio/Diffimage-2-emb")
    # unzip_file('data.zip', './output_folder')