
#验证 ModelScope token
from modelscope.hub.api import HubApi
api = HubApi()
api.login('ms-b99bca69-c5d6-48b9-aa6d-58cb49ed57ac')


model_list = ['lzwecnu/civitai-lora','lzwecnu/civitai-lora-10k-20k','lzwecnu/civitai-lora-20k-30k',
                'lzwecnu/civitai-lora-30k-50k','lzwecnu/civitai-lora-40k-50k','lzwecnu/civitai-lora-2','lzwecnu/civitai-lora-56k-60k',
                'lzwecnu/civitai-lora-66k-70k','lzwecnu/civitai-lora-60k-70k','lzwecnu/civitai-lora-70k-76k']
model_list = ['lzwecnu/civitai-lora-16k-20k','lzwecnu/civitai-lora-26k-30k','lzwecnu/civitai-lora-46k-50k']
#模型下载
from modelscope import snapshot_download
# for model_id in model_list[:]:
#     print(f"downloading {model_id}")
#     model_dir = snapshot_download(model_id,local_dir=f"sd_lora/{model_id}")

# id_prefix = [27]
# mdoel_id = 'lzwecnu/civitai-lora'
# for i in id_prefix:
#     print(f"downloading {mdoel_id}/{i}")
#     model_dir = snapshot_download(f"{mdoel_id}",allow_patterns=f"{i}/*.safetensors",local_dir=f"sd_lora_2/{mdoel_id}")



# model_dir = snapshot_download('lzwecnu/SDv15-LoRA-DiffVec',local_dir='SDv15-LoRA-DiffVec')


import zipfile
import os

def unzip_fix_encoding(zip_src, dst_dir):
    # 1. 确保目标目录存在
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    with zipfile.ZipFile(zip_src, 'r') as zfile:
        print(f"正在解压: {zip_src} ...")
        
        for info in zfile.infolist():
            # 2. 核心代码：尝试修复中文编码
            # 先尝试用 cp437 编码（zipfile默认），再尝试 gbk
            try:
                # 很多中文 Windows 打包的文件实际上是 GBK 编码
                info.filename = info.filename.encode('cp437').decode('gbk')
            except:
                # 如果解码失败，保持原样或尝试 utf-8
                pass
            
            # 3. 解压单个文件
            zfile.extract(info, dst_dir)
            
    print("✅ 解压完成！")


import os
import shutil


# ----------------

def organize_pth_files(src, dst):
    # 如果目标根目录不存在则创建
    if not os.path.exists(dst):
        os.makedirs(dst)

    # 遍历源文件夹下的所有文件
    for filename in os.listdir(src):
        if filename.endswith(".pth"):
            # 提取文件名（不含后缀），例如 "10017"
            name_part = os.path.splitext(filename)[0]

            # 确保文件名长度足够进行切片 (至少4位)
            if len(name_part) >= 4:
                # 提取前两位和中间两位
                dir_1 = name_part[0:2]  # "10"
                dir_2 = name_part[2:4]  # "00"
                
                # 构建目标目录路径：target_root/10/00/
                target_path = os.path.join(dst, dir_1, dir_2)
                
                # 递归创建文件夹
                os.makedirs(target_path, exist_ok=True)
                
                # 执行移动操作
                old_file_path = os.path.join(src, filename)
                new_file_path = os.path.join(target_path, filename)
                
                shutil.move(old_file_path, new_file_path)
                print(f"✅ 已移动: {filename} -> {dir_1}/{dir_2}/")
            else:
                print(f"⚠️ 跳过: {filename} (文件名长度不足4位)")

def move_file():
    emb_dir = "/shark/zhiwen/LoRAHunter/Diffimage-SD-emb-qwen/Diffimage-SD-emb-qwen"
    tar_dir = "/shark/zhiwen/LoRAHunter/Diffimage-SD-emb-qwen"
    dir_1_list = os.listdir(emb_dir)
    for dir_1 in dir_1_list:
        dir_2_list = os.listdir(os.path.join(emb_dir, dir_1))
        for dir_2 in dir_2_list:
            emb_files = os.listdir(os.path.join(emb_dir, dir_1, dir_2))
            os.makedirs(os.path.join(tar_dir, dir_1, dir_2), exist_ok=True)
            for emb_file in emb_files:
                target_path = os.path.join(tar_dir, dir_1, dir_2, emb_file)
                
                shutil.move(os.path.join(emb_dir, dir_1, dir_2, emb_file), target_path)
    print("✅ 移动完成！")

if __name__ == "__main__":
    # --- 配置区域 ---
    source_folder = "/shark/zhiwen/LoRAHunter/Diffimage-SD-emb"  # 存放原始 .pth 文件的文件夹
    target_root = "/shark/zhiwen/LoRAHunter/Diffimage-SD-emb"      # 重新组织后的根目录
    # organize_pth_files(source_folder, target_root)
    # print("\n所有文件整理完毕！")

    # 使用示例
    unzip_fix_encoding("SDv15-LoRA-DiffVec/Diffimage-SD-9000-9400.zip", "./DiffImage_SD_1")
    # model_dir = snapshot_download('lzwecnu/SDv15-LoRA-DiffVec',local_dir='SDv15-LoRA-DiffVec')
    # move_file()




# modelscope download --model lzwecnu/civitai-lora lora_4062.safetensors --local_dir ./dir
# nohup python3 download_sd_lora.py > download_sd_lora_3.log 2>&1 &
# nohup git clone https://oauth2:ms-b99bca69-c5d6-48b9-aa6d-58cb49ed57ac@www.modelscope.cn/lzwecnu/civitai-lora-reload-5k-10k.git > clone-reload-0-5k.log 2>&1 &