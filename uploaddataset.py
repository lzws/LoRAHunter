import os
import zipfile
import modelscope
from modelscope.hub.api import HubApi
def zip_folder_detailed(folder_path, output_zip_path):
    """
    使用 zipfile 压缩文件夹，支持递归遍历
    """
    with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # 遍历文件夹
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                # 计算相对路径，避免压缩包内包含绝对路径
                arcname = os.path.relpath(file_path, folder_path)
                
                # 【可选】在这里添加过滤逻辑，例如排除 .git 文件夹
                if '.git' in arcname or '__pycache__' in arcname:
                    continue 
                
                zipf.write(file_path, arcname)
    
    print(f"✅ 压缩完成: {output_zip_path}")



def upload_file(file_path,repo_path):
    YOUR_ACCESS_TOKEN = 'ms-f2fdbf7c-5f13-4f11-ab7c-45ba458c0077'
    api = HubApi()
    api.login(YOUR_ACCESS_TOKEN)
    owner_name = 'lzwecnu'
    model_name = 'LoRAHunter_Training_data_2'
    # filename = file_path.split('/')[-1]
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=repo_path,
        repo_id=f"{owner_name}/{model_name}",
        repo_type = 'model',
        commit_message='upload dataset',
    )


if __name__ == "__main__":
    # 使用示例
    save_path = "/shark/zhiwen/LoRAHunter/training_dataset"

    folder_path = "/shark/zhiwen/LoRAHunter/Diffimage-SD-emb-qwen-diffvec"
    output_zip_path = f"{save_path}/{folder_path.split('/')[-1]}.zip"
    # zip_folder_detailed(folder_path, output_zip_path)

    file_list = os.listdir(save_path)
    for file in file_list:
        if file.endswith(".zip"):
            upload_file(os.path.join(save_path,file),file)

