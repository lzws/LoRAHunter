
#模型下载
from modelscope import snapshot_download
import zipfile
import os
def download_model(model_id):
    model_dir = snapshot_download(model_id,local_dir=f"./dataset/{model_id}")


def unzip_file(file_path='',target_dir=''):
    os.makedirs(target_dir,exist_ok=True)

    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(target_dir)


if __name__ == "__main__":
    file_path = '/shark/zhiwen/LoRAHunter/baseline/dataset/Stylusdocs/stylusdocsv2.zip'
    target_dir = '/shark/zhiwen/LoRAHunter/baseline/dataset/Stylusdocsv2'
    unzip_file(file_path,target_dir)