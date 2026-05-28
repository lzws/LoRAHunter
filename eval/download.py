#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('ZhipuAI/ImageReward',local_dir='./models/ZhipuAI/ImageReward')