
#模型下载
from modelscope import snapshot_download
# model_dir = snapshot_download('AI-ModelScope/CLIP-ViT-H-14-laion2B-s32B-b79K',local_dir='./models/AI-ModelScope/CLIP-ViT-H-14-laion2B-s32B-b79K')


#验证 ModelScope token
from modelscope.hub.api import HubApi
api = HubApi()
api.login('ms-b99bca69-c5d6-48b9-aa6d-58cb49ed57ac')


model_dir = snapshot_download('Qwen/Qwen3-VL-Reranker-8B', local_dir="./models/Qwen/Qwen3-VL-Reranker-8B")
