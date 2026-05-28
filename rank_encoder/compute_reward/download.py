#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('AI-ModelScope/PickScore_v1',local_dir='models/AI-ModelScope/PickScore_v1')

python compute_reward.py \
  --input_jsonl train_dataset/train_prompts_candidates.jsonl \
  --output_jsonl debug_reward.jsonl \
  --image_root images \
  --device cuda:0 \
  --start 0 \
  --end 2