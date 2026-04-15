import os,json
import torch

metadata_path = "/shark/zhiwen/LoRAHunter/DiffSynth-Studio/LoRA_Merger_train_data.csv"

# 每一层都是一个 vae 模型
# 有 a b 矩阵，不同rank处理还是统一 乘成一个矩阵再处理
# 稀疏 vae
# 输入一个矩阵，输出一个向量
# cnn

# 怎么训练 每一层单独训练 所有层怎么一起训练

# 每一层独立的vae，还是共享vae
