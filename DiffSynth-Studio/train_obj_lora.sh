#!/bin/bash

# ================= 配置区域 =================
# 请修改此处为你存放所有 csv 文件的文件夹绝对路径
CSV_FOLDER_PATH="/shark/zhiwen/LoRAHunter/DiffSynth-Studio/data/obj_data/"

# accelerate 的基础命令部分 (不包含 dataset_metadata_path 和 output_path)
BASE_CMD="accelerate launch --num_processes 1 --gpu_ids 3 examples/qwen_image/model_training/train.py \
  --dataset_base_path '' \
  --max_pixels 1048576 \
  --dataset_repeat 30 \
  --model_id_with_origin_paths 'Qwen/Qwen-Image:transformer/diffusion_pytorch_model*.safetensors,Qwen/Qwen-Image:text_encoder/model*.safetensors,Qwen/Qwen-Image:vae/diffusion_pytorch_model.safetensors' \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt 'pipe.dit.' \
  --lora_base_model 'dit' \
  --lora_target_modules 'to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1' \
  --lora_rank 32 \
  --use_gradient_checkpointing \
  --dataset_num_workers 8 \
  --find_unused_parameters"

# ================= 逻辑区域 =================

echo "开始检查文件夹: $CSV_FOLDER_PATH"

# 检查文件夹是否存在
if [ ! -d "$CSV_FOLDER_PATH" ]; then
    echo "错误: 文件夹不存在！请检查 CSV_FOLDER_PATH 设置。"
    exit 1
fi

# 循环读取文件夹下所有的 .csv 文件
# 使用 find 命令确保只处理 .csv 文件，并按文件名排序
find "$CSV_FOLDER_PATH" -maxdepth 1 -name "*.csv" -type f | sort | while read -r csv_file; do
    
    # 获取文件名（不带路径）
    filename=$(basename "$csv_file")
    
    # 获取文件名（不带 .csv 后缀），用于构建输出目录名
    # 例如: backpack_dog.csv -> backpack_dog
    dataset_name="${filename%.csv}"
    
    echo "----------------------------------------"
    echo "正在处理数据集: $dataset_name"
    echo "源文件: $csv_file"
    
    # 构建输出路径
    # 逻辑：./models/train/Qwen-Image_lora_<dataset_name>
    OUTPUT_PATH="./models/train/Qwen-Image_lora/obj_lora/${dataset_name}"
    
    # 创建输出目录（如果不存在）
    mkdir -p "$OUTPUT_PATH"
    
    # 拼接完整命令
    # 注意：这里重新组合命令，插入动态变化的路径
    FULL_CMD="$BASE_CMD \
      --dataset_metadata_path '$csv_file' \
      --output_path '$OUTPUT_PATH'"
    
    echo "执行命令: $FULL_CMD"
    echo "开始训练..."
    
    # 执行训练命令
    # 如果某个任务失败，脚本可以选择继续或停止。这里设置为继续下一个 (去掉 || exit 1)
    eval $FULL_CMD
    
    if [ $? -ne 0 ]; then
        echo "警告: 数据集 $dataset_name 训练失败，继续下一个..."
    else
        echo "成功: 数据集 $dataset_name 训练完成。"
    fi
    
done

echo "----------------------------------------"
echo "所有任务已提交完毕。"

# nohup bash train_obj_lora.sh > z_train_lora_log/training_obj_log.txt 2>&1 &