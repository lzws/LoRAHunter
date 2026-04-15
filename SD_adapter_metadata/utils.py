import pandas as pd


def convert_jsonl_to_csv(input_jsonl, output_csv):


    # 一行代码搞定
    # lines=True 告诉 pandas 每一行都是一个独立的 JSON 对象

    df = pd.read_json(input_jsonl, lines=True)
    df.reset_index(inplace=True)
    df.to_csv(output_csv, index=False) # index=False 避免写入行号
    

    print("✅ 转换完成")


import json
import random
import os

def sample_unique_lines(source_file, existing_file, output_file, target_count=10000):
    """
    从 source_file 中随机挑选 target_count 行，
    确保选出的行中的 'adapter_id' 不在 existing_file 中。
    """
    
    # 1. 读取现有文件中的 adapter_id，存入集合以便快速查找
    print(f"正在读取现有文件: {existing_file} ...")
    existing_ids = set()
    
    if os.path.exists(existing_file):
        with open(existing_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    # 假设字段名为 'adapter_id'，根据实际情况修改
                    if 'adapter_id' in data:
                        existing_ids.add(data['adapter_id'])
                except json.JSONDecodeError:
                    continue
        print(f"现有文件中包含 {len(existing_ids)} 个唯一的 adapter_id。")
    else:
        print(f"警告: 现有文件 {existing_file} 不存在，将视为无重复项。")

    # 2. 读取源文件的所有有效行（排除掉重复的 ID）
    print(f"正在筛选源文件: {source_file} ...")
    candidates = []
    
    with open(source_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                adapter_id = data.get('adapter_id')
                
                # 核心逻辑：如果 ID 不在现有集合中，则加入候选列表
                if adapter_id not in existing_ids:
                    candidates.append(line)
            except json.JSONDecodeError:
                continue

    print(f"筛选后剩余 {len(candidates)} 个可用候选行。")

    # 3. 随机抽取
    if len(candidates) < target_count:
        print(f"警告: 可用候选行 ({len(candidates)}) 少于目标数量 ({target_count})。将返回所有可用行。")
        selected_lines = candidates
    else:
        # 使用 random.sample 进行无放回随机抽样
        selected_lines = random.sample(candidates, target_count)
        print(f"成功随机抽取 {target_count} 行。")

    # 4. 写入新文件
    print(f"正在写入输出文件: {output_file} ...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in selected_lines:
            f.write(line + '\n')
    
    print("✅ 完成！")

# --- 配置区域 ---
# 请在这里修改你的文件路径
SOURCE_JSONL = "data_source.jsonl"      # 你的源文件（大文件）
EXISTING_JSONL = "data_existing.jsonl" # 你的现有文件（黑名单）
OUTPUT_JSONL = "data_sampled.jsonl"    # 输出结果


    


if __name__ == "__main__":
    input_jsonl = "/shark/zhiwen/LoRAHunter/SD_adapter_metadata/sd_lora_1/bad_file_adapters.jsonl"
    output_csv = "/shark/zhiwen/LoRAHunter/SD_adapter_metadata/sd_lora_1/bad_file_adapters.csv"
    # convert_jsonl_to_csv(input_jsonl, output_csv)

    SOURCE_JSONL = "exist_file_adapters.jsonl"      # 你的源文件（大文件）
    EXISTING_JSONL = "train_lora_10k.jsonl" # 你的现有文件（黑名单）
    OUTPUT_JSONL = "train_lora_10k_2.jsonl"    # 输出结果
    sample_unique_lines(SOURCE_JSONL, EXISTING_JSONL, OUTPUT_JSONL, target_count=10000)