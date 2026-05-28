import os,json

def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def pre_file():
    metadata_path = "/shark/zhiwen/LoRAHunter/source_files/Available_LoRA_all.jsonl"
    meatadata_1 = "/shark/zhiwen/LoRAHunter/source_files/Available_LoRA_4k-13k_tags2_prompt.jsonl"
    meatadata_2 = "/shark/zhiwen/LoRAHunter/source_files/Available_LoRA_carlos_tags2_prompt.jsonl"

    data_main = load_jsonl(metadata_path)
    data_1 = load_jsonl(meatadata_1)
    data_2 = load_jsonl(meatadata_2)

    # 把 _1 和 _2 建成按 model_file 检索的字典
    extra_map = {}

    for item in data_1 + data_2:
        model_file = item["model_file"]
        extra_map[model_file] = item
    print(f"共有 {len(extra_map)} 条数据")

    # 把额外字段补到主数据里
    merged_data = []
    for item in data_main:
        model_file = item["model_file"]
        extra_item = extra_map.get(model_file, {})

        new_item = item.copy()
        new_item["short_description"] = extra_item.get("short_description")
        new_item['prompts'] = extra_item.get("prompts")



        merged_data.append(new_item)
    
    output_path = "Available_LoRA_all_merged.jsonl"

    with open(output_path, "w", encoding="utf-8") as f:
        for item in merged_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")



def main():
    gcl_tok_file = "/shark/zhiwen/LoRAHunter/DiffSynth-Studio/rank_dataset/train_gcl_topk_qwen.jsonl"
    metadata_path = "/shark/zhiwen/LoRAHunter/DiffSynth-Studio/rank_dataset/Available_LoRA_all_merged.jsonl"

    top_datas = load_jsonl(gcl_tok_file)
    metadatas = load_jsonl(metadata_path)

    extra_map = {}
    for item in metadatas:
        model_file = item["model_file"]
        extra_map[model_file] = item

    exit_files = []
    lora_datas = []
    for td in top_datas:
        model_file = td['model_file']
        if model_file not in exit_files:
            exit_files.append(model_file)
            lora_datas.append(extra_map[model_file])
    
    output_path = "train_rank_lora_all.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for item in lora_datas:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def filter_lora():

    jsonl_path = "/shark/zhiwen/LoRAHunter/DiffSynth-Studio/rank_dataset/Available_LoRA_all_merged.jsonl"
    broken_txt_path = "/shark/zhiwen/LoRAHunter/DiffSynth-Studio/broken_loras.txt"
    output_jsonl_path = "Available_LoRA_filtered_merged.jsonl"

    # 读取坏文件列表
    with open(broken_txt_path, "r", encoding="utf-8") as f:
        broken_loras = {line.strip() for line in f if line.strip()}

    # 过滤并写入新 jsonl
    kept = 0
    removed = 0

    with open(jsonl_path, "r", encoding="utf-8") as fin, \
        open(output_jsonl_path, "w", encoding="utf-8") as fout:
        
        for line in fin:
            line = line.strip()
            if not line:
                continue

            data = json.loads(line)
            model_file = data.get("model_file")

            if model_file in broken_loras:
                removed += 1
                continue

            fout.write(json.dumps(data, ensure_ascii=False) + "\n")
            kept += 1

    print(f"完成: 保留 {kept} 条, 删除 {removed} 条")
    print(f"输出文件: {output_jsonl_path}")



if __name__ == "__main__":
    filter_lora()