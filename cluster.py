import os,json



def merge_tag_from_b_to_a(file_a_path, file_b_path, output_path, id_field="model_id", tag_field="t_tag"):
    """
    从文件 B 中提取 tag 字段，根据 model_id 合并到文件 A 的对应数据中，并保存到新文件。
    
    :param file_a_path: 主数据文件 (基础数据)
    :param file_b_path: 参考文件 (提供 tag 数据)
    :param output_path: 输出文件路径
    :param id_field: 关联键，默认 "model_id"
    :param tag_field: 需要从 B 提取的字段名，默认 "tag"
    """
    
    # --- 第一步：读取文件 B，构建 {model_id: tag} 映射字典 ---
    print(f"正在加载参考文件 ({file_b_path}) 构建映射...")
    id_to_tag_map = {}
    missing_tag_count = 0
    
    try:
        with open(file_b_path, 'r', encoding='utf-8') as f_b:
            for line_num, line in enumerate(f_b, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    mid = data.get(id_field)
                    tag_val = data.get(tag_field)
                    tag_val2 = data.get("prompts")
                    
                    if mid is not None:
                        # 如果 B 中有重复的 model_id，后面的会覆盖前面的
                        # 如果 tag 缺失，可以存 None 或者跳过，这里选择存 None 以便调试
                        id_to_tag_map[mid] = (tag_val, tag_val2)
                        if tag_val is None:
                            missing_tag_count += 1
                    else:
                        # 如果没有 model_id，这行数据无法用于关联
                        pass
                except json.JSONDecodeError:
                    print(f"警告: 文件 B 第 {line_num} 行 JSON 格式错误，已跳过。")
        
        print(f"映射构建完成。共加载 {len(id_to_tag_map)} 个唯一 ID。")
        if missing_tag_count > 0:
            print(f"注意: 其中有 {missing_tag_count} 条数据缺少 '{tag_field}' 字段。")
            
    except FileNotFoundError:
        print(f"错误: 找不到文件 {file_b_path}")
        return

    # --- 第二步：遍历文件 A，匹配并注入 tag 字段 ---
    print(f"正在处理主文件 ({file_a_path}) 并合并数据...")
    match_count = 0
    no_match_count = 0
    
    with open(file_a_path, 'r', encoding='utf-8') as f_a, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        
        for line_num, line in enumerate(f_a, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                mid = data.get(id_field)
                
                # 核心逻辑：查找并合并
                if mid is not None and mid in id_to_tag_map:
                    # 获取 B 中的 tag
                    tag_val = id_to_tag_map[mid][0]
                    tag_val2 = id_to_tag_map[mid][1]
                    
                    # 【关键步骤】将 tag 注入到当前数据字典中
                    # 如果 A 中原本就有 tag 字段，这行代码会覆盖它。
                    # 如果想保留旧的，可以改名为 source_tag 或检查是否存在
                    data[tag_field] = tag_val
                    data["prompts"] = tag_val2
                    # 写入更新后的数据
                    f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
                    match_count += 1
                else:
                    # 如果不需要保留未匹配的数据，注释掉下面这行即可
                    # 如果需要保留 A 中所有数据（没匹配的就不加 tag），请取消下面这行的注释：
                    # f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
                    no_match_count += 1
                    continue 
                    
            except json.JSONDecodeError:
                print(f"警告: 文件 A 第 {line_num} 行 JSON 格式错误，已跳过。")
                continue

    print("-" * 30)
    print("处理完成！")
    print(f"成功匹配并注入 tag 的数量: {match_count}")
    print(f"未找到对应 model_id 的数量: {no_match_count} (这些行未被写入输出文件)")
    print(f"结果已保存至: {output_path}")

# --- 使用示例 ---
if __name__ == "__main__":
    # 替换为你的实际文件名
    file_a = "Available_LoRA_carlos_tags2.jsonl"  # 包含完整数据的文件
    file_b = "cluster_prompt.jsonl"  # 包含目标 model_id 的文件
    output_file = "train_mini_datas.jsonl"
    
    merge_tag_from_b_to_a(file_a, file_b, output_file)
