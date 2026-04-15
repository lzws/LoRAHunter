import sqlite3
import json
import os

def load_sqlite():

    # 数据库文件路径
    db_path = '/shark/zhiwen/LoRAHunter/20260302-194002/lora_records.sqlite'
    output_file = 'LoRA_QWEN_IMAGE_20_B_json.jsonl'
    target_fields = ['model_id','owner','name','title','description','tags','trigger_words','preview_urls','model_card_url','version','stats','files','base_model','chinese_name','official_tags','llm_description','extra_description','cleaned_extra_description']
    json_fields = ['tags','trigger_words','preview_urls','files','stats','official_tags']
    # 使用 with 语句自动处理关闭连接，防止资源泄露
    try:
        # 1. 连接到数据库 (如果文件不存在，connect 会创建一个新的，但查询时通常文件已存在)
        conn = sqlite3.connect(db_path)
        
        # 2. 创建游标对象
        cursor = conn.cursor()
        
        # 3. 执行 SQL 查询
        # 示例：查询 users 表中 age > 25 的所有用户
        # query = "SELECT * FROM lora_records WHERE vision_foundation = ? AND adapter_type = ? AND task = ?"
        query = """
            SELECT * FROM lora_records 
            WHERE vision_foundation = ? 
            AND adapter_type = ? 
            AND task = ? 
            AND (base_model IS NULL OR base_model NOT LIKE '%Image-Edit%')
        """
        cursor.execute(query, ('QWEN_IMAGE_20_B','LoRA','text-to-image-synthesis'))  # 使用参数化查询防止 SQL 注入
        # 3. 获取所有列名
        # cursor.description 是一个元组列表，每个元组的第0个元素是列名
        all_columns = [desc[0] for desc in cursor.description]
        print("所有列名:", all_columns)

        # 4. 获取结果
        # fetchall() 获取所有剩余行，返回一个列表，每个元素是一个元组
        rows = cursor.fetchall()
        
        # 打印结果
        print(f"找到 {len(rows)} 条记录:")

        # 5. 打开文件并逐行写入
        with open(output_file, 'w', encoding='utf-8') as f:
            count = 0
            
            # 逐行处理 (适合大数据量)
            for row in rows:
                # 1. 将行元组转为字典
                full_record = dict(zip(all_columns, row))
                
                # 2. 筛选目标字段
                filtered_record = {key: full_record[key] for key in target_fields}
                
                # 3. 【核心步骤】解析指定的 JSON 字符串字段
                for j_field in json_fields:
                    if j_field in filtered_record:
                        val = filtered_record[j_field]
                        
                        # 只有当值不为 None 且是字符串时才尝试解析
                        if isinstance(val, str):
                            try:
                                # 将字符串 '{"a":1}' 转换为 Python 字典 {'a': 1}
                                filtered_record[j_field] = json.loads(val)
                            except json.JSONDecodeError:
                                # 如果字符串不是合法的 JSON，可以选择保留原字符串或报错
                                # 这里选择保留原字符串并打印警告，防止程序中断
                                print(f"警告: 第 {count+1} 条数据的字段 '{j_field}' 不是合法的 JSON 字符串，保持原样。")
                                pass 
                        elif val is None:
                            # 如果是 NULL，保持 None
                            pass
                        else:
                            # 如果数据库中该字段已经是其他类型（极少见），跳过
                            pass

                # 4. 写入文件
                # 此时 filtered_record 中的 json_fields 已经是 dict/list 类型
                # json.dumps 会正确地将其序列化为嵌套的 JSON 结构
                json_line = json.dumps(filtered_record, ensure_ascii=False)
                f.write(json_line + '\n')
                
                count += 1
                
        print(f"完成！共处理 {count} 条数据，已保存至 {output_file}")

        # for row in rows:
        #     # row 是一个元组，例如: (1, 'Alice', 30)
        #     print(f"ID: {row[0]}, 姓名: {row[1]}, 年龄: {row[2]}")
            
        # 如果你想查看列名（可选）
        # col_names = [description[0] for description in cursor.description]
        # print("列名:", col_names)

    except sqlite3.Error as e:
        print(f"数据库发生错误: {e}")
    finally:
        # 5. 关闭连接
        if conn:
            conn.close()


def sort_jsonl_by_stats(input_file, output_file, stats_key='gen_count'):
    """
    读取 JSONL 文件，根据 stats 字段中的 gen_count 进行降序排序，并保存到新文件。
    
    :param input_file: 输入的 JSONL 文件路径
    :param output_file: 输出的 JSONL 文件路径
    :param stats_key: stats 字典中用于排序的键名 (默认为 'gen_count')
    """
    data_list = []
    error_count = 0
    
    print(f"正在读取文件: {input_file} ...")
    
    try:
        # 1. 读取所有数据到内存
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    record = json.loads(line)
                    
                    # 验证结构是否存在
                    if 'stats' not in record:
                        print(f"警告: 第 {line_num} 行缺少 'stats' 字段，跳过。")
                        error_count += 1
                        continue
                    
                    stats = record['stats']
                    
                    if not isinstance(stats, dict):
                        print(f"警告: 第 {line_num} 行的 'stats' 不是字典，跳过。")
                        error_count += 1
                        continue
                        
                    if stats_key not in stats:
                        print(f"警告: 第 {line_num} 行的 'stats' 中缺少 '{stats_key}' 字段，跳过。")
                        error_count += 1
                        continue
                    
                    # 获取值并尝试转换为整数/浮点数
                    # 防止数据库中存的是字符串 "100" 导致排序错误 (字符串排序 "100" < "2")
                    val = stats[stats_key]
                    if isinstance(val, str):
                        # 尝试转为数字，如果失败则保留原样（可能导致排序不符合预期）
                        try:
                            val = float(val) if '.' in val else int(val)
                        except ValueError:
                            print(f"警告: 第 {line_num} 行的 '{stats_key}' 值 '{val}' 无法转换为数字，按0处理。")
                            val = 0
                    if not isinstance(val, (int, float)):
                        print(f"警告: 第 {line_num} 行的 '{stats_key}' 值 '{val}' 不是数字，按0处理。")
                        val = 0
                    
                    # 将数值附加到记录中，方便排序 (也可以不修改原数据，只在 key 函数里处理)
                    # 这里我们直接在 sort 的 key 中使用提取逻辑，不修改原数据
                    data_list.append((record, val))
                    
                except json.JSONDecodeError as e:
                    print(f"错误: 第 {line_num} 行 JSON 格式无效: {e}")
                    error_count += 1
                    continue

        if not data_list:
            print("没有发现有效数据进行排序。")
            return

        print(f"读取完成，共 {len(data_list)} 条有效数据。开始排序...")

        # 2. 执行排序
        # x[1] 是我们提取出来的数值，reverse=True 表示降序 (从大到小)
        data_list.sort(key=lambda x: x[1], reverse=True)
        
        print(f"排序完成。正在写入文件: {output_file} ...")

        # 3. 写入新文件
        with open(output_file, 'w', encoding='utf-8') as f:
            for record, _ in data_list:
                # 确保输出的是标准的 JSON 行
                json_line = json.dumps(record, ensure_ascii=False)
                f.write(json_line + '\n')
        
        print(f"成功！已生成排序后的文件: {output_file}")
        if error_count > 0:
            print(f"注意: 处理过程中有 {error_count} 条数据因格式问题被跳过。")

    except FileNotFoundError:
        print(f"错误: 找不到文件 {input_file}")
    except Exception as e:
        print(f"发生未知错误: {e}")

def extract_top_n(input_file, output_file, n=2500):
    count = 0
    
    print(f"正在从 {input_file} 提取前 {n} 条数据...")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f_in, \
             open(output_file, 'w', encoding='utf-8') as f_out:
            
            for line in f_in:
                line = line.strip()
                if not line:
                    continue
                
                # 可选：验证一下是否是合法的 JSON，防止拷贝了坏数据
                # 如果确定上游文件没问题，可以去掉 try-except 加快速度
                try:
                    json.loads(line) 
                except json.JSONDecodeError:
                    print(f"警告: 跳过第 {count+1} 行，格式无效。")
                    continue
                
                # 写入文件
                f_out.write(line + '\n')
                count += 1
                
                # 达到数量限制，立即停止
                if count >= n:
                    break
        
        if count == 0:
            print("未找到任何有效数据。")
        elif count < n:
            print(f"完成！文件总共只有 {count} 条数据，已全部提取到 {output_file}。")
        else:
            print(f"成功！已提取前 {n} 条数据到 {output_file}。")
            
    except FileNotFoundError:
        print(f"错误: 找不到文件 {input_file}")
    except Exception as e:
        print(f"发生错误: {e}")


import json

def extract_jsonl_lines(input_file, output_file, start_idx, end_idx):
    """
    从 JSONL 文件中提取第 start_idx 行 到 第 end_idx 行 (包含两端) 的数据，
    并保存到新文件。
    
    参数说明:
        input_file: 输入文件路径
        output_file: 输出文件路径
        start_idx: 起始行号 (从 0 开始计数，类似 Python 列表索引)
        end_idx:   结束行号 (包含该行，类似切片 [start:end+1])
                   如果设为 -1 或 None，表示提取到文件末尾
    """
    
    # 校验参数
    if start_idx < 0:
        raise ValueError("start_idx 必须大于等于 0")
    if end_idx is not None and end_idx != -1 and end_idx < start_idx:
        raise ValueError("end_idx 必须大于等于 start_idx")

    count = 0
    extracted_count = 0
    
    print(f"开始处理：提取行 [{start_idx} : {end_idx}]")

    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            # 跳过空行
            if not line.strip():
                continue
            
            # 当前行索引
            current_idx = count
            
            # 判断是否在目标范围内
            if current_idx >= start_idx:
                # 如果设置了结束行且当前行超过了结束行，则停止
                if end_idx is not None and end_idx != -1 and current_idx > end_idx:
                    break
                
                # 写入文件 (直接写入原始字符串即可，因为每行本身就是合法的 JSON)
                # 为了保险，也可以先解析再序列化，但直接写入速度更快
                f_out.write(line)
                extracted_count += 1
            
            count += 1

    print(f"处理完成！")
    print(f"总读取行数: {count}")
    print(f"成功提取行数: {extracted_count}")
    print(f"已保存至: {output_file}")

# --- 使用示例 ---

# 假设你的文件叫 'data.jsonl'
input_path = 'source_files/LoRA_QWEN_IMAGE_20_B_json_sorted.jsonl'
output_path = 'source_files/LoRA_QWEN_IMAGE_20_B_4k-13k.jsonl'

# 场景 1: 提取第 100 行 到 第 200 行 (索引从 0 开始，即第 101 条到第 201 条数据)
# 注意：这里 end_idx=200 表示包含第 200 行
# extract_jsonl_lines(input_path, output_path, start_idx=100, end_idx=200)

# 场景 2: 提取从第 500 行开始直到文件末尾
extract_jsonl_lines(input_path, output_path, start_idx=4000, end_idx=-1)

# ==========================================
# 使用示例
# ==========================================
# load_sqlite()

input_path = 'LoRA_QWEN_IMAGE_20_B_json.jsonl'   # 上一步生成的文件
output_path = 'LoRA_QWEN_IMAGE_20_B_json_sorted.jsonl'

# # 执行排序
# sort_jsonl_by_stats(
#     input_file=input_path, 
#     output_file=output_path, 
#     stats_key='downloads'
# )

sorted_file = 'source_files/LoRA_QWEN_IMAGE_20_B_json_sorted.jsonl'  # 上一步排序生成的文件
top_2500_file = 'source_files/LoRA_QWEN_IMAGE_20_B_top_4000.jsonl'           # 最终想要的文件

# extract_top_n(
#     input_file=sorted_file, 
#     output_file=top_2500_file, 
#     n=4000
# )