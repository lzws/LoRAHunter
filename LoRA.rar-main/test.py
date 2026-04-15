import torch

def inspect_pth_file(file_path):
    try:
        # 加载文件
        # weights_only=True 是较新版本 PyTorch 的安全建议，但如果文件包含非张量数据可能需要设为 False
        # 如果不确定，先尝试 weights_only=False (默认行为)
        data = torch.load(file_path, map_location='cpu', weights_only=False)
        
        print(f"文件加载成功: {file_path}")
        print(f"数据类型: {type(data)}")
        print("-" * 30)

        # 判断数据类型并处理
        if isinstance(data, dict):
            print("这是一个字典 (dict)，包含以下 keys:")
            for key in data.keys():
                # 尝试获取对应值的类型和形状（如果是张量）
                value = data[key]
                info = ""
                if hasattr(value, 'shape'):
                    info = f" -> 形状: {value.shape}, 类型: {value.dtype}"
                elif isinstance(value, torch.nn.Module):
                    info = " -> 类型: nn.Module (模型)"
                else:
                    info = f" -> 类型: {type(value)}"
                
                print(f"  - '{key}'{info}")
            
            # 如果想看具体内容，可以取消下面这行的注释
            # print("\n完整数据内容:\n", data)
            
        elif isinstance(data, list):
            print(f"这是一个列表 (list)，长度: {len(data)}")
            for i, item in enumerate(data):
                print(f"  索引 [{i}]: 类型 {type(item)}")
                
        elif hasattr(data, 'state_dict'):
            # 如果加载的直接是一个模型对象
            print("这是一个 PyTorch 模型对象。")
            print("你可以调用 .state_dict() 来查看参数 keys:")
            state_dict = data.state_dict()
            for key in state_dict.keys():
                val = state_dict[key]
                print(f"  - '{key}' -> 形状: {val.shape}")
                
        else:
            print("这不是一个字典或模型，而是一个单一对象：")
            print(f"  类型: {type(data)}")
            if hasattr(data, 'shape'):
                print(f"  形状: {data.shape}")
            # print(f"  内容预览: {data}")

    except Exception as e:
        print(f"加载失败: {e}")
        print("提示：如果文件是用高版本 PyTorch 保存的，低版本可能无法加载，反之亦然。")

# 使用示例：将 'model.pth' 替换为你的文件路径
inspect_pth_file('/shark/zhiwen/LoRAHunter/LoRA.rar-main/models/koala_700m_hypernet.pth')