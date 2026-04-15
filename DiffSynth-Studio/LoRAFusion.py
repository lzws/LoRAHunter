from CARLoS_prompt import prompts_for_indexing
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
import torch
import torch.nn as nn
import os
from diffsynth.utils.lora import GeneralLoRALoader
from diffsynth.core import load_state_dict
import math
import torch.nn.functional as F
# qwen lora 的动态加载
# 融合模块需要分层加载

# 融合模块用注意力机制
# 用第一个 token 来作为门控的输入
# 替换 dit 里面的某些线性层
# 加attention gate module
# 用第一个 token 来作为门控的输入
# 输入 x 经过一个 linear 层，输出 q
# x 过LoRA的 lora out 作为 v，v 经过一个 linear 层，输出 k
# q @ k 然后softmax得到每个 lora 的权重

# 不一定要每个层都加这个自适应门控
# 在 query 和 output 层用门控
# 在 key 和 value 层用 平均聚合

class LoRAMerger(torch.nn.Module):
    def __init__(
        self,
        base_layer: torch.nn.Linear,
        name: str="",
        gate_dim: int=16
    ):
        super().__init__()
        self.base_layer = base_layer
        self.lora_A_weights = []
        self.lora_B_weights = []
        self.name = name

        self.q_linear = nn.Linear(base_layer.in_features, gate_dim, bias=False)
        self.k_linear = nn.Linear(base_layer.out_features, gate_dim, bias=False)  # 共享
        self.gate_dim = gate_dim

        # 冻结原始权重
        for param in self.base_layer.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        original_is_2d = (x.dim() == 2)
        if original_is_2d:
            x = x.unsqueeze(1)

        out = self.base_layer(x)

        if len(self.lora_A_weights) == 0:
            return out.squeeze(1) if original_is_2d else out

        q = self.q_linear(x[:, 0, :])

        lora_outs = []
        scores = []

        for lora_A, lora_B in zip(self.lora_A_weights, self.lora_B_weights):
            v = x @ lora_A.T @ lora_B.T
            lora_outs.append(v)

            k = self.k_linear(v[:, 0, :])
            score = (q * k).sum(dim=-1) / math.sqrt(self.gate_dim)
            scores.append(score)

        scores = torch.stack(scores, dim=-1)
        weights = F.softmax(scores, dim=-1)

        for i, v in enumerate(lora_outs):
            out = out + weights[:, i].view(-1, 1, 1) * v

        if original_is_2d:
            out = out.squeeze(1)

        return out


def replace_target_modules_with_lora_merger(
    model: torch.nn.Module,
    target_modules: list[str],
    upcast_dtype: torch.dtype = None,
):
    """
    将所有模块名包含 target_modules 中任意字符串的 Linear 层替换为 MoE LoRA
    
    Args:
        model: 要修改的模型
    """
    # 阶段1: 收集所有目标模块名称
    target_names = []
    for name, module in model.named_modules():
        # 检查模块名是否包含 target_modules 中的任意一个字符串
        if any(target_str in name for target_str in target_modules):
            if isinstance(module, nn.Linear):
                target_names.append(name)
    
    if not target_names:
        print(f"⚠️ No modules found containing any of: {target_modules}")
        return model
    
    print(f"✅ Found {len(target_names)} modules to replace with MoE LoRA:")
    # for name in target_names:
    #     print(f"  - {name}")
    
    # 阶段2: 按路径深度降序排序（先处理深层模块）
    target_names.sort(key=lambda x: len(x.split('.')), reverse=True)
    
    # 阶段3: 安全替换
    for full_name in target_names:
        *parent_path, module_name = full_name.split('.')
        
        # 导航到父模块
        parent_module = model
        for p in parent_path:
            parent_module = getattr(parent_module, p)
        
        # 获取原始层
        original_layer = getattr(parent_module, module_name)
        
        # 创建 MoE LoRA 层
        lora_merger_layer = LoRAMerger(
            base_layer=original_layer,
            name=full_name,
        )
        
        # 替换
        setattr(parent_module, module_name, lora_merger_layer)
        # print(f"✅ Replaced {full_name} with MoE LoRA ({num_experts} experts)")
    if upcast_dtype is not None:
        for param in model.parameters():
            if param.requires_grad:
                param.data = param.to(upcast_dtype)
    return model


def load_lora(pipe, lora_path):

    lora = load_state_dict(lora_path, torch_dtype=pipe.torch_dtype, device="cpu")
    lora_loader = GeneralLoRALoader()
    lora = lora_loader.convert_state_dict(lora)

    module = pipe.dit
    updated_num = 0
    for _, module in module.named_modules():
        if isinstance(module, LoRAMerger):
            # print(module.name)
            name = module.name
            lora_a_name = f'{name}.lora_A.weight'
            lora_b_name = f'{name}.lora_B.weight'
            if lora_a_name in lora and lora_b_name in lora:
                updated_num += 1
                module.lora_A_weights.append(lora[lora_a_name].detach().to(pipe.device))
                module.lora_B_weights.append(lora[lora_b_name].detach().to(pipe.device))
    # print(f"{updated_num} tensors are patched by LoRA.")
            
            
def clear_lora(pipe):
    module = pipe.dit
    # for _, module in module.named_modules():
    #     if isinstance(module, LoRAMerger):
    #         print(f"clear {len(module.lora_A_weights)} tensors in {module.name}")
    #         break
    for _, module in module.named_modules():
        if isinstance(module, LoRAMerger):
            module.lora_A_weights.clear()
            module.lora_B_weights.clear()
            del module.lora_A_weights
            del module.lora_B_weights
            module.lora_A_weights = []
            module.lora_B_weights = []
    # for _, module in module.named_modules():
    #     if isinstance(module, LoRAMerger):
    #         print(f"{len(module.lora_A_weights)} tensors in {module.name}")
    #         break


if __name__ == "__main__":
    pipe = QwenImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=f"cuda:1",
        model_configs=[
            ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
            ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
            ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
        ],
        tokenizer_config=ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/"),
    )


    # moelora_target_modules = "to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1"
    # model_ = replace_target_modules_with_lora_merger(
    #     getattr(pipe, 'dit'),
    #     target_modules=moelora_target_modules.split(","),
    # )
    # setattr(pipe, 'dit', model_.to(pipe.device,pipe.torch_dtype))
    # load_lora(pipe, "/shark/zhiwen/LoRAHunter/DiffSynth-Studio/models/train/Qwen-Image_lora/epoch-4.safetensors")

    lora_path = "/shark/zhiwen/LoRAHunter/DiffSynth-Studio/models/train/Qwen-Image_lora/style_lora/3D_Chibi/epoch-1.safetensors"
    lora_path = "/shark/zhiwen/LoRAHunter/DiffSynth-Studio/models/train/Qwen-Image_lora/style_lora/Chinese_Ink/epoch-1.safetensors"
    lora_path = "/shark/zhiwen/LoRAHunter/DiffSynth-Studio/models/train/Qwen-Image_lora/style_lora/Van_Gogh/epoch-1.safetensors"

    pipe.load_lora(pipe.dit, lora_path)

    # prompt = "In 3D_Chibi style，A backpack on table"
    prompt = "In Van_Gogh style，A backpack on table"
    image = pipe(prompt, seed=42, num_inference_steps=30)

    image.save("test10.png")