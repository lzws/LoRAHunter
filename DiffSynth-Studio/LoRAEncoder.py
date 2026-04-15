import torch
from transformers import CLIPTokenizer, CLIPModel
import pandas as pd
from tqdm import tqdm
import os
from einops import rearrange
from diffsynth.core import load_state_dict
from diffsynth.core.loader import hash_model_file,convert_keys_dict_to_single_str,load_keys_dict
import json

def low_version_attention(query, key, value, attn_bias=None):
    scale = 1 / query.shape[-1] ** 0.5
    query = query * scale
    attn = torch.matmul(query, key.transpose(-2, -1))
    if attn_bias is not None:
        attn = attn + attn_bias
    attn = attn.softmax(-1)
    return attn @ value


class Attention(torch.nn.Module):

    def __init__(self, q_dim, num_heads, head_dim, kv_dim=None, bias_q=False, bias_kv=False, bias_out=False):
        super().__init__()
        dim_inner = head_dim * num_heads
        kv_dim = kv_dim if kv_dim is not None else q_dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.to_q = torch.nn.Linear(q_dim, dim_inner, bias=bias_q)
        self.to_k = torch.nn.Linear(kv_dim, dim_inner, bias=bias_kv)
        self.to_v = torch.nn.Linear(kv_dim, dim_inner, bias=bias_kv)
        self.to_out = torch.nn.Linear(dim_inner, q_dim, bias=bias_out)

    def interact_with_ipadapter(self, hidden_states, q, ip_k, ip_v, scale=1.0):
        batch_size = q.shape[0]
        ip_k = ip_k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        ip_v = ip_v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        ip_hidden_states = torch.nn.functional.scaled_dot_product_attention(q, ip_k, ip_v)
        hidden_states = hidden_states + scale * ip_hidden_states
        return hidden_states

    def torch_forward(self, hidden_states, encoder_hidden_states=None, attn_mask=None, ipadapter_kwargs=None, qkv_preprocessor=None):
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        batch_size = encoder_hidden_states.shape[0]

        q = self.to_q(hidden_states)
        k = self.to_k(encoder_hidden_states)
        v = self.to_v(encoder_hidden_states)

        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        if qkv_preprocessor is not None:
            q, k, v = qkv_preprocessor(q, k, v)

        hidden_states = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        if ipadapter_kwargs is not None:
            hidden_states = self.interact_with_ipadapter(hidden_states, q, **ipadapter_kwargs)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.head_dim)
        hidden_states = hidden_states.to(q.dtype)

        hidden_states = self.to_out(hidden_states)

        return hidden_states
    
    def xformers_forward(self, hidden_states, encoder_hidden_states=None, attn_mask=None):
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        q = self.to_q(hidden_states)
        k = self.to_k(encoder_hidden_states)
        v = self.to_v(encoder_hidden_states)

        q = rearrange(q, "b f (n d) -> (b n) f d", n=self.num_heads)
        k = rearrange(k, "b f (n d) -> (b n) f d", n=self.num_heads)
        v = rearrange(v, "b f (n d) -> (b n) f d", n=self.num_heads)

        if attn_mask is not None:
            hidden_states = low_version_attention(q, k, v, attn_bias=attn_mask)
        else:
            import xformers.ops as xops
            hidden_states = xops.memory_efficient_attention(q, k, v)
        hidden_states = rearrange(hidden_states, "(b n) f d -> b f (n d)", n=self.num_heads)

        hidden_states = hidden_states.to(q.dtype)
        hidden_states = self.to_out(hidden_states)

        return hidden_states

    def forward(self, hidden_states, encoder_hidden_states=None, attn_mask=None, ipadapter_kwargs=None, qkv_preprocessor=None):
        return self.torch_forward(hidden_states, encoder_hidden_states=encoder_hidden_states, attn_mask=attn_mask, ipadapter_kwargs=ipadapter_kwargs, qkv_preprocessor=qkv_preprocessor)

class CLIPEncoderLayer(torch.nn.Module):
    def __init__(self, embed_dim, intermediate_size, num_heads=12, head_dim=64, use_quick_gelu=True):
        super().__init__()
        self.attn = Attention(q_dim=embed_dim, num_heads=num_heads, head_dim=head_dim, bias_q=True, bias_kv=True, bias_out=True)
        self.layer_norm1 = torch.nn.LayerNorm(embed_dim)
        self.layer_norm2 = torch.nn.LayerNorm(embed_dim)
        self.fc1 = torch.nn.Linear(embed_dim, intermediate_size)
        self.fc2 = torch.nn.Linear(intermediate_size, embed_dim)

        self.use_quick_gelu = use_quick_gelu

    def quickGELU(self, x):
        return x * torch.sigmoid(1.702 * x)
    
    def forward(self, hidden_states, attn_mask=None):
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.attn(hidden_states, attn_mask=attn_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.fc1(hidden_states)
        if self.use_quick_gelu:
            hidden_states = self.quickGELU(hidden_states)
        else:
            hidden_states = torch.nn.functional.gelu(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class LoRALayerBlock(torch.nn.Module):
    def __init__(self, L, dim_in):
        super().__init__()
        self.x = torch.nn.Parameter(torch.randn(1, L, dim_in))

    def forward(self, lora_A, lora_B):
        out = self.x @ lora_A.T @ lora_B.T
        return out
    

class LoRAEmbedder(torch.nn.Module):
    def __init__(self, lora_patterns=None, L=1, out_dim=2048):
        super().__init__()
        if lora_patterns is None:
            lora_patterns = self.default_lora_patterns()
            
        model_dict = {}
        for lora_pattern in lora_patterns:
            name, dim = lora_pattern["name"], lora_pattern["dim"][0]
            model_dict[name.replace(".", "___")] = LoRALayerBlock(L, dim)
        self.model_dict = torch.nn.ModuleDict(model_dict)
        
        proj_dict = {}
        for lora_pattern in lora_patterns:
            layer_type, dim = lora_pattern["type"], lora_pattern["dim"][1]
            if layer_type not in proj_dict:
                proj_dict[layer_type.replace(".", "___")] = torch.nn.Linear(dim, out_dim)
        self.proj_dict = torch.nn.ModuleDict(proj_dict)
        
        self.lora_patterns = lora_patterns
        

    def default_lora_patterns(self):
        lora_patterns = []
        # lora_dict = {
        #     "attn.add_k_proj": (3072, 3072), "attn.add_q_proj": (3072, 3072), "attn.add_v_proj": (3072, 3072), "attn.to_add_out": (3072, 3072),
        #     "attn.to_k": (3072, 3072), "attn.to_out.0": (3072, 3072), "attn.to_q": (3072, 3072), "attn.to_v": (3072, 3072),
        #     "img_mlp.net.2": (12288, 3072), "img_mod.1": (3072, 18432), "txt_mlp.net.2": (12288, 3072), "txt_mod.1": (3072, 18432),
        # }
        lora_dict = {
            "attn.add_k_proj": (3072, 3072), "attn.add_q_proj": (3072, 3072), "attn.add_v_proj": (3072, 3072), "attn.to_add_out": (3072, 3072),
            "attn.to_k": (3072, 3072), "attn.to_out.0": (3072, 3072), "attn.to_q": (3072, 3072), "attn.to_v": (3072, 3072),
            "img_mlp.net.2": (12288, 3072), "txt_mlp.net.2": (12288, 3072)
        }
        # 如果只编码30层呢
        for i in range(60):
            for suffix in lora_dict:
                lora_patterns.append({
                    "name": f"transformer_blocks.{i}.{suffix}",
                    "dim": lora_dict[suffix],
                    "type": suffix,
                })
        return lora_patterns
        
    def forward(self, lora):
        lora_emb = []
        for lora_pattern in self.lora_patterns:
            name, layer_type = lora_pattern["name"], lora_pattern["type"]
            lora_A = lora[name + ".lora_A.weight"]
            lora_B = lora[name + ".lora_B.weight"]
            lora_out = self.model_dict[name.replace(".", "___")](lora_A, lora_B)
            lora_out = self.proj_dict[layer_type.replace(".", "___")](lora_out)
            lora_emb.append(lora_out)
        lora_emb = torch.concat(lora_emb, dim=1)
        return lora_emb


class LoRAEncoder(torch.nn.Module):
    def __init__(self, embed_dim=768, max_position_embeddings=600, num_encoder_layers=4, encoder_intermediate_size=3072, L=1):
        super().__init__()
        max_position_embeddings *= L

        self.cls_token = torch.nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        
        # Embedder
        self.embedder = LoRAEmbedder(L=L, out_dim=embed_dim)

        # position_embeds (This is a fixed tensor)
        self.position_embeds = torch.nn.Parameter(torch.zeros(1, max_position_embeddings, embed_dim))

        # encoders
        self.encoders = torch.nn.ModuleList([CLIPEncoderLayer(embed_dim, encoder_intermediate_size) for _ in range(num_encoder_layers)])

        # attn_mask
        # self.attn_mask = self.attention_mask(max_position_embeddings)

        # final_layer_norm
        self.final_layer_norm = torch.nn.LayerNorm(embed_dim)

    def attention_mask(self, length):
        mask = torch.empty(length, length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    def forward(self, lora):
        embeds = self.embedder(lora) + self.position_embeds
        # attn_mask = self.attn_mask.to(device=embeds.device, dtype=embeds.dtype)
        cls = self.cls_token.expand(embeds.size(0), -1, -1)
        embeds = torch.cat([cls, embeds], dim=1)
        for encoder_id, encoder in enumerate(self.encoders):
            embeds = encoder(embeds, attn_mask=None)
        embeds = self.final_layer_norm(embeds)
        out = embeds[:, 0]
        return out



def model_stats(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("=" * 60)
    print(model.__class__.__name__)
    print(f"Total params:      {total:,} ({total / 1e6:.3f} M)")
    print(f"Trainable params:  {trainable:,} ({trainable / 1e6:.3f} M)")
    print(f"Frozen params:     {total - trainable:,} ({(total - trainable) / 1e6:.3f} M)")
    print("=" * 60)

if __name__ == "__main__":
    lora_path = "/shark/zhiwen/LoRAHunter/Qwen_LoRA/Devilworld/Sandrone/25.safetensors"

    lora_encoder = LoRAEncoder().to(dtype=torch.bfloat16, device="cuda:4")

    print(lora_encoder.type)

    # model_stats(lora_encoder)


    # lora_state_dict1 = load_state_dict(lora_path, torch_dtype=torch.bfloat16, device="cuda:4")
    # lora_emb = lora_encoder(lora_state_dict1)
    # print(lora_emb.shape)

    # # lora_state_dict1 = load_keys_dict(lora_path)
    # with open('lora_state_dict1.txt', 'w') as f:
    #     for k,v in lora_state_dict1.items():
    #         f.write(f"{k}: {v.shape}\n")
    # print(len(lora_state_dict1))
    # keys_str = convert_keys_dict_to_single_str(lora_state_dict1)
    # print(keys_str)

    lora_files = '/shark/zhiwen/LoRAHunter/train_mini_datas.jsonl'
    lora_root_path = '/shark/zhiwen/LoRAHunter/Qwen_LoRA'

