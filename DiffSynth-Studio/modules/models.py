import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

from .qwen3_vl_embedding import Qwen3VLEmbedder

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

# "openai/clip-vit-large-patch14"
class TextImageEncoder(torch.nn.Module):
    def __init__(self, model_name="/shark/zhiwen/LoRAHunter/DiffSynth-Studio/models/AI-ModelScope/clip-vit-large-patch14", dtype=torch.float, device="cuda"):
        super().__init__()
        self.device = device
        self.dtype = dtype
        # self.model, self.preprocess = clip.load('ViT-L/14@336px', download_root=model_name, device=device)
        print(f"load clip model from: {model_name}")
        self.model = CLIPModel.from_pretrained(model_name).to(dtype=torch.float)
        self.processor = CLIPProcessor.from_pretrained(model_name)



    @torch.no_grad()
    def encoding_image(self,image_path=''):
        image = Image.open(image_path)
        image_input = self.processor(images=image, return_tensors="pt").to(self.model.device)
        image_features = self.model.get_image_features(**image_input)
        return image_features.pooler_output #[1,768]
        

    @torch.no_grad()
    def encoding_text(self,text=''):
        text_input = self.processor(text=text, return_tensors="pt", max_length=77, truncation=True, padding=True).to(self.model.device)
        text_features = self.model.get_text_features(**text_input)
        return text_features.pooler_output #[1, 768]


class QwenVLEncoder(torch.nn.Module):
    def __init__(self, model_name="/shark/zhiwen/LoRAHunter/DiffSynth-Studio/models/Qwen/Qwen3-VL-Embedding-2B", device='cuda'):
        super().__init__()
        self.device = device
        self.model = Qwen3VLEmbedder(model_name_or_path=model_name,device = torch.device(device))
        print(f"load model from: {model_name}")

    @torch.no_grad()
    def encoding_text(self, text):
        if isinstance(text, str):
            text = [text]

        queries = [{"text": t} for t in text]
        text_features = self.model.process(queries) # [N, 2048]
        return text_features
    
    @torch.no_grad()
    def encoding_image(self, image_path):
        if isinstance(image_path, str):
            image_path = [image_path]

        queries = [{"image": i} for i in image_path]
        image_features = self.model.process(queries) # [N, 2048]
        return image_features

    @torch.no_grad()
    def encoding_images(self, inputs):
        image_features = self.model.get_embeddings(inputs) # [N, 2048]
        return image_features



