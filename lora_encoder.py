import torch
from torch import nn

try:
    from .models import CLIPEncoderLayer
except ImportError:
    from models import CLIPEncoderLayer


class LoRALayerBlock(nn.Module):
    def __init__(self, L, dim_in, num_probes=4):
        super().__init__()
        self.x = nn.Parameter(torch.randn(num_probes, dim_in) * 0.02)
        self.out_proj = nn.Linear(num_probes, L)

    def forward(self, lora_a, lora_b):
        if lora_a.dim() == 4:
            lora_a = lora_a[:, :, 0, 0]
            lora_b = lora_b[:, :, 0, 0]
        x = self.x @ lora_a.transpose(0, 1) @ lora_b.transpose(0, 1)
        x = self.out_proj(x.transpose(0, 1))
        return x.transpose(0, 1).unsqueeze(0)


class LoRALayerBlock2(nn.Module):
    def __init__(self, L, dim_in, num_probes=4):
        super().__init__()
        self.x = nn.Parameter(torch.randn(num_probes, dim_in) * 0.02)
        self.out_proj = nn.Linear(num_probes, L)
        self.pre_norm = nn.LayerNorm(num_probes)
        self.mlp = nn.Sequential(
            nn.Linear(num_probes, num_probes * 4),
            nn.GELU(),
            nn.Linear(num_probes * 4, num_probes),
        )

    def forward(self, lora_a, lora_b):
        if lora_a.dim() == 4:
            lora_a = lora_a[:, :, 0, 0]
            lora_b = lora_b[:, :, 0, 0]
        x = self.x @ lora_a.transpose(0, 1) @ lora_b.transpose(0, 1)
        x = x.transpose(0, 1)
        x = x + self.mlp(self.pre_norm(x))
        x = self.out_proj(x)
        return x.transpose(0, 1).unsqueeze(0)


def default_lora_patterns():
    down_block_dict = {
        "attentions.0.proj_in.lora": [(320, 320), (640, 640), (1280, 1280)],
        "attentions.0.proj_out.lora": [(320, 320), (640, 640), (1280, 1280)],
        "attentions.0.transformer_blocks.0.attn1.processor.to_k_lora": [(320, 320), (640, 640), (1280, 1280)],
        "attentions.0.transformer_blocks.0.attn1.processor.to_v_lora": [(320, 320), (640, 640), (1280, 1280)],
        "attentions.0.transformer_blocks.0.attn1.processor.to_q_lora": [(320, 320), (640, 640), (1280, 1280)],
        "attentions.0.transformer_blocks.0.attn1.processor.to_out_lora": [(320, 320), (640, 640), (1280, 1280)],
        "attentions.0.transformer_blocks.0.attn2.processor.to_k_lora": [(768, 320), (768, 640), (768, 1280)],
        "attentions.0.transformer_blocks.0.attn2.processor.to_v_lora": [(768, 320), (768, 640), (768, 1280)],
        "attentions.0.transformer_blocks.0.attn2.processor.to_q_lora": [(320, 320), (640, 640), (1280, 1280)],
        "attentions.0.transformer_blocks.0.attn2.processor.to_out_lora": [(320, 320), (640, 640), (1280, 1280)],
        "attentions.0.transformer_blocks.0.ff.net.0.proj.lora": [(320, 2560), (640, 5120), (1280, 10240)],
        "attentions.0.transformer_blocks.0.ff.net.2.lora": [(1280, 320), (2560, 640), (5120, 1280)],
        "attentions.1.proj_in.lora": [(320, 320), (640, 640), (1280, 1280)],
        "attentions.1.transformer_blocks.0.attn1.processor.to_k_lora": [(320, 320), (640, 640), (1280, 1280)],
        "attentions.1.transformer_blocks.0.attn1.processor.to_v_lora": [(320, 320), (640, 640), (1280, 1280)],
        "attentions.1.transformer_blocks.0.attn1.processor.to_q_lora": [(320, 320), (640, 640), (1280, 1280)],
        "attentions.1.transformer_blocks.0.attn1.processor.to_out_lora": [(320, 320), (640, 640), (1280, 1280)],
        "attentions.1.transformer_blocks.0.attn2.processor.to_k_lora": [(768, 320), (768, 640), (768, 1280)],
        "attentions.1.transformer_blocks.0.attn2.processor.to_v_lora": [(768, 320), (768, 640), (768, 1280)],
        "attentions.1.transformer_blocks.0.attn2.processor.to_q_lora": [(320, 320), (640, 640), (1280, 1280)],
        "attentions.1.transformer_blocks.0.attn2.processor.to_out_lora": [(320, 320), (640, 640), (1280, 1280)],
        "attentions.1.transformer_blocks.0.ff.net.0.proj.lora": [(320, 2560), (640, 5120), (1280, 10240)],
        "attentions.1.transformer_blocks.0.ff.net.2.lora": [(1280, 320), (2560, 640), (5120, 1280)],
    }
    mid_block_dict = {
        "attentions.0.proj_in.lora": [(320, 320), (640, 640), (1280, 1280)],
        "attentions.0.proj_out.lora": [(320, 320), (640, 640), (1280, 1280)],
        "attentions.0.transformer_blocks.0.attn1.processor.to_k_lora": [(320, 320), (640, 640), (1280, 1280)],
        "attentions.0.transformer_blocks.0.attn1.processor.to_v_lora": [(320, 320), (640, 640), (1280, 1280)],
        "attentions.0.transformer_blocks.0.attn1.processor.to_q_lora": [(320, 320), (640, 640), (1280, 1280)],
        "attentions.0.transformer_blocks.0.attn1.processor.to_out_lora": [(320, 320), (640, 640), (1280, 1280)],
        "attentions.0.transformer_blocks.0.attn2.processor.to_k_lora": [(768, 320), (768, 640), (768, 1280)],
        "attentions.0.transformer_blocks.0.attn2.processor.to_v_lora": [(768, 320), (768, 640), (768, 1280)],
        "attentions.0.transformer_blocks.0.attn2.processor.to_q_lora": [(320, 320), (640, 640), (1280, 1280)],
        "attentions.0.transformer_blocks.0.attn2.processor.to_out_lora": [(320, 320), (640, 640), (1280, 1280)],
        "attentions.0.transformer_blocks.0.ff.net.0.proj.lora": [(320, 2560), (640, 5120), (1280, 10240)],
        "attentions.0.transformer_blocks.0.ff.net.2.lora": [(1280, 320), (2560, 640), (5120, 1280)],
    }
    patterns = []
    for i in range(3):
        for suffix, dims in down_block_dict.items():
            patterns.append({
                "name": f"unet.down_blocks.{i}.{suffix}",
                "dim": dims[i],
                "type": f"{suffix}_{i}",
            })
    for suffix, dims in mid_block_dict.items():
        patterns.append({
            "name": f"unet.mid_block.{suffix}",
            "dim": dims[2],
            "type": f"{suffix}_2",
        })
    for i in range(3):
        dim_idx = 2 - i
        for suffix, dims in down_block_dict.items():
            patterns.append({
                "name": f"unet.up_blocks.{i + 1}.{suffix}",
                "dim": dims[dim_idx],
                "type": f"{suffix}_{dim_idx}",
            })
    return patterns


class LoRAEmbedder(nn.Module):
    def __init__(self, lora_patterns=None, L=1, out_dim=768, num_probes=4, block_type="block"):
        super().__init__()
        self.lora_patterns = default_lora_patterns() if lora_patterns is None else lora_patterns
        self.L = L
        self.num_probes = num_probes
        block_cls = {"block": LoRALayerBlock, "block2": LoRALayerBlock2}[block_type]
        self.model_dict = nn.ModuleDict()
        self.proj_dict = nn.ModuleDict()
        for pattern in self.lora_patterns:
            name = pattern["name"]
            layer_type = pattern["type"]
            self.model_dict[name.replace(".", "___")] = block_cls(L, pattern["dim"][0], num_probes)
            key = layer_type.replace(".", "___")
            if key not in self.proj_dict:
                self.proj_dict[key] = nn.Linear(pattern["dim"][1], out_dim)

    def forward(self, lora):
        tokens = []
        for pattern in self.lora_patterns:
            name = pattern["name"]
            layer_type = pattern["type"]
            model_key = name.replace(".", "___")
            proj_key = layer_type.replace(".", "___")
            key_a = f"{name}.down.weight"
            key_b = f"{name}.up.weight"
            if key_a in lora and key_b in lora:
                token = self.model_dict[model_key](lora[key_a], lora[key_b])
                token = self.proj_dict[proj_key](token)
            else:
                proj = self.proj_dict[proj_key]
                token = torch.zeros(
                    1,
                    self.L,
                    proj.out_features,
                    device=proj.weight.device,
                    dtype=proj.weight.dtype,
                )
            tokens.append(token)
        return torch.cat(tokens, dim=1)


class LoRAEncoder(nn.Module):
    def __init__(
        self,
        embed_dim=768,
        max_position_embeddings=150,
        num_encoder_layers=4,
        encoder_intermediate_size=2560,
        L=1,
        num_probes=4,
        block_type="block",
        head_mode="single",
        pooling="cls",
        out_dim=None,
        text_out_dim=None,
        img_out_dim=None,
    ):
        super().__init__()
        max_position_embeddings *= L
        self.head_mode = head_mode
        self.pooling = pooling
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.embedder = LoRAEmbedder(L=L, out_dim=embed_dim, num_probes=num_probes, block_type=block_type)
        self.position_embeds = nn.Parameter(torch.zeros(1, max_position_embeddings * L, embed_dim))
        self.encoders = nn.ModuleList([
            CLIPEncoderLayer(embed_dim, encoder_intermediate_size)
            for _ in range(num_encoder_layers)
        ])
        self.final_layer_norm = nn.LayerNorm(embed_dim)
        if pooling == "cls_mean":
            self.pool_proj = nn.Sequential(
                nn.LayerNorm(embed_dim * 2),
                nn.Linear(embed_dim * 2, embed_dim),
            )
        elif pooling == "cls":
            self.pool_proj = nn.Identity()
        else:
            raise ValueError(f"unsupported pooling: {pooling}")
        if head_mode == "single":
            final_out_dim = embed_dim if out_dim is None else out_dim
            self.out_proj = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, final_out_dim),
            )
        elif head_mode == "dual":
            text_out_dim = embed_dim if text_out_dim is None else text_out_dim
            img_out_dim = embed_dim if img_out_dim is None else img_out_dim
            self.text_head = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, text_out_dim),
            )
            self.img_head = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, img_out_dim),
            )
        else:
            raise ValueError(f"unsupported head_mode: {head_mode}")

    def encode_tokens(self, lora):
        embeds = self.embedder(lora)
        if embeds.size(1) > self.position_embeds.size(1):
            raise ValueError(f"LoRA token count {embeds.size(1)} exceeds position capacity {self.position_embeds.size(1)}")
        embeds = embeds + self.position_embeds[:, :embeds.size(1)]
        cls_token = self.cls_token.expand(embeds.size(0), -1, -1)
        embeds = torch.cat([cls_token, embeds], dim=1)
        for encoder in self.encoders:
            embeds = encoder(embeds)
        return self.final_layer_norm(embeds)

    def pool_features(self, embeds):
        if self.pooling == "cls":
            return embeds[:, 0]
        cls_feature = embeds[:, 0]
        mean_feature = embeds[:, 1:].mean(dim=1)
        return self.pool_proj(torch.cat([cls_feature, mean_feature], dim=-1))

    def encode_trunk(self, lora):
        return self.pool_features(self.encode_tokens(lora))

    def forward(self, lora):
        hidden = self.encode_trunk(lora)
        if self.head_mode == "single":
            return self.out_proj(hidden)
        return self.text_head(hidden), self.img_head(hidden)
