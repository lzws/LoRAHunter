import torch
import torch.nn as nn
from safetensors.torch import load_file
from diffusers.loaders import StableDiffusionLoraLoaderMixin
from models import CLIPEncoderLayer
# SD 版本的 LoRA encoder
#


# class LoRALayerBlock(torch.nn.Module):
#     def __init__(self, L, dim_in, full_name=''):
#         super().__init__()
#         self.x = torch.nn.Parameter(torch.randn(1, L, dim_in))
#         self.full_name = full_name

#     def forward(self, lora_A, lora_B):
#         if lora_A.dim() == 4:
#             lora_A = lora_A[:, :, 0, 0]
#             lora_B = lora_B[:, :, 0, 0]
#         out = self.x @ lora_A.T @ lora_B.T
#         return out

class LoRALayerBlock(torch.nn.Module):
    def __init__(self, L, dim_in, num_probes=4, full_name=''):
        super().__init__()
        self.x = torch.nn.Parameter(torch.randn(num_probes, dim_in) * 0.02)
        self.out_proj = torch.nn.Linear(num_probes, L)
        self.full_name = full_name

    def forward(self, lora_A, lora_B):
        if lora_A.dim() == 4:
            lora_A = lora_A[:, :, 0, 0]
            lora_B = lora_B[:, :, 0, 0]

        # [P, d_in] @ [d_in, r] -> [P, r]
        # [P, r] @ [r, d_out] -> [P, d_out]
        x = self.x @ lora_A.T @ lora_B.T   # [P, d_out]

        # project probe dimension P -> L
        x = x.transpose(0, 1)              # [d_out, P]
        x = self.out_proj(x)               # [d_out, L]
        out = x.transpose(0, 1).unsqueeze(0)   # [1, L, d_out]

        return out

class LoRALayerBlock2(torch.nn.Module):
    def __init__(self, L, dim_in, num_probes=4, full_name=''):
        super().__init__()
        self.x = torch.nn.Parameter(torch.randn(num_probes, dim_in) * 0.02)
        self.out_proj = torch.nn.Linear(num_probes, L)
        self.full_name = full_name

        self.pre_norm = nn.LayerNorm(num_probes)
        self.mlp = nn.Sequential(
            nn.Linear(num_probes, num_probes * 4),
            nn.GELU(),
            nn.Linear(num_probes * 4, num_probes),
        )


    def forward(self, lora_A, lora_B):
        if lora_A.dim() == 4:
            lora_A = lora_A[:, :, 0, 0]
            lora_B = lora_B[:, :, 0, 0]

        # [P, d_in] @ [d_in, r] -> [P, r]
        # [P, r] @ [r, d_out] -> [P, d_out]
        x = self.x @ lora_A.T @ lora_B.T   # [P, d_out]

        # project probe dimension P -> L
        x = x.transpose(0, 1)              # [d_out, P]
        x = x + self.mlp(self.pre_norm(x))
        x = self.out_proj(x)               # [d_out, L]
        out = x.transpose(0, 1).unsqueeze(0)   # [1, L, d_out]

        return out


block_map = {
    "block": LoRALayerBlock,
    "block2": LoRALayerBlock2
}

class LoRAEmbedder(torch.nn.Module):
    def __init__(self, lora_patterns=None, L=1, out_dim=2048, num_probes=4,block_type="block"):
        super().__init__()
        if lora_patterns is None:
            lora_patterns = self.default_lora_patterns()
        self.L = L
        self.num_probes = num_probes
        self.block_type = block_type
            
        model_dict = {}
        for lora_pattern in lora_patterns:
            name, dim = lora_pattern["name"], lora_pattern["dim"][0]
            model_dict[name.replace(".", "___")] = block_map[block_type](L, dim, num_probes,name)
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


        down_block_dict = {
            "attentions.0.proj_in.lora": [(320, 320),(640, 640), (1280, 1280)], "attentions.0.proj_out.lora": [(320, 320),(640, 640), (1280, 1280)],
            "attentions.0.transformer_blocks.0.attn1.processor.to_k_lora":[(320, 320),(640, 640),(1280, 1280)], "attentions.0.transformer_blocks.0.attn1.processor.to_v_lora":[(320, 320),(640, 640), (1280, 1280)],
            "attentions.0.transformer_blocks.0.attn1.processor.to_q_lora":[(320, 320),(640, 640),(1280, 1280)], "attentions.0.transformer_blocks.0.attn1.processor.to_out_lora":[(320, 320),(640, 640), (1280, 1280)],
            "attentions.0.transformer_blocks.0.attn2.processor.to_k_lora":[(768, 320),(768, 640),(768, 1280)], "attentions.0.transformer_blocks.0.attn2.processor.to_v_lora":[(768, 320),(768, 640),(768, 1280)],
            "attentions.0.transformer_blocks.0.attn2.processor.to_q_lora":[(320, 320),(640, 640),(1280, 1280)], "attentions.0.transformer_blocks.0.attn2.processor.to_out_lora":[(320, 320),(640, 640), (1280, 1280)],
            "attentions.0.transformer_blocks.0.ff.net.0.proj.lora":[(320, 2560),(640, 5120),(1280,10240)], "attentions.0.transformer_blocks.0.ff.net.2.lora":[(1280, 320),(2560, 640),(5120,1280)],
            "attentions.1.proj_in.lora": [(320, 320),(640, 640), (1280, 1280)], "attentions.1.proj_in.lora": [(320, 320),(640, 640), (1280, 1280)],
            "attentions.1.transformer_blocks.0.attn1.processor.to_k_lora":[(320, 320),(640, 640),(1280, 1280)], "attentions.1.transformer_blocks.0.attn1.processor.to_v_lora":[(320, 320),(640, 640),(1280, 1280)],
            "attentions.1.transformer_blocks.0.attn1.processor.to_q_lora":[(320, 320),(640, 640),(1280, 1280)], "attentions.1.transformer_blocks.0.attn1.processor.to_out_lora":[(320, 320),(640, 640),(1280, 1280)],
            "attentions.1.transformer_blocks.0.attn2.processor.to_k_lora":[(768, 320),(768, 640),(768, 1280)], "attentions.1.transformer_blocks.0.attn2.processor.to_v_lora":[(768, 320),(768, 640),(768, 1280)],
            "attentions.1.transformer_blocks.0.attn2.processor.to_q_lora":[(320, 320),(640, 640),(1280, 1280)], "attentions.1.transformer_blocks.0.attn2.processor.to_out_lora":[(320, 320),(640, 640),(1280, 1280)],
            "attentions.1.transformer_blocks.0.ff.net.0.proj.lora":[(320, 2560),(640, 5120),(1280,10240)], "attentions.1.transformer_blocks.0.ff.net.2.lora":[(1280, 320),(2560, 640),(5120,1280)],
        }
        for i in range(3):
            for suffix in down_block_dict:
                lora_patterns.append({
                    "name": f"unet.down_blocks.{i}.{suffix}",
                    "dim": down_block_dict[suffix][i],
                    "type": suffix+'_'+str(i),
                })
        
        mid_block_dict = {
            "attentions.0.proj_in.lora": [(320, 320),(640, 640), (1280, 1280)], "attentions.0.proj_out.lora": [(320, 320),(640, 640), (1280, 1280)],
            "attentions.0.transformer_blocks.0.attn1.processor.to_k_lora":[(320, 320),(640, 640),(1280, 1280)], "attentions.0.transformer_blocks.0.attn1.processor.to_v_lora":[(320, 320),(640, 640), (1280, 1280)],
            "attentions.0.transformer_blocks.0.attn1.processor.to_q_lora":[(320, 320),(640, 640),(1280, 1280)], "attentions.0.transformer_blocks.0.attn1.processor.to_out_lora":[(320, 320),(640, 640), (1280, 1280)],
            "attentions.0.transformer_blocks.0.attn2.processor.to_k_lora":[(768, 320),(768, 640),(768, 1280)], "attentions.0.transformer_blocks.0.attn2.processor.to_v_lora":[(768, 320),(768, 640),(768, 1280)],
            "attentions.0.transformer_blocks.0.attn2.processor.to_q_lora":[(320, 320),(640, 640),(1280, 1280)], "attentions.0.transformer_blocks.0.attn2.processor.to_out_lora":[(320, 320),(640, 640), (1280, 1280)],
            "attentions.0.transformer_blocks.0.ff.net.0.proj.lora":[(320, 2560),(640, 5120),(1280,10240)], "attentions.0.transformer_blocks.0.ff.net.2.lora":[(1280, 320),(2560, 640),(5120,1280)],
        }

        for i in range(1):
            for suffix in mid_block_dict:
                lora_patterns.append({
                    "name": f"unet.mid_block.{suffix}",
                    "dim": mid_block_dict[suffix][2],
                    "type": suffix + '_2',
                })
        
        for i in range(3):
            dim_idx = 3 - 1 - i
            for suffix in down_block_dict:
                lora_patterns.append({
                    "name": f"unet.up_blocks.{i+1}.{suffix}",
                    "dim": down_block_dict[suffix][dim_idx],
                    "type": suffix + '_' + str(dim_idx),
                })


        return lora_patterns
        
    def forward(self, lora):
        lora_emb = []
        for lora_pattern in self.lora_patterns:
            name, layer_type = lora_pattern["name"], lora_pattern["type"]
            name_key = name.replace(".", "___")
            type_key = layer_type.replace(".", "___")

            A_key = name + ".down.weight"
            B_key = name + ".up.weight"

            if A_key in lora and B_key in lora:
                lora_A = lora[A_key]
                lora_B = lora[B_key]
                lora_out = self.model_dict[name_key](lora_A, lora_B)
                lora_out = self.proj_dict[type_key](lora_out)
            else:
                # missing layer -> zero token
                proj = self.proj_dict[type_key]
                zero_token = torch.zeros(
                    1, self.L, proj.out_features,
                    device=proj.weight.device,
                    dtype=proj.weight.dtype
                )
                lora_out = zero_token

            lora_emb.append(lora_out)

        lora_emb = torch.concat(lora_emb, dim=1)
        return lora_emb


class LoRAEncoder(torch.nn.Module):
    def __init__(self, embed_dim=768, max_position_embeddings=150, num_encoder_layers=4, encoder_intermediate_size=2560, L=1, num_probes=4, 
                    block_type="block", head_mode="single", pooling="cls", out_dim=None, text_out_dim=None, img_out_dim=None):
        super().__init__()
        max_position_embeddings *= L
        self.head_mode = head_mode
        self.pooling = pooling

        self.cls_token = torch.nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        
        # Embedder
        self.embedder = LoRAEmbedder(L=L, out_dim=embed_dim, num_probes=num_probes, block_type=block_type)

        # position_embeds (This is a fixed tensor)
        self.position_embeds = torch.nn.Parameter(torch.zeros(1, max_position_embeddings * L, embed_dim))

        # encoders
        self.encoders = torch.nn.ModuleList([CLIPEncoderLayer(embed_dim, encoder_intermediate_size) for _ in range(num_encoder_layers)])

        # attn_mask
        # self.attn_mask = self.attention_mask(max_position_embeddings)

        # final_layer_norm
        self.final_layer_norm = torch.nn.LayerNorm(embed_dim)
        if pooling == "cls_mean":
            self.pool_proj = nn.Sequential(
                nn.LayerNorm(embed_dim * 2),
                nn.Linear(embed_dim * 2, embed_dim),
            )
        else:
            self.pool_proj = nn.Identity()
        
        if head_mode == "single":
            final_out_dim = out_dim if out_dim is not None else embed_dim
            self.out_proj = torch.nn.Linear(embed_dim, final_out_dim)
        elif head_mode == "dual":
            text_out_dim = text_out_dim if text_out_dim is not None else embed_dim
            img_out_dim = img_out_dim if img_out_dim is not None else embed_dim

            self.text_head = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, text_out_dim),
            )
            self.img_head = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, img_out_dim),
            )
        else:
            raise ValueError(f"head_mode {head_mode} is not supported")

    def attention_mask(self, length):
        mask = torch.empty(length, length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    def encode_tokens(self, lora):
        embeds = self.embedder(lora) 
        embeds = embeds + self.position_embeds[:, :embeds.size(1), :]
        # attn_mask = self.attn_mask.to(device=embeds.device, dtype=embeds.dtype)
        cls_ = self.cls_token.expand(embeds.size(0), -1, -1)
        embeds = torch.cat([cls_, embeds], dim=1)
        for encoder_id, encoder in enumerate(self.encoders):
            embeds = encoder(embeds, attn_mask=None)
        embeds = self.final_layer_norm(embeds)
        return embeds
    
    def pool_features(self, embeds):
        if self.pooling == 'cls':
            h = embeds[:, 0]
        elif self.pooling == "cls_mean":
            cls_feat = embeds[:, 0]
            mean_feat = embeds[:, 1:].mean(dim=1)
            h = self.pool_proj(torch.cat([cls_feat, mean_feat], dim=-1))
        else:
            raise ValueError(f"pooling {self.pooling} is not supported")
        
        return h

    def encode_trunk(self, lora):
        embeds = self.encode_tokens(lora)
        h = self.pool_features(embeds)
        return h

    def forward(self, lora):
        h = self.encode_trunk(lora)
        if self.head_mode == "single":
            return self.out_proj(h)
        elif self.head_mode == "dual":
            z_text = self.text_head(h)
            z_img = self.img_head(h)
            return z_text, z_img



def check_unet():
    from diffusers import StableDiffusionPipeline
    model_id = ""
    pipe = StableDiffusionPipeline.from_pretrained(
        "/shark/zhiwen/LoRAHunter/DiffSynth-Studio/models/AI-ModelScope/stable-diffusion-v1-5", 
        torch_dtype=torch.float16
    )
    unet = pipe.unet
    for name, module in unet.named_modules():
        print(name)



if __name__ == "__main__":

    lora_path = "/shark/zhiwen/LoRAHunter/sd_lora/sd_lora_1/civitai-lora/69/6900.safetensors"
    lora_path = "/shark/zhiwen/LoRAHunter/sd_lora/lzwecnu/civitai-lora-66k-70k/28/280276.safetensors"
    # lora_path = "/shark/zhiwen/LoRAHunter/sd_lora/sd_lora_1/civitai-lora/10/10024.safetensors"

    device = 'cuda:4'

    lora_embedder = LoRAEncoder(embed_dim=768).to(device)
    convert_dict = StableDiffusionLoraLoaderMixin.lora_state_dict(lora_path)[0]
    convert_dict = {k:v.to(device) for k, v in convert_dict.items()}
    lora_emb = lora_embedder(convert_dict)
    print(lora_emb.shape)

    # print(len(convert_dict[0]))
    # for i ,(k, v) in enumerate(convert_dict[0].items()):
    #     print(i, k, v.shape)
    #     if i >10:
    #         break
    
    # check_unet()

