import torch
import torch.nn as nn

from utils import *

# A/B encoder head
class FactorInputAdapter(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
    
    def forward(self, x):
        return self.net(x)

# A/B decoder head
class FactorOutputAdapter(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )
    
    def forward(self, x):
        return self.net(x)

"""
编码阶段
对每层：
    A_flat -> A_input_adapter[skey] -> a_feat
    B_flat -> B_input_adapter[skey] -> b_feat
    concat(a_feat, b_feat) -> fuse -> token
再加：
    layer index emb
    layer type emb
    block emb
然后送进 transformer
潜变量
    transformer 输出 [B, L, d_model]
    pooling 得到 [B, d_model]
    mu/logvar

解码阶段
对每层：
    z -> global decoder feat
    加 layer/type/block emb
    得到 per-layer decoder token
每层分别解码：
    A_hat
    B_hat
"""

class MultiLayerABLoRAVAE(nn.Module):
    def __init__(
        self, 
        lora_patterns,
        rank=16,
        d_model = 256,
        latent_dim = 128,
        ab_hidden_dim = 512,
        ab_token_dim = 128,
        num_encoder_layers = 4,
        num_heads = 8,
        ff_dim = 512,
        max_blocks = 60
    ):
        super().__init__()

        self.lora_patterns = lora_patterns
        self.rank = rank
        self.num_layers = len(lora_patterns)
        self.d_model = d_model
        self.latent_dim = latent_dim
        self.ab_token_dim = ab_token_dim

        # type ids
        self.layer_types = sorted(list(set([p['type'] for p in lora_patterns])))
        self.type2id = {t: i for i, t in enumerate(self.layer_types)}

        # shape groups
        self.shape2dims = {}
        for p in lora_patterns:
            in_dim, out_dim = p['dim']
            skey = shape_key(out_dim, in_dim, rank)
            self.shape2dims[skey] = {
                "out_dim": out_dim,
                "in_dim": in_dim,
                "rank": rank,
                "A_dim": rank * in_dim,
                "B_dim": out_dim * rank
            }

        # A/B specific input adapters
        self.A_input_adapters = nn.ModuleDict({
            skey: FactorInputAdapter(
                input_dim=info['A_dim'],
                hidden_dim=ab_hidden_dim,
                out_dim=ab_token_dim
            )
            for skey, info in self.shape2dims.items()
        })

        self.B_input_adapters = nn.ModuleDict({
            skey: FactorInputAdapter(
                input_dim=info['B_dim'],
                hidden_dim=ab_hidden_dim,
                out_dim=ab_token_dim
            )
            for skey, info in self.shape2dims.items()
        })

        # fuse A/B -> layer token
        self.factor_fuse = nn.Sequential(
            nn.Linear(2 * ab_token_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

        # embeddings
        self.layer_index_emb = nn.Embedding(self.num_layers, d_model)
        self.layer_type_emb = nn.Embedding(len(self.layer_types), d_model)
        self.block_emb = nn.Embedding(max_blocks, d_model)

        # transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers)

        #VAE bottleneck
        self.to_mu = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, latent_dim)
        )

        self.to_logvar = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, latent_dim)
        )
        
        self.from_latent = nn.Sequential(
            nn.Linear(latent_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

        # decoder token builder
        self.decoder_fuse = nn.Sequential(
            nn.Linear(d_model * 4, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

        # A/B specific output adapters
        self.A_output_adapters = nn.ModuleDict({
            skey: FactorOutputAdapter(
                input_dim=d_model,
                hidden_dim=ab_hidden_dim,
                out_dim=info["A_dim"]
            )
            for skey, info in self.shape2dims.items()
        })

        self.B_out_adapters = nn.ModuleDict({
            skey: FactorOutputAdapter(
                input_dim=d_model,
                hidden_dim=ab_hidden_dim,
                out_dim=info["B_dim"]
            )
            for skey, info in self.shape2dims.items()
        })

        # cache metadata
        self.pattern_shape_keys = []
        self.pattern_type_ids = []
        self.pattern_block_ids = []

        for i, p in enumerate(lora_patterns):
            in_dim, out_dim = p['dim']
            skey = shape_key(out_dim, in_dim, rank)
            self.pattern_shape_keys.append(skey)
            self.pattern_type_ids.append(self.type2id[p['type']])
            self.pattern_block_ids.append(p['block_id'])
        
        self.register_buffer(
            'pattern_type_ids_tensor',
            torch.tensor(self.pattern_type_ids, dtype=torch.long),
            persistent=False
        )
        self.register_buffer(
            "pattern_block_ids_tensor",
            torch.tensor(self.pattern_block_ids, dtype=torch.long),
            persistent=False
        )
        self.register_buffer(
            "pattern_layer_idx_tensor",
            torch.arange(self.num_layers, dtype=torch.long),
            persistent=False
        )

    def encode_layer_tokens(self, layer_inputs):
        """
        layer_inputs: list of dict, len = L
            each item:
                {
                    "A": [B, A_dim],
                    "B": [B, B_dim]
                }
        
        return:
            tokens: [B, L, d_model]
        """
        token_list = []
        Bsz = layer_inputs[0]["A"].shape[0]

        for i, item in enumerate(layer_inputs):
            skey = self.pattern_shape_keys[i]
            A_flat = item['A']
            B_flat = item['B']

            a_feat = self.A_input_adapters[skey](A_flat)  # [B, ab_token_dim]
            b_feat = self.B_input_adapters[skey](B_flat)  # [B, ab_token_dim]

            token = self.factor_fuse(torch.cat([a_feat, b_feat], dim=-1)) # [B, d_model]
            token_list.append(token)

        tokens = torch.stack(token_list, dim=1) # [B, L, d_model]
        layer_idx_emb = self.layer_index_emb(self.pattern_layer_idx_tensor).unsqueeze(0).expand(Bsz, -1, -1)
        type_emb = self.layer_type_emb(self.pattern_type_ids_tensor).unsqueeze(0).expand(Bsz, -1, -1)
        block_emb = self.block_emb(self.pattern_block_ids_tensor).unsqueeze(0).expand(Bsz, -1, -1)

        tokens = tokens + layer_idx_emb + type_emb + block_emb
        return tokens

    def encode(self, layer_inputs):
        tokens = self.encoder_layer_tokens(layer_inputs) # [B, L, d_model]
        h = self.transformer(tokens) # [B, L, d_model]
        g = h.mean(dim=1) # [B, d_model]
        mu = self.to_mu(g) # [B, latent_dim]
        logvar = self.to_logvar(g)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        Bsz = z.shape[0]
        z_feat = self.from_latent(z) # [B, d_model]

        layer_idx_emb = self.layer_index_emb(self.pattern_layer_idx_tensor).unsqueeze(0).expand(Bsz, -1, -1)
        type_emb = self.layer_type_emb(self.pattern_type_ids_tensor).unsqueeze(0).expand(Bsz, -1, -1)
        block_emb = self.block_emb(self.pattern_block_ids_tensor).unsqueeze(0).expand(Bsz, -1, -1) # [B, L, d_model]

        z_expand = z_feat.unsqueeze(1),expand(Bsz, self.num_layers, self.d_model)
        
        dec_tokens = self.decoder_fuse(
            torch.cat([z_expand, layer_idx_emb, type_emb, block_emb], dim=-1)
        ) # [B, L, d_model]

        outputs = []
        for i in range(self.num_layers):
            skey = self.pattern_shape_keys[i]
            token = dec_tokens[:, i, :]
            
            A_hat = self.A_output_adapters[skey](token)
            B_hat = self.B_output_adapters[skey](token)
            outputs.append({
                "A": A_hat,
                "B": B_hat
            })
        
        return outputs

    def forward(self, layer_inputs):
        mu, logvar, h = self.encode(layer_inputs)
        z = self.reparameterize(mu, logvar)
        recon_outputs = self.decode(z)
        return recon_outputs, mu, logvar, z, h    



