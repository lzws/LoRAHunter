import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model:int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, return_attn=False):
        B, N, D = x.shape

        q = self.q_proj(x)   # [B, N, D]
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, N, head_dim]
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, N, head_dim]
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, N, head_dim]

        # attention score
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale   # [B, H, N, N]
        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask == 0, float("-inf"))
        
        attn_weights = F.softmax(attn_scores, dim=-1)   # [B, H, N, N]
        attn_weights = self.attn_dropout(attn_weights)

        # weighted sum
        out = torch.matmul(attn_weights, v)   # [B, H, N, Dh]

        # merge heads
        out = out.transpose(1, 2).contiguous().view(B, N, D)  # [B, N, D]
        out = self.out_proj(out)
        out = self.proj_dropout(out)

        if return_attn:
            return out, attn_weights
        return out

class FeedForward(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        hidden_dim = int(d_model * mlp_ratio)

        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, hidden_dim, dropout)

    def forward(self, x, attn_mask=None, return_attn=False):
        if return_attn:
            attn_out, attn_weights = self.attn(self.norm1(x), attn_mask=attn_mask, return_attn=True)
            x = x + attn_out
            x = x + self.ffn(self.norm2(x))
            return x, attn_weights
        else:
            x = x + self.attn(self.norm1(x), attn_mask=attn_mask)
            x = x + self.ffn(self.norm2(x))
            return x

class TransformerFusionEncoder(nn.Module):
    def __init__(
        self,
        d_text: int,
        d_img: int,
        d_lora: int,
        d_model: int = 512,
        num_heads: int = 4,
        num_layers: int = 1,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
        output_dim: int = 512,
        use_type_embed: bool = True,
    ):
        super().__init__()

        # cls token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # type embeddings: CLS / TEXT / IMG / LORA
        self.use_type_embed = use_type_embed
        if use_type_embed:
            self.type_embed = nn.Parameter(torch.randn(4, d_model) * 0.02)

        self.dropout = nn.Dropout(dropout)

        # transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, output_dim)

    def forward(self, text_emb, img_emb, lora_emb, return_attn=False):
        """
        text_emb: [B, d_text]
        img_emb:  [B, d_img]
        lora_emb: [B, d_lora]
        """
        B = text_emb.shape[0]

        t = self.text_proj(self.text_norm(text_emb))   # [B, D]
        i = self.img_proj(self.img_norm(img_emb))
        l = self.lora_proj(self.lora_norm(lora_emb))

        cls = self.cls_token.expand(B, 1, -1)         # [B, 1, D]
        tokens = torch.stack([t, i, l], dim=1)        # [B, 3, D]
        x = torch.cat([cls, tokens], dim=1)           # [B, 4, D]

        if self.use_type_embed:
            x = x + self.type_embed.unsqueeze(0)

        x = self.dropout(x)

        all_attn = []

        for block in self.blocks:
            if return_attn:
                x, attn = block(x, return_attn=True)
                all_attn.append(attn)
            else:
                x = block(x)

        x = self.final_norm(x)

        fused = x[:, 0]               # CLS token
        fused = self.out_proj(fused)
        fused = F.normalize(fused, dim=-1)

        if return_attn:
            return fused, all_attn
        return fused





class CLSOnlyMultiHeadAttention(nn.Module):
    """
    One learnable CLS query attends to 3 modality tokens: text / img / lora.
    Query shape: [B, 1, D]
    Key/Value shape: [B, 3, D]
    """
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, cls_token, tokens, return_attn=False):
        """
        cls_token: [B, 1, D]
        tokens:    [B, 3, D]   # [text, img, lora]

        returns:
            out: [B, 1, D]
            attn_weights: [B, H, 1, 3] if return_attn=True
        """
        B = cls_token.size(0)

        q = self.q_proj(cls_token)   # [B, 1, D]
        k = self.k_proj(tokens)      # [B, 3, D]
        v = self.v_proj(tokens)      # [B, 3, D]

        # split heads
        q = q.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, 1, Dh]
        k = k.view(B, 3, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, 3, Dh]
        v = v.view(B, 3, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, 3, Dh]

        # attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale   # [B, H, 1, 3]
        attn_weights = F.softmax(attn_scores, dim=-1)                     # [B, H, 1, 3]
        attn_weights = self.attn_dropout(attn_weights)

        out = torch.matmul(attn_weights, v)                               # [B, H, 1, Dh]
        out = out.transpose(1, 2).contiguous().view(B, 1, self.d_model)  # [B, 1, D]
        out = self.out_proj(out)
        out = self.out_dropout(out)

        if return_attn:
            return out, attn_weights
        return out

class CLSOnlyFusionBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 4, mlp_ratio: float = 2.0, dropout: float = 0.1):
        super().__init__()
        hidden_dim = int(d_model * mlp_ratio)

        self.cls_norm = nn.LayerNorm(d_model)
        self.token_norm = nn.LayerNorm(d_model)

        self.attn = CLSOnlyMultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout
        )

        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, hidden_dim, dropout)

    def forward(self, cls_token, tokens, return_attn=False):
        """
        cls_token: [B, 1, D]
        tokens:    [B, 3, D]
        """
        if return_attn:
            attn_out, attn_weights = self.attn(
                self.cls_norm(cls_token),
                self.token_norm(tokens),
                return_attn=True
            )
            cls_token = cls_token + attn_out
            cls_token = cls_token + self.ffn(self.ffn_norm(cls_token))
            return cls_token, attn_weights
        else:
            cls_token = cls_token + self.attn(
                self.cls_norm(cls_token),
                self.token_norm(tokens)
            )
            cls_token = cls_token + self.ffn(self.ffn_norm(cls_token))
            return cls_token

class FusionEncoder(nn.Module):
    def __init__(
        self,
        d_text: int = 768,
        d_img: int = 768,
        d_lora: int = 768,
        d_model: int = 768,
        num_heads: int = 4,
        num_layers: int = 1,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
        output_dim: int = 768,
        use_type_embed: bool = True,
    ):
        super().__init__()



        # learnable cls token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # modality type embedding for text/img/lora
        self.use_type_embed = use_type_embed
        if use_type_embed:
            self.type_embed = nn.Parameter(torch.randn(3, d_model) * 0.02)

        self.input_dropout = nn.Dropout(dropout)

        # stacked cls-only fusion blocks
        self.blocks = nn.ModuleList([
            CLSOnlyFusionBlock(
                d_model=d_model,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, output_dim)

    def forward(self, text_emb, img_emb, lora_emb, return_attn=False):
        """
        text_emb: [B, d_text]
        img_emb:  [B, d_img]
        lora_emb: [B, d_lora]

        returns:
            fused_emb: [B, output_dim]
            attn_list: list of attention maps, each [B, H, 1, 3] if return_attn=True
        """
        B = text_emb.shape[0]

        # project inputs
        t = text_emb   # [B, D]
        i = img_emb      # [B, D]
        l = lora_emb   # [B, D]

        # 3 modality tokens
        tokens = torch.stack([t, i, l], dim=1)            # [B, 3, D]

        if self.use_type_embed:
            tokens = tokens + self.type_embed.unsqueeze(0)

        tokens = self.input_dropout(tokens)

        # cls token
        cls = self.cls_token.expand(B, 1, -1)             # [B, 1, D]

        attn_list = []
        for block in self.blocks:
            if return_attn:
                cls, attn = block(cls, tokens, return_attn=True)
                attn_list.append(attn)
            else:
                cls = block(cls, tokens, return_attn=False)

        fused = self.final_norm(cls[:, 0])                # [B, D]
        fused = self.out_proj(fused)                      # [B, output_dim]

        if return_attn:
            return fused, attn_list
        return fused


if __name__ == "__main__":
    fusion_encoder = FusionEncoder()
    text_emb = torch.randn(2, 768)
    img_emb = torch.randn(2, 768)
    lora_emb = torch.randn(2, 768)
    fused_emb = fusion_encoder(text_emb, img_emb, lora_emb)
    print(fused_emb.shape)