import torch
from torch import nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, q_dim, num_heads=12, head_dim=64):
        super().__init__()
        inner_dim = num_heads * head_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.to_q = nn.Linear(q_dim, inner_dim, bias=True)
        self.to_k = nn.Linear(q_dim, inner_dim, bias=True)
        self.to_v = nn.Linear(q_dim, inner_dim, bias=True)
        self.to_out = nn.Linear(inner_dim, q_dim, bias=True)

    def forward(self, x, attn_mask=None):
        batch_size = x.shape[0]
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x = x.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.head_dim)
        return self.to_out(x)


class CLIPEncoderLayer(nn.Module):
    def __init__(self, embed_dim, intermediate_size, num_heads=12, head_dim=64):
        super().__init__()
        self.attn = Attention(embed_dim, num_heads=num_heads, head_dim=head_dim)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, embed_dim)

    def forward(self, hidden_states, attn_mask=None):
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = residual + self.attn(hidden_states, attn_mask=attn_mask)
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = hidden_states * torch.sigmoid(1.702 * hidden_states)
        hidden_states = self.fc2(hidden_states)
        return residual + hidden_states

