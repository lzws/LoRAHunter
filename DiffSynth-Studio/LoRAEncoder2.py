import torch
import torch.nn as nn
import torch.nn.functional as F


class TypeSharedLoRALayerEncoderDynamic(nn.Module):
    """
    Type-shared LoRA layer encoder with dynamic rank support.

    Input:
        A: [r, d_in]
        B: [d_out, r]

    Output:
        tokens: [num_tokens, embed_dim]
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        embed_dim: int = 768,
        hidden_dim: int = 512,
        num_tokens: int = 1,
        num_probes: int = 8,
        use_rank_feature: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_tokens = num_tokens
        self.num_probes = num_probes
        self.use_rank_feature = use_rank_feature
        self.eps = eps

        # learnable probes
        # A: [r, d_in], so probes live in d_in space
        self.A_probes = nn.Parameter(torch.randn(num_probes, d_in) * 0.02)

        # B: [d_out, r], so probes live in d_out space
        self.B_probes = nn.Parameter(torch.randn(num_probes, d_out) * 0.02)

        # stats dim:
        # A_stats(8) + B_stats(8) + interaction(4) + rank_feature(1 optional)
        stat_dim = 8 + 8 + 4
        if use_rank_feature:
            stat_dim += 1

        # probe pooled features:
        # for each of A/B:
        # response shape [P, r]
        # pool over rank dim using mean/max/std -> [P*3]
        probe_feat_dim = num_probes * 3 * 2  # A and B

        total_feat_dim = stat_dim + probe_feat_dim

        self.feat_norm = nn.LayerNorm(total_feat_dim)

        self.mlp = nn.Sequential(
            nn.Linear(total_feat_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_tokens * embed_dim)
        )

        self.token_norm = nn.LayerNorm(embed_dim)

    def _matrix_stats(self, x: torch.Tensor):
        """
        x: [M, N]
        return 8 scalar features
        """
        x = x.float()

        mean = x.mean()
        std = x.std(unbiased=False)
        mean_abs = x.abs().mean()
        fro_norm = torch.norm(x, p="fro")

        row_norms = torch.norm(x, dim=1)
        col_norms = torch.norm(x, dim=0)

        row_norm_mean = row_norms.mean()
        row_norm_std = row_norms.std(unbiased=False)
        col_norm_mean = col_norms.mean()
        col_norm_std = col_norms.std(unbiased=False)

        return torch.stack([
            mean,
            std,
            mean_abs,
            fro_norm,
            row_norm_mean,
            row_norm_std,
            col_norm_mean,
            col_norm_std
        ], dim=0)

    def _probe_pool_features(self, resp: torch.Tensor):
        """
        resp: [P, r]
        pool over rank dimension -> fixed-size feature [P*3]
        """
        mean_feat = resp.mean(dim=-1)                  # [P]
        max_feat = resp.max(dim=-1).values             # [P]
        std_feat = resp.std(dim=-1, unbiased=False)    # [P]

        feat = torch.cat([mean_feat, max_feat, std_feat], dim=0)  # [P*3]
        return feat

    def _A_probe_features(self, A: torch.Tensor):
        """
        A: [r, d_in]
        probes: [P, d_in]
        probes @ A^T => [P, r]
        """
        A = A.float()
        probes = self.A_probes.float()
        resp = probes @ A.t()   # [P, r]
        return self._probe_pool_features(resp)

    def _B_probe_features(self, B: torch.Tensor):
        """
        B: [d_out, r]
        probes: [P, d_out]
        probes @ B => [P, r]
        """
        B = B.float()
        probes = self.B_probes.float()
        resp = probes @ B       # [P, r]
        return self._probe_pool_features(resp)

    def forward(self, lora_A: torch.Tensor, lora_B: torch.Tensor):
        """
        lora_A: [r, d_in]
        lora_B: [d_out, r]

        return:
            tokens: [num_tokens, embed_dim]
        """
        assert lora_A.dim() == 2
        assert lora_B.dim() == 2
        assert lora_A.shape[1] == self.d_in
        assert lora_B.shape[0] == self.d_out
        assert lora_A.shape[0] == lora_B.shape[1]

        rank = lora_A.shape[0]

        # matrix stats
        A_stats = self._matrix_stats(lora_A)   # [8]
        B_stats = self._matrix_stats(lora_B)   # [8]

        A_fro = A_stats[3]
        B_fro = B_stats[3]

        interaction_feats = torch.stack([
            A_fro / (B_fro + self.eps),        # norm ratio
            B_fro / (A_fro + self.eps),        # inverse norm ratio
            A_fro * B_fro,                     # scale product
            A_stats[2] * B_stats[2],           # mean_abs product
        ], dim=0)                              # [4]

        feat_list = [A_stats, B_stats, interaction_feats]

        if self.use_rank_feature:
            rank_feat = torch.tensor(
                [rank / max(self.d_in, self.d_out)],
                device=lora_A.device,
                dtype=torch.float32
            )
            feat_list.append(rank_feat)

        # probe features
        A_probe_feat = self._A_probe_features(lora_A)   # [P*3]
        B_probe_feat = self._B_probe_features(lora_B)   # [P*3]

        feat_list.extend([A_probe_feat, B_probe_feat])

        feat = torch.cat(feat_list, dim=0)              # [total_feat_dim]
        feat = feat.to(self.feat_norm.weight.dtype)
        feat = self.feat_norm(feat)

        tokens = self.mlp(feat)                         # [num_tokens * embed_dim]
        tokens = tokens.view(self.num_tokens, self.embed_dim)
        tokens = self.token_norm(tokens)

        return tokens
