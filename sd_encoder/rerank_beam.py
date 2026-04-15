import os
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple

class LoRABeamSearch:
    def __init__(
        self, 
        beam_size: int = 5,
        alpha: float = 1.0,      # Individual weight
        beta: float = 0.8,       # Compatibility weight  
        gamma: float = 0.4,      # DPP weight
        delta: float = 0.4,      # Global weight
        dpp_sigma: float = 0.5   # DPP kernel bandwidth
    ):
        self.beam_size = beam_size
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.dpp_sigma = dpp_sigma
        
    def compute_gram_matrix(self, embeddings: List[torch.Tensor]) -> torch.Tensor:
        """
        构建 DPP Gram 矩阵 K
        K_ii = 1 (质量项，可替换为 exp(individual_score))
        K_ij = exp(-||e_i - e_j||^2 / (2*sigma^2))
        """
        k = len(embeddings)
        if k == 0:
            return torch.zeros(0, 0)
        
        # Stack to [k, d]
        E = torch.stack(embeddings)  # [k, dim]
        
        # 计算距离平方 ||e_i - e_j||^2 = 2 - 2*cos_sim (假设已归一化)
        cos_sim = E @ E.T  # [k, k]
        dist_sq = 2 - 2 * cos_sim
        
        # RBF Kernel
        K = torch.exp(-dist_sq / (2 * self.dpp_sigma ** 2))
        
        # 对角线置为质量 (这里简化为1，也可设为各 concept 的匹配度)
        K.diagonal().fill_(1.0)
        return K
    
    def compute_reward(
        self, 
        selected_embs: List[torch.Tensor],
        concept_indices: List[int],
        all_concept_embs: List[torch.Tensor],
        full_emb: torch.Tensor
    ) -> float:
        """
        计算部分或完整组合的收益 R(L)
        
        Args:
            selected_embs: 已选 LoRA 的 embedding 列表，长度 k
            concept_indices: 每个已选 LoRA 对应的概念索引 (0, 1, 2...)
            all_concept_embs: 所有概念的文本 embedding 列表，长度 n
            full_emb: 完整 Prompt 的 embedding
        Returns:
            total_score: 标量分数
        """
        k = len(selected_embs)
        if k == 0:
            return 0.0
        
        device = selected_embs[0].device
        
        # 1. Individual: sum alpha * <e_l, e_c>
        ind_score = 0.0
        for emb, c_idx in zip(selected_embs, concept_indices):
            target_emb = all_concept_embs[c_idx]
            sim = torch.dot(emb, target_emb)
            ind_score += self.alpha * sim
        
        # 2. Pairwise Compatibility: sum beta * (-<e_i, e_j>)
        pair_score = 0.0
        for i in range(k):
            for j in range(i + 1, k):
                # 负余弦：越相似越冲突
                compat = -torch.dot(selected_embs[i], selected_embs[j])
                pair_score += self.beta * compat
        
        # 3. DPP Diversity: gamma * log det(K)
        dpp_score = 0.0
        if k >= 1:
            K = self.compute_gram_matrix(selected_embs)
            # 数值稳定性：加微小对角正则化
            eps = 1e-5
            sign, logdet = torch.slogdet(K + eps * torch.eye(k, device=device))
            if sign.item() > 0:  # 正定保证
                dpp_score = self.gamma * logdet
        
        # 4. Global Alignment: delta * <mean(e), e_full>
        global_score = 0.0
        mean_emb = torch.stack(selected_embs).mean(dim=0)
        global_sim = torch.dot(mean_emb, full_emb)
        global_score = self.delta * global_sim
        
        total = ind_score + pair_score + dpp_score + global_score
        return total.item()
    
    def search(
        self,
        candidate_embs: List[torch.Tensor],  # List of [m_i, dim]
        concept_embs: List[torch.Tensor],    # List of [dim], n concepts
        full_emb: torch.Tensor               # [dim]
    ) -> List[Dict]:
        """
        执行 Beam Search
        
        Args:
            candidate_embs: n 个概念的候选 embedding，每个是 [m_i, dim]
            concept_embs: n 个概念的目标 embedding
            full_emb: 完整 prompt embedding
            
        Returns:
            list of beams，每个包含：
            - indices: tuple (idx_0, idx_1, ...) 在原始 candidate 中的位置
            - embeddings: list of tensors
            - score: 总收益
        """
        n_concepts = len(candidate_embs)
        assert len(concept_embs) == n_concepts
        
        # 归一化所有输入（安全起见）
        full_emb = F.normalize(full_emb, dim=0)
        concept_embs = [F.normalize(c, dim=0) for c in concept_embs]
        candidate_embs = [F.normalize(c, dim=1) for c in candidate_embs]
        
        # 初始化：第一个概念的所有候选各自成一个 beam
        beams = []
        m0 = candidate_embs[0].shape[0]
        for j in range(m0):
            emb = candidate_embs[0][j]
            score = self.compute_reward(
                [emb], [0], concept_embs, full_emb
            )
            beams.append({
                'indices': (j,),
                'embeddings': [emb],
                'concepts': [0],
                'score': score
            })
        
        # 剪枝到 beam_size
        beams = sorted(beams, key=lambda x: x['score'], reverse=True)
        beams = beams[:self.beam_size]
        
        # 逐层扩展
        for concept_idx in range(1, n_concepts):
            new_beams = []
            m = candidate_embs[concept_idx].shape[0]
            
            # 对每个现有 beam，尝试拼接当前概念的所有候选
            for beam in beams:
                for j in range(m):
                    new_emb = candidate_embs[concept_idx][j]
                    new_embs = beam['embeddings'] + [new_emb]
                    new_concepts = beam['concepts'] + [concept_idx]
                    
                    # 计算新组合的完整收益（含 DPP 和 Global）
                    score = self.compute_reward(
                        new_embs, new_concepts, concept_embs, full_emb
                    )
                    
                    new_beams.append({
                        'indices': beam['indices'] + (j,),
                        'embeddings': new_embs,
                        'concepts': new_concepts,
                        'score': score
                    })
            
            # 剪枝
            new_beams = sorted(new_beams, key=lambda x: x['score'], reverse=True)
            beams = new_beams[:self.beam_size]
        
        return beams


# ========== 使用示例 ==========

if __name__ == "__main__":
    # 假设参数
    n_concepts = 3
    m_candidates = 30  # 每概念候选数
    dim = 768
    
    # 模拟数据（实际应来自你的 LoRA Encoder 和 CLIP）
    torch.manual_seed(42)
    candidate_embs = [
        F.normalize(torch.randn(m_candidates, dim), dim=1) 
        for _ in range(n_concepts)
    ] # List of [30, 768]
    concept_embs = [F.normalize(torch.randn(dim), dim=0) for _ in range(n_concepts)] # List of [768]
    full_emb = F.normalize(torch.randn(dim), dim=0) # [768]
    
    # 执行搜索
    selector = LoRABeamSearch(beam_size=8)
    top_beams = selector.search(candidate_embs, concept_embs, full_emb)
    
    # 输出结果
    for i, beam in enumerate(top_beams):
        print(f"Rank {i+1}: indices={beam['indices']}, score={beam['score']:.4f}")
