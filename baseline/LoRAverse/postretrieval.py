import numpy as np
from collections import defaultdict
from typing import List, Dict, Set, Tuple
import json

from adapter_database_clip import AdapterDatabase  # 你的数据库类


class PostRetrievalProcessor:
    """
    后处理流程：召回后的聚类 + 子模选择
    
    输入: 每个concept的候选adapter列表 (已召回的top-200)
    输出: 最终选择的adapter IDs (去重后)
    """
    
    def __init__(self, 
                 database: AdapterDatabase,
                 lambda_relevance: float = 7.0,
                 lambda_diversity: float = 1.0,
                 n_select: int = 8):
        """
        Args:
            database: 已加载的AdapterDatabase（包含全部预计算embedding）
            lambda_relevance: λ₁ = 7.0 (论文)
            lambda_diversity: λ₂ = 1.0 (论文)
            n_select: 每个concept选择n个adapter (论文: 8)
        """
        self.db = database
        self.lambda1 = lambda_relevance
        self.lambda2 = lambda_diversity
        self.n_select = n_select
        
        print(f"Processor initialized: λ₁={lambda_relevance}, λ₂={lambda_diversity}, n={n_select}")
    
    # ========================================================================
    # 步骤1: 对每个concept的候选做聚类
    # ========================================================================
    
    def cluster_candidates_for_concept(self, 
                                       concept: str, 
                                       candidate_ids: List[str]) -> Dict[int, List[str]]:
        """
        对单个concept的候选adapter进行聚类 (论文: HDBSCAN on BERTopic)
        
        Args:
            concept: 概念文本
            candidate_ids: 该concept召回的200个adapter IDs
            
        Returns:
            Dict[int, List[str]]: cluster_id -> list of adapter_ids
        """
        print(f"\nClustering {len(candidate_ids)} candidates for concept: '{concept}'")
        
        # 提取这些候选的embeddings和descriptions
        candidate_embeddings = []
        candidate_descriptions = []
        
        for aid in candidate_ids:
            if aid not in self.db.adapters:
                print(f"Warning: {aid} not in database, skipping")
                continue
            
            adapter = self.db.adapters[aid]
            candidate_embeddings.append(adapter.embedding)
            candidate_descriptions.append(adapter.description)
        
        if len(candidate_embeddings) < 10:
            print(f"Too few candidates ({len(candidate_embeddings)}), assigning individual clusters")
            # 样本太少，每个自成一簇
            clusters = {i: [aid] for i, aid in enumerate(candidate_ids)}
            return clusters
        
        import numpy as np
        from bertopic import BERTopic
        from umap import UMAP
        from hdbscan import HDBSCAN
        
        # 论文配置: UMAP + HDBSCAN
        umap_model = UMAP(
            n_neighbors=5, n_components=5, 
            min_dist=0.0, metric='cosine', random_state=42
        )
        hdbscan_model = HDBSCAN(
            min_cluster_size=3, metric='euclidean', 
            cluster_selection_method='eom', prediction_data=True
        )
        
        # BERTopic聚类
        topic_model = BERTopic(
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            embedding_model=None,  # 已有embedding
            nr_topics=10,
            verbose=False
        )
        
        topics, _ = topic_model.fit_transform(
            candidate_descriptions,
            embeddings=np.array(candidate_embeddings)
        )
        
        # 组织成cluster_id -> adapter_ids
        clusters = defaultdict(list)
        actual_candidates = [aid for aid in candidate_ids if aid in self.db.adapters]
        
        for aid, topic_id in zip(actual_candidates, topics):
            clusters[int(topic_id)].append(aid)
            # 记录到数据库的cluster缓存（可选）
            self.db.clusters[aid] = int(topic_id)
        
        n_clusters = len([k for k in clusters.keys() if k != -1])
        print(f"  Formed {n_clusters} clusters (plus {len(clusters.get(-1, []))} outliers)")
        
        return dict(clusters)
    
    # ========================================================================
    # 步骤2: 子模优化选择 (核心)
    # ========================================================================
    
    def submodular_selection(self,
                            concept: str,
                            candidate_ids: List[str],
                            full_prompt: str) -> List[str]:
        """
        对单个concept使用贪心算法进行子模选择
        
        目标函数 (论文公式4):
            F(P) = λ₁ * Σ F_sim(ϕ(a_i), ϕ(s)) + λ₂ * Σₖ log(1 + Σ_{c_k∩P} F_reward(ϕ(a_i)))
        
        Args:
            concept: 当前concept
            candidate_ids: 该concept的200个候选IDs
            full_prompt: 完整prompt用于计算relevance
            
        Returns:
            List[str]: 选中的adapter IDs (数量≤n_select)
        """
        print(f"\nSubmodular selection for '{concept}'")
        
        # 计算embeddings
        concept_emb = self.db.compute_concept_embedding(concept)
        prompt_emb = self.db.compute_prompt_embedding(full_prompt)
        
        # Step 1: 聚类
        clusters = self.cluster_candidates_for_concept(concept, candidate_ids)
        # clusters: {cluster_id: [aid1, aid2, ...]}
        
        # Step 2: 贪心选择
        selected = set()
        cluster_sums = defaultdict(float)  # cluster_id -> sum of rewards
        
        # 辅助函数: 计算reward (与concept的相似度)
        def compute_reward(aid: str) -> float:
            adapter_emb = self.db.adapters[aid].embedding
            # Cosine similarity
            sim = np.dot(adapter_emb, concept_emb) / (
                np.linalg.norm(adapter_emb) * np.linalg.norm(concept_emb) + 1e-8
            )
            return max(0.0, float(sim))
        
        # 贪心算法 (论文Section 3)
        for iteration in range(min(self.n_select, len(candidate_ids))):
            best_aid = None
            best_gain = -float('inf')
            
            for aid in candidate_ids:
                if aid in selected or aid not in self.db.adapters:
                    continue
                
                # 边际增益 = λ₁ * relevance_gain + λ₂ * diversity_gain
                
                # 1. Relevance gain (模函数部分)
                adapter_emb = self.db.adapters[aid].embedding
                rel_sim = np.dot(adapter_emb, prompt_emb) / (
                    np.linalg.norm(adapter_emb) * np.linalg.norm(prompt_emb) + 1e-8
                )
                rel_gain = float(rel_sim)
                
                # 2. Diversity gain (子模函数部分)
                cluster_id = self.db.get_cluster_for_adapter(aid)
                reward = compute_reward(aid)
                current_sum = cluster_sums.get(cluster_id, 0.0)
                # 边际: log(1 + current_sum + reward) - log(1 + current_sum)
                div_gain = np.log(1.0 + current_sum + reward) - np.log(1.0 + current_sum)
                
                # 总增益
                total_gain = self.lambda1 * rel_gain + self.lambda2 * div_gain
                
                if total_gain > best_gain:
                    best_gain = total_gain
                    best_aid = aid
            
            if best_aid is None:
                break
            
            # 添加到选择集
            selected.add(best_aid)
            
            # 更新cluster_sums
            cid = self.db.get_cluster_for_adapter(best_aid)
            if cid != -1:
                cluster_sums[cid] += compute_reward(best_aid)
            
            # 打印进度
            if (iteration + 1) % 2 == 0 or iteration < 3:
                print(f"  Selected {iteration+1}/{self.n_select}: {best_aid[:20]}... "
                      f"(gain={best_gain:.4f})")
        
        selected_list = list(selected)
        print(f"  Final: {len(selected_list)} adapters selected")
        return selected_list
    
    # ========================================================================
    # 步骤3: 多concept合并 (论文公式6)
    # ========================================================================
    
    def process_all_concepts(self,
                              concepts: List[str],
                              candidates_map: Dict[str, List[str]],
                              full_prompt: str) -> List[str]:
        """
        处理所有concepts并合并结果
        
        论文公式6:
            R(T(s)) = {a_i | a_i ∈ P*_t_i, P*_t_i ⊆ A, t_i ∈ T(s)}
        
        Args:
            concepts: 概念列表 [t₁, t₂, ...]
            candidates_map: {concept: [candidate_ids]} 每个concept的200个候选
            full_prompt: 完整用户prompt
            
        Returns:
            List[str]: 所有选中的adapter IDs（去重）
        """
        print(f"\n{'='*70}")
        print(f"Processing {len(concepts)} concepts")
        print(f"{'='*70}")
        
        all_selected = []
        
        for i, concept in enumerate(concepts, 1):
            print(f"\n{'='*70}")
            print(f"[Concept {i}/{len(concepts)}] '{concept}'")
            print(f"{'='*70}")
            
            candidate_ids = candidates_map.get(concept, [])
            if not candidate_ids:
                print(f"Warning: No candidates for concept '{concept}'")
                continue
            
            # 对该concept进行子模选择
            selected = self.submodular_selection(concept, candidate_ids, full_prompt)
            all_selected.extend(selected)
        
        # 去重
        unique_selected = list(set(all_selected))
        
        print(f"\n{'='*70}")
        print(f"Final merged result: {len(unique_selected)} unique adapters "
              f"(from {len(all_selected)} total)")
        print(f"{'='*70}")
        
        return unique_selected


# ========================================================================
# 使用示例
# ========================================================================
def example_usage():
    """
    实际使用示例：假设你已完成概念提取和召回
    """
    
    # ========== 步骤1: 加载预计算的embedding数据库 ==========
    print("Loading database...")
    db = AdapterDatabase.load_embeddings("embeddings_clip.npz")
    
    # ========== 步骤2: 你的输入数据 ==========
    # 假设这是你已经完成的步骤的输出
    user_prompt = "a British shorthair cat playing in a cherry blossom garden"
    
    # 已提取的概念
    concepts = ["British shorthair cat", "cherry blossom garden"]
    
    # 每个concept召回的200个候选 (假设这是你的召回结果)
    # 实际中这些ID来自cosine similarity检索
    candidates_per_concept = {
        "British shorthair cat": db.adapter_ids[:200],      # 模拟你的召回结果
        "cherry blossom garden": db.adapter_ids[200:400]   # 模拟你的召回结果
    }
    
    # ========== 步骤3: 创建处理器并运行 ==========
    processor = PostRetrievalProcessor(
        database=db,
        lambda_relevance=7.0,   # λ₁
        lambda_diversity=1.0,   # λ₂
        n_select=8              # 每个concept选8个
    )
    
    # 运行完整流程
    final_adapters = processor.process_all_concepts(
        concepts=concepts,
        candidates_map=candidates_per_concept,
        full_prompt=user_prompt
    )
    
    # ========== 步骤4: 使用结果 ==========
    print(f"\nFinal selected adapters:")
    for aid in final_adapters:
        adapter = db.adapters[aid]
        print(f"  - {adapter.name}: {adapter.description[:50]}...")
    
    # 可选：安全检查
    # from safety_checker import SafetyChecker
    # checker = SafetyChecker()
    # final_adapters = checker.filter_adapters(final_adapters, db, user_prompt)


# 更简洁的API封装
def quick_select_for_concepts(
    database: AdapterDatabase,
    concepts: List[str],
    candidates_per_concept: Dict[str, List[str]],
    full_prompt: str,
    lambda1: float = 7.0,
    lambda2: float = 1.0,
    n_select: int = 8
) -> List[str]:
    """
    快速API：输入concepts和候选，返回最终选择
    
    Args:
        database: 已加载的AdapterDatabase
        concepts: 概念列表
        candidates_per_concept: {concept: [aid list]} 每个concept的200候选
        full_prompt: 原始prompt
        lambda1, lambda2, n_select: 子模参数
        
    Returns:
        List[str]: 最终unique adapter IDs
    """
    processor = PostRetrievalProcessor(database, lambda1, lambda2, n_select)
    return processor.process_all_concepts(concepts, candidates_per_concept, full_prompt)


if __name__ == "__main__":
    example_usage()
