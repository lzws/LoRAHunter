"""
模块3: Submodular Retriever (子模检索器)

功能: 使用贪心算法求解子模函数最大化问题，选择相关性高且多样化的LoRA子集

论文参考:
- Section 4.2: Submodular Retriever
- Section 3: Background on submodular optimization
- 公式(2)-(6): 所有子模优化相关的数学公式
"""

import numpy as np
from typing import List, Set, Dict, Tuple, Callable
from collections import defaultdict

from adapter_database import AdapterDatabase


class SubmodularRetriever:
    """
    子模检索器
    
    实现论文的核心算法：使用贪心算法求解单调子模函数最大化问题。
    
    核心数学概念:
    1. 子模性 (Submodularity): 边际增益递减性质
    2. 单调性 (Monotonicity): 添加元素不会降低函数值
    3. 贪心近似: (1 - 1/e) ≈ 0.63 的最优性保证
    
    目标函数 (论文公式4):
        F(P) = λ₁ * F_relevance(P) + λ₂ * F_diversity(P)
    
    其中:
        F_relevance(P) = Σ_{a_i∈P} F_sim(ϕ(a_i), ϕ(s))  (公式2)
        
        F_diversity(P) = Σ_k log(1 + Σ_{a_i∈C_k∩P} F_reward(ϕ(a_i)))  (公式3)
    
    F_reward(ϕ(a_i)) = cosine_similarity(ϕ(a_i), ϕ(concept)) ≥ 0
    """
    
    def __init__(self,
                 database: AdapterDatabase,
                 lambda_relevance: float = 7.0,
                 lambda_diversity: float = 1.0,
                 n_select: int = 8):
        """
        初始化子模检索器
        
        Args:
            database: AdapterDatabase实例
            lambda_relevance: λ₁, 相关性权重 (论文: 7.0)
            lambda_diversity: λ₂, 多样性权重 (论文: 1.0)
            n_select: 选择预算n，即|P|的上限 (论文: 8)
        """
        self.db = database
        self.lambda1 = lambda_relevance
        self.lambda2 = lambda_diversity
        self.n_select = n_select
        
        print(f"SubmodularRetriever initialized:")
        print(f"  λ₁ (relevance): {self.lambda1}")
        print(f"  λ₂ (diversity): {self.lambda2}")
        print(f"  n (budget): {self.n_select}")
    
    # ========================================================================
    # 子目标函数计算 (论文公式2, 3)
    # ========================================================================
    
    def _compute_reward(self, 
                      adapter_id: str, 
                      concept_embedding: np.ndarray) -> float:
        """
        计算F_reward: adapter对特定concept的reward
        
        F_reward(ϕ(a_i)) = cosine_similarity(ϕ(a_i), ϕ(concept))
        
        这是adapter对特定concept的相关性度量，用于多样性计算。
        
        Args:
            adapter_id: LoRA适配器ID
            concept_embedding: concept的embedding ϕ(concept)
            
        Returns:
            float: reward值，≥ 0
        """
        adapter = self.db.adapters[adapter_id]
        adapter_emb = adapter.embedding
        
        # Cosine similarity作为reward
        sim = np.dot(adapter_emb, concept_embedding) / (
            np.linalg.norm(adapter_emb) * np.linalg.norm(concept_embedding) + 1e-8
        )
        
        return max(0.0, float(sim))
    
    def _compute_relevance_gain(self,
                                 adapter_id: str,
                                 prompt_embedding: np.ndarray) -> float:
        """
        计算F_sim: adapter与完整prompt的相似度
        
        用于F_relevance(P) = Σ F_sim(ϕ(a_i), ϕ(s))
        
        由于F_relevance是模函数(modular)，其边际增益就是该点的值本身。
        
        Args:
            adapter_id: LoRA适配器ID
            prompt_embedding: 完整prompt的embedding ϕ(s)
            
        Returns:
            float: similarity值
        """
        adapter = self.db.adapters[adapter_id]
        adapter_emb = adapter.embedding
        
        # Cosine similarity
        sim = np.dot(adapter_emb, prompt_embedding) / (
            np.linalg.norm(adapter_emb) * np.linalg.norm(prompt_embedding) + 1e-8
        )
        
        return float(sim)
    
    def _compute_diversity_gain(self,
                                 adapter_id: str,
                                 cluster_sums: Dict[int, float],
                                 concept_embedding: np.ndarray) -> float:
        """
        计算多样性部分的边际增益 (论文公式3的边际形式)
        
        对于clustering-based diversity函数:
            F(P) = Σ_k log(1 + Σ_{a_i∈C_k∩P} F_reward(a_i))
        
        边际增益为:
            log(1 + S_k + r) - log(1 + S_k)
        
        其中:
            S_k = Σ_{a_i∈C_k∩P} F_reward(a_i)  (当前cluster已选reward之和)
            r = F_reward(新adapter)  (新adapter的reward)
        
        这个边际增益体现了子模性：随着cluster中已选adapter增多，
        再选同cluster的adapter带来的增益递减。
        
        Args:
            adapter_id: 候选adapter ID
            cluster_sums: 每个cluster当前已选adapter的F_reward之和
            concept_embedding: concept的embedding
            
        Returns:
            float: 多样性边际增益
        """
        cluster_id = self.db.get_cluster_for_adapter(adapter_id)
        
        # 未分类的adapter(-1)不贡献多样性增益
        if cluster_id == -1:
            return 0.0
        
        # 计算新adapter的reward
        reward = self._compute_reward(adapter_id, concept_embedding)
        
        # 当前cluster已选的reward之和
        current_sum = cluster_sums.get(cluster_id, 0.0)
        
        # 边际增益: log(1 + current_sum + reward) - log(1 + current_sum)
        # 证明这是子模的: 当current_sum增大时，同样reward带来的增益减小
        gain = np.log(1.0 + current_sum + reward) - np.log(1.0 + current_sum)
        
        return float(gain)
    
    def _compute_marginal_gain(self,
                                adapter_id: str,
                                selected_set: Set[str],
                                cluster_sums: Dict[int, float],
                                prompt_embedding: np.ndarray,
                                concept_embedding: np.ndarray) -> Tuple[float, float, float]:
        """
        计算总边际增益 (论文公式4的边际形式)
        
        ΔF(v|P) = λ₁ * ΔF_relevance(v|P) + λ₂ * ΔF_diversity(v|P)
        
        由于F_relevance是模函数:
            ΔF_relevance(v|P) = F_relevance({v}) = F_sim(ϕ(v), ϕ(s))
        
        F_diversity是子模函数，边际增益计算见_compute_diversity_gain。
        
        论文定理: 当F是单调子模函数时，贪心算法可达到(1-1/e)近似比。
        
        Args:
            adapter_id: 候选adapter ID
            selected_set: 已选adapter集合 P
            cluster_sums: 各cluster当前reward之和
            prompt_embedding: 完整prompt embedding
            concept_embedding: concept embedding
            
        Returns:
            Tuple[float, float, float]: (total_gain, rel_gain, div_gain)
        """
        # 相关性增益 (模函数部分)
        rel_gain = self._compute_relevance_gain(adapter_id, prompt_embedding)
        
        # 多样性增益 (子模函数部分)
        div_gain = self._compute_diversity_gain(
            adapter_id, cluster_sums, concept_embedding
        )
        
        # 加权和
        total_gain = self.lambda1 * rel_gain + self.lambda2 * div_gain
        
        return total_gain, rel_gain, div_gain
    
    # ========================================================================
    # 贪心算法实现 (论文Section 3, Background)
    # ========================================================================
    
    def _greedy_selection(self,
                         candidate_ids: List[str],
                         prompt_embedding: np.ndarray,
                         concept_embedding: np.ndarray) -> List[str]:
        """
        贪心算法求解子模最大化
        
        算法过程 (论文描述):
            P₀ = ∅
            for i = 1 to n:
                v_i = argmax_{v∈V\P_{i-1}} [F(P_{i-1} ∪ {v}) - F(P_{i-1})]
                P_i = P_{i-1} ∪ {v_i}
            return P_n
        
        理论保证: F(P_n) ≥ (1 - 1/e) * F(P_opt) ≈ 0.63 * OPT
        
        Args:
            candidate_ids: 候选adapter IDs (召回的top-200)
            prompt_embedding: 完整prompt embedding ϕ(s)
            concept_embedding: concept embedding ϕ(t)
            
        Returns:
            List[str]: 选中的adapter IDs，数量为n_select或更少
        """
        selected = []
        selected_set = set()
        
        # cluster_sums[k] = Σ_{a_i∈C_k∩P} F_reward(ϕ(a_i))
        # 记录每个cluster当前已选adapter的reward之和
        cluster_sums = defaultdict(float)
        
        n = min(self.n_select, len(candidate_ids))
        
        for iteration in range(n):
            best_adapter = None
            best_gain = -float('inf')
            best_details = None
            
            # 遍历所有未选中的candidates，找边际增益最大的
            for aid in candidate_ids:
                if aid in selected_set:
                    continue
                
                total_gain, rel_gain, div_gain = self._compute_marginal_gain(
                    aid, selected_set, cluster_sums, prompt_embedding, concept_embedding
                )
                
                if total_gain > best_gain:
                    best_gain = total_gain
                    best_adapter = aid
                    best_details = (rel_gain, div_gain)
            
            # 没有找到合适的adapter（理论上不会发生，除非candidates为空）
            if best_adapter is None:
                break
            
            # 将最佳adapter加入选择集
            selected.append(best_adapter)
            selected_set.add(best_adapter)
            
            # 更新cluster_sums
            cluster_id = self.db.get_cluster_for_adapter(best_adapter)
            if cluster_id != -1:
                reward = self._compute_reward(best_adapter, concept_embedding)
                cluster_sums[cluster_id] += reward
            
            # 打印选择信息（调试用）
            if best_details:
                rel, div = best_details
                print(f"  [{iteration+1}/{n}] Selected {best_adapter[:20]}... "
                      f"Gain={best_gain:.4f} (rel={rel:.4f}, div={div:.4f})")
        
        return selected
    
    # ========================================================================
    # 主检索接口
    # ========================================================================
    
    def select_adapters_for_concept(self,
                                     concept: str,
                                     prompt: str) -> List[str]:
        """
        为单个concept选择多样化的LoRA子集 P*_t_i
        
        对应论文公式6中的 P*_t_i ⊆ A
        
        完整流程:
        1. 计算concept和prompt的embedding
        2. 召回top-200候选（基于cosine similarity）
        3. 对候选进行聚类
        4. 使用贪心算法选择n个adapter
        
        Args:
            concept: 提取的概念 t_i
            prompt: 完整的用户prompt s
            
        Returns:
            List[str]: 选中的adapter IDs
        """
        print(f"\nProcessing concept: '{concept}'")
        
        # Step 1: 计算embeddings
        concept_embedding = self.db.compute_concept_embedding(concept)
        prompt_embedding = self.db.compute_prompt_embedding(prompt)
        
        # Step 2: 召回候选 (论文: "retrieve a subset relevant to the prompt using cosine similarity")
        candidates = self.db.retrieve_candidates(concept_embedding, top_k=200)
        candidate_ids = [aid for aid, _ in candidates]
        print(f"  Retrieved {len(candidate_ids)} candidates")
        
        # Step 3: 聚类 (论文: "apply HDBSCAN on their textual embeddings via BERTopic")
        self.db.build_clusters(candidate_ids)
        
        # Step 4: 贪心选择 (论文: "greedy algorithm can approximate the solution within a factor of 1-1/e")
        selected = self._greedy_selection(
            candidate_ids, prompt_embedding, concept_embedding
        )
        
        print(f"  Selected {len(selected)} adapters for '{concept}'")
        return selected
    
    def retrieve_for_concepts(self,
                               concepts: List[str],
                               prompt: str) -> List[str]:
        """
        为多个concepts检索适配器 (论文公式6)
        
        R(T(s)) = {a_i | a_i ∈ P*_t_i, P*_t_i ⊆ A, t_i ∈ T(s)}
        
        对每个concept分别执行子模选择，然后合并结果。
        
        Args:
            concepts: 概念列表 [t₁, t₂, ..., tₙ]
            prompt: 完整的用户prompt s
            
        Returns:
            List[str]: 所有选中的adapter IDs（去重后）
        """
        print(f"\n{'='*60}")
        print(f"Submodular Retrieval for {len(concepts)} concepts")
        print(f"{'='*60}")
        
        all_selected = set()
        
        for i, concept in enumerate(concepts, 1):
            print(f"\n[Concept {i}/{len(concepts)}]")
            adapters = self.select_adapters_for_concept(concept, prompt)
            all_selected.update(adapters)
        
        # 去重并返回
        result = list(all_selected)
        print(f"\n{'='*60}")
        print(f"Total unique adapters selected: {len(result)}")
        print(f"{'='*60}")
        
        return result
    
    def explain_selection(self,
                         adapter_id: str,
                         concept: str,
                         prompt: str,
                         selected_set: Set[str] = None) -> Dict:
        """
        解释某个adapter被选中的原因（可解释性分析）
        
        可用于调试和理解模型行为。
        
        Args:
            adapter_id: 要分析的adapter ID
            concept: 对应的concept
            prompt: 完整的prompt
            selected_set: 当前已选的adapter集合
            
        Returns:
            Dict: 包含各项得分和聚类信息的字典
        """
        concept_emb = self.db.compute_concept_embedding(concept)
        prompt_emb = self.db.compute_prompt_embedding(prompt)
        
        adapter = self.db.adapters[adapter_id]
        
        info = {
            "adapter_id": adapter_id,
            "name": adapter.name,
            "description": adapter.description,
            "concept": concept,
        }
        
        # 相关性得分
        info["relevance_score"] = self._compute_relevance_gain(adapter_id, prompt_emb)
        
        # Reward (与concept的相似度)
        info["concept_reward"] = self._compute_reward(adapter_id, concept_emb)
        
        # 聚类信息
        cluster_id = self.db.get_cluster_for_adapter(adapter_id)
        info["cluster_id"] = cluster_id
        
        # 多样性贡献
        cluster_sums = {}
        if selected_set:
            for aid in selected_set:
                cid = self.db.get_cluster_for_adapter(aid)
                if cid != -1 and cid == cluster_id:
                    cluster_sums[cid] = cluster_sums.get(cid, 0) + \
                        self._compute_reward(aid, concept_emb)
        
        info["diversity_gain"] = self._compute_diversity_gain(
            adapter_id, cluster_sums, concept_emb
        )
        
        return info


# =========================================================================
# 单机测试代码
# =========================================================================
if __name__ == "__main__":
    from adapter_database import AdapterDatabase
    
    # 创建模拟数据
    mock_adapters = [
        {"id": f"lora_{i:03d}", "name": f"Style_{i}", "description": desc}
        for i, desc in enumerate([
            "Photorealistic portrait style with high detail",
            "Anime manga illustration kawaii aesthetic",
            "Watercolor painting soft artistic effect",
            "Oil painting classical fine art texture",
            "Cyberpunk neon futuristic cityscape night",
            "Steampunk mechanical gears brass vintage",
            "Fantasy medieval castle mystical landscape",
            "Sci-fi spaceship futuristic design concept",
            "Nature forest green atmospheric foggy",
            "Urban street photography gritty documentary",
            "Portrait realistic human face detailed",
            "Cartoon cute character illustration colorful",
            "Landscape mountain scenic nature outdoor",
            "Abstract modern art geometric pattern",
            "Vintage retro style classic nostalgic",
        ] * 15)  # 225个adapter
    ]
    
    # 初始化数据库
    db = AdapterDatabase()
    db.add_adapters(mock_adapters)
    
    # 初始化检索器
    retriever = SubmodularRetriever(db, lambda_relevance=7.0, lambda_diversity=1.0, n_select=5)
    
    # 测试
    prompt = "A photorealistic portrait of a fantasy character"
    concepts = ["photorealistic portrait", "fantasy character"]
    
    selected = retriever.retrieve_for_concepts(concepts, prompt)
    
    print("\nFinal selected adapters:")
    for aid in selected:
        adapter = db.adapters[aid]
        print(f"  - {adapter.name}: {adapter.description[:50]}...")
