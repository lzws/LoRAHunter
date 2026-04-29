"""
LoRAverse 配置文件
包含所有可调整的参数和超参数
"""

from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    """
    LoRAverse 配置类
    
    所有参数均按照论文描述设置
    """
    
    # =========================================================================
    # OpenAI API 配置
    # =========================================================================
    openai_api_key: Optional[str] = None  # 从环境变量 OPENAI_API_KEY 读取
    concept_extractor_model: str = "gpt-4o-mini"  # 论文使用 gpt-4o-mini
    safety_checker_model: str = "gpt-4o"          # 论文使用 gpt-4o
    
    # =========================================================================
    # 子模优化参数 (论文第4页)
    # =========================================================================
    lambda_relevance: float = 7.0   # λ₁: 相关性权重 (论文Section 5)
    lambda_diversity: float = 1.0     # λ₂: 多样性权重 (论文Section 5)
    n_select: int = 8               # 每个concept选择的adapter数量 n (论文top-8)
    top_k_candidates: int = 200     # 召回阶段候选数量 (论文top-200)
    
    # =========================================================================
    # 聚类参数 (论文补充材料§15)
    # =========================================================================
    # UMAP 配置
    umap_n_neighbors: int = 5         # UMAP邻居数
    umap_n_components: int = 5      # UMAP降维后的维度
    umap_min_dist: float = 0.0      # UMAP最小距离
    umap_metric: str = "cosine"     # UMAP距离度量
    
    # HDBSCAN 配置
    hdbscan_min_cluster_size: int = 3   # HDBSCAN最小簇大小
    hdbscan_metric: str = "euclidean"   # HDBSCAN距离度量
    hdbscan_cluster_selection_method: str = "eom"  # HDBSCAN簇选择方法
    
    # BERTopic 配置
    bertopic_n_topics: int = 10      # BERTopic主题数量
    
    # =========================================================================
    # Embedding 配置
    # =========================================================================
    # 论文使用Vision Language Model(VLM)编码
    # 这里使用Sentence Transformer作为轻量级替代
    # 实际应用中可替换为CLIP等VLM
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # =========================================================================
    # 图像生成配置 (论文Section 5)
    # =========================================================================
    num_inference_steps: int = 35           # 去噪步数
    guidance_scale: float = 7.0               # CFG值 (论文Table 1)
    scheduler: str = "DPM-Solver++"          # 调度器
    
    # Debias prompts (论文Footnote 1)
    debias_prompt_realistic: str = "realistic, high quality"        # 用于Realistic-Vision-v6
    debias_prompt_anime: str = "anime style, high quality"         # 用于Counterfeit-v3
    
    # =========================================================================
    # 概念提取配置
    # =========================================================================
    concept_extraction_temperature: float = 0.2   # LLM温度参数，低温度确保一致性
    concept_extraction_max_tokens: int = 200
    
    # =========================================================================
    # 安全检查配置
    # =========================================================================
    safety_check_temperature: float = 0.1
    safety_check_max_tokens: int = 150
    apply_safety_check: bool = True  # 是否启用安全检查


# 全局配置实例
config = Config()
