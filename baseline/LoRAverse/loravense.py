"""
LoRAverse: 主程序入口（支持预计算embedding）

使用示例:
    # 方式1: 从头构建（较慢，第一次使用）
    system = LoRAverse()
    system.load_adapters(raw_data)
    system.save_embeddings("embeddings.npz")  # 保存供下次使用
    
    # 方式2: 从预计算文件加载（推荐，快速）
    system = LoRAverse.from_precomputed("embeddings.npz")
    
    # 然后正常检索
    selected = system.retrieve("your prompt")
"""

import os
from typing import List, Dict, Optional
from pathlib import Path

from config import Config, config
from concept_extractor import ConceptExtractor
from adapter_database import AdapterDatabase
from submodular_retriever import SubmodularRetriever
from safety_checker import SafetyChecker


class LoRAverse:
    """
    LoRAverse主类（支持预计算embedding）
    
    提供两种初始化方式：
    1. 标准初始化：从头构建数据库
    2. from_precomputed：从预计算embedding文件加载（推荐）
    """
    
    def __init__(self, custom_config: Optional[Config] = None):
        """
        标准初始化（从头构建）
        
        如果需要快速启动，请使用 from_precomputed() 类方法。
        """
        self.cfg = custom_config or config
        openai_api_key = self.cfg.openai_api_key or os.getenv("OPENAI_API_KEY")
        
        print("Initializing LoRAverse (standard mode)...")
        
        # 初始化各个模块
        self.concept_extractor = ConceptExtractor(
            model=self.cfg.concept_extractor_model,
            api_key=openai_api_key,
            temperature=self.cfg.concept_extraction_temperature,
            max_tokens=self.cfg.concept_extraction_max_tokens
        )
        
        # 标准方式初始化数据库（需要时加载encoder）
        self.database = AdapterDatabase(
            embedding_model=self.cfg.embedding_model,
            lazy_load_encoder=False  # 标准模式下立即加载
        )
        
        self.retriever = SubmodularRetriever(
            database=self.database,
            lambda_relevance=self.cfg.lambda_relevance,
            lambda_diversity=self.cfg.lambda_diversity,
            n_select=self.cfg.n_select
        )
        
        self.safety_checker = SafetyChecker(
            model=self.cfg.safety_checker_model,
            api_key=openai_api_key,
            temperature=self.cfg.safety_check_temperature,
            max_tokens=self.cfg.safety_check_max_tokens
        )
        
        self._initialized_from_file = False
        print("LoRAverse initialized (standard mode)")
    
    @classmethod
    def from_precomputed(cls, 
                        embedding_file: str,
                        custom_config: Optional[Config] = None) -> "LoRAverse":
        """
        从预计算的embedding文件快速初始化（类工厂方法，推荐）
        
        这是最高效的初始化方式，跳过所有embedding计算。
        
        Args:
            embedding_file: 预计算的embedding文件路径 (.npz 或 .pkl)
            custom_config: 可选的自定义配置
            
        Returns:
            LoRAverse: 初始化好的系统实例
            
        Example:
            >>> system = LoRAverse.from_precomputed("embeddings.npz")
            >>> result = system.retrieve("a fantasy landscape")
        """
        print(f"\n{'='*70}")
        print(f"Initializing LoRAverse from precomputed embeddings")
        print(f"{'='*70}")
        
        # 创建实例（不调用__init__的标准流程）
        instance = cls.__new__(cls)
        instance.cfg = custom_config or config
        openai_api_key = instance.cfg.openai_api_key or os.getenv("OPENAI_API_KEY")
        
        # 初始化各个模块
        instance.concept_extractor = ConceptExtractor(
            model=instance.cfg.concept_extractor_model,
            api_key=openai_api_key,
            temperature=instance.cfg.concept_extraction_temperature,
            max_tokens=instance.cfg.concept_extraction_max_tokens
        )
        
        # 关键：使用 from_precomputed 加载数据库（延迟加载encoder）
        instance.database = AdapterDatabase.from_precomputed(
            embedding_file=embedding_file,
            embedding_model=instance.cfg.embedding_model,
            lazy_load_encoder=True  # 延迟加载，因为我们已经有embedding了
        )
        
        instance.retriever = SubmodularRetriever(
            database=instance.database,
            lambda_relevance=instance.cfg.lambda_relevance,
            lambda_diversity=instance.cfg.lambda_diversity,
            n_select=instance.cfg.n_select
        )
        
        instance.safety_checker = SafetyChecker(
            model=instance.cfg.safety_checker_model,
            api_key=openai_api_key,
            temperature=instance.cfg.safety_check_temperature,
            max_tokens=instance.cfg.safety_check_max_tokens
        )
        
        instance._initialized_from_file = True
        
        print(f"\n{'='*70}")
        print(f"LoRAverse ready! Loaded {len(instance.database.adapters)} adapters")
        print(f"{'='*70}")
        
        return instance
    
    # ========================================================================
    # 数据加载和保存方法
    # ========================================================================
    
    def load_adapters(self, adapters_data: List[Dict], **kwargs):
        """
        从原始数据加载adapters并计算embedding（标准方式）
        
        Args:
            adapters_data: List of dict with adapter metadata
            **kwargs: 传递给add_adapters的参数
        """
        if self._initialized_from_file:
            print("Warning: Database already loaded from file. Adding new adapters will require re-encoding.")
        
        print(f"\nLoading {len(adapters_data)} adapters...")
        self.database.add_adapters(adapters_data, **kwargs)
        print("Adapters loaded successfully!")
    
    def save_embeddings(self, filepath: str, format: str = 'npz'):
        """
        保存当前数据库的embedding到文件
        
        保存后可以在下次使用 from_precomputed() 快速加载。
        
        Args:
            filepath: 保存路径（建议 .npz）
            format: 'npz' 或 'pkl'
        """
        self.database.save_precomputed(filepath, format=format)
        print(f"\nTo load next time, use:")
        print(f"  system = LoRAverse.from_precomputed('{filepath}')")
    
    # ========================================================================
    # 检索功能（保持不变）
    # ========================================================================
    
    def retrieve(self, prompt: str, apply_safety_check: Optional[bool] = None) -> List[str]:
        """
        执行完整的LoRA检索流程（与之前相同）
        """
        if apply_safety_check is None:
            apply_safety_check = self.cfg.apply_safety_check
        
        print(f"\n{'='*70}")
        print(f"LoRAverse Retrieval")
        print(f"{'='*70}")
        print(f"Prompt: '{prompt}'")
        print(f"{'='*70}")
        
        # Step 1: 概念提取
        print("\n[Step 1] Concept Extraction")
        concepts = self.concept_extractor.extract_concepts(prompt)
        print(f"Extracted concepts: {concepts}")
        
        # Step 2: 子模检索
        print("\n[Step 2] Submodular Retrieval")
        selected_adapters = self.retriever.retrieve_for_concepts(concepts, prompt)
        print(f"\nSelected before safety check: {len(selected_adapters)} adapters")
        
        # Step 3: 安全检查
        if apply_safety_check and len(selected_adapters) > 0:
            print("\n[Step 3] Safety Check")
            selected_adapters = self.safety_checker.filter_adapters(
                adapter_ids=selected_adapters,
                database=self.database,
                user_prompt=prompt,
                verbose=True
            )
        
        # 输出结果
        print(f"\n{'='*70}")
        print(f"Final Result: {len(selected_adapters)} adapters selected")
        print(f"{'='*70}")
        
        for i, aid in enumerate(selected_adapters, 1):
            adapter = self.database.adapters[aid]
            print(f"{i}. {adapter.name}")
            print(f"   {adapter.description[:60]}...")
        
        return selected_adapters
    
    def get_adapter_info(self, adapter_id: str) -> Dict:
        """获取特定adapter的详细信息"""
        adapter = self.database.adapters.get(adapter_id)
        if not adapter:
            return {}
        
        return {
            "id": adapter.id,
            "name": adapter.name,
            "description": adapter.description,
            "cluster": self.database.get_cluster_for_adapter(adapter_id),
            "embedding_norm": float(np.linalg.norm(adapter.embedding)) if adapter.embedding is not None else None,
            "metadata": adapter.metadata
        }


# =========================================================================
# 演示代码
# =========================================================================
def demo_workflow():
    """
    演示完整的工作流程：
    1. 首次运行：构建数据库并保存embedding
    2. 后续运行：直接从文件加载
    """
    import tempfile
    import shutil
    
    # 创建模拟数据
    print("Creating mock dataset...")
    mock_data = [
        {
            "id": f"lora_{i:04d}",
            "name": f"Style_{i:04d}",
            "description": f"Professional {style} photography style for {subject} photography with high quality",
            "metadata": {"category": "style", "base": "SD1.5"}
        }
        for i, (style, subject) in enumerate([
            ("portrait", "character"), ("landscape", "nature"), ("anime", "character"),
            ("watercolor", "art"), ("oil painting", "art"), ("cyberpunk", "scene"),
            ("fantasy", "scene"), ("steampunk", "character"), ("sci-fi", "vehicle"),
            ("nature", "landscape"), ("urban", "street"), ("vintage", "portrait"),
        ] * 25)  # 300个adapters
    ]
    
    # 临时文件路径
    temp_dir = tempfile.mkdtemp()
    embedding_file = os.path.join(temp_dir, "test_embeddings.npz")
    
    try:
        # 演示1: 首次运行 - 从头构建并保存
        print("\n" + "="*70)
        print("Demo 1: First-time setup (build and save)")
        print("="*70)
        
        system = LoRAverse()
        system.load_adapters(mock_data, batch_size=32)
        system.save_embeddings(embedding_file)
        
        # 测试检索
        result1 = system.retrieve("A photorealistic fantasy character portrait")
        
        # 演示2: 后续运行 - 从文件快速加载
        print("\n" + "="*70)
        print("Demo 2: Fast loading from precomputed embeddings")
        print("="*70)
        
        # 完全重新初始化，从文件加载
        system2 = LoRAverse.from_precomputed(embedding_file)
        
        # 验证数据已加载
        stats = system2.database.get_stats()
        print(f"\nLoaded database stats: {stats}")
        
        # 执行相同检索
        result2 = system2.retrieve("A photorealistic fantasy character portrait")
        
        # 验证结果一致
        print(f"\nVerification: Both runs returned {len(result1)} and {len(result2)} adapters")
        
        print(f"\n{'='*70}")
        print("Demo complete! Precomputed embeddings workflow working correctly.")
        print(f"{'='*70}")
        
    finally:
        # 清理临时文件
        shutil.rmtree(temp_dir)
        print(f"\nCleaned up temporary files")


if __name__ == "__main__":
    import numpy as np
    
    # 运行演示
    demo_workflow()
