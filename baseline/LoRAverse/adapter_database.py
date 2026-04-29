"""
模块2: Adapter Database & Embedding (适配器数据库与嵌入)

功能: 管理LoRA模型的embedding存储、索引、检索和聚类

论文参考:
- Section 4.2: Submodular Retriever (涉及embedding和聚类的部分)
- Supplementary Material §15: Clustering implementation details
"""

import numpy as np
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from tqdm import tqdm


@dataclass
class LoRAModel:
    """
    LoRA模型数据类
    
    存储单个LoRA适配器的元数据和embedding
    
    Attributes:
        id: 唯一标识符
        name: 模型名称
        description: 模型描述/训练标签
        embedding: CLIP/VLM embedding向量 (ϕ(a_i))
        metadata: 其他元数据
    """
    id: str
    name: str
    description: str
    embedding: Optional[np.ndarray] = None
    metadata: Optional[Dict] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class AdapterDatabase:
    """
    适配器数据库
    
    管理大规模LoRA模型的存储、embedding计算、索引和聚类。
    
    关键功能:
    1. 存储LoRA模型及其embedding ϕ(a_i)
    2. 计算prompt/concept的embedding ϕ(s)
    3. 基于cosine similarity召回候选
    4. 使用BERTopic+UMAP+HDBSCAN进行语义聚类
    
    论文对应:
    - Embedding计算: ϕ(·) 函数 (论文Section 4.2)
    - 召回策略: top-K candidates via cosine similarity (论文Section 4.2)
    - 聚类: HDBSCAN + UMAP + BERTopic (论文SM §15)
    """
    
    def __init__(self,
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 umap_n_neighbors: int = 5,
                 umap_n_components: int = 5,
                 umap_min_dist: float = 0.0,
                 umap_metric: str = "cosine",
                 hdbscan_min_cluster_size: int = 3,
                 hdbscan_metric: str = "euclidean",
                 bertopic_n_topics: int = 10):
        """
        初始化适配器数据库
        
        Args:
            embedding_model: 用于计算embedding的模型
            umap_*: UMAP降维参数
            hdbscan_*: HDBSCAN聚类参数
            bertopic_n_topics: BERTopic主题数
        """
        # =========================================================================
        # Embedding编码器 (论文中的ϕ函数)
        # 论文使用Vision Language Model，这里使用SentenceTransformer作为轻量级替代
        # =========================================================================
        print(f"Loading embedding model: {embedding_model}")
        self.encoder = SentenceTransformer(embedding_model)
        
        # 数据存储
        self.adapters: Dict[str, LoRAModel] = {}  # id -> LoRAModel
        self.embeddings_matrix: Optional[np.ndarray] = None  # 所有adapter的embedding矩阵
        self.adapter_ids: List[str] = []  # 与embeddings_matrix对应的id列表
        
        # 聚类结果缓存
        self.clusters: Dict[str, int] = {}  # adapter_id -> cluster_id
        self.topic_model: Optional[BERTopic] = None  # BERTopic模型实例
        
        # =========================================================================
        # 聚类模型配置 (论文SM §15)
        # 使用BERTopic框架配合UMAP和HDBSCAN
        # =========================================================================
        
        # UMAP: 降维，保留局部结构
        self.umap_model = UMAP(
            n_neighbors=umap_n_neighbors,      # 论文: 5
            n_components=umap_n_components,      # 论文: 5
            min_dist=umap_min_dist,              # 论文: 0.0
            metric=umap_metric,                  # 论文: cosine
            random_state=42                      # 确保可复现
        )
        
        # HDBSCAN: 基于密度的层次聚类
        self.hdbscan_model = HDBSCAN(
            min_cluster_size=hdbscan_min_cluster_size,    # 论文: 3
            metric=hdbscan_metric,                       # 论文: euclidean
            cluster_selection_method='eom',
            prediction_data=True                         # 支持对新数据预测
        )
        
        self.bertopic_n_topics = bertopic_n_topics
    
    def add_adapter(self, adapter_data: Dict):
        """
        添加单个LoRA适配器 (用于增量添加)
        
        Args:
            adapter_data: Dict with keys: id, name, description, [metadata]
        """
        adapter = LoRAModel(
            id=adapter_data['id'],
            name=adapter_data['name'],
            description=adapter_data['description'],
            metadata=adapter_data.get('metadata', {})
        )
        self.adapters[adapter.id] = adapter
        self.adapter_ids.append(adapter.id)
    
    def add_adapters(self, adapters_data: List[Dict], compute_embeddings: bool = True):
        """
        批量添加LoRA适配器并计算embedding
        
        Args:
            adapters_data: List of dict with keys: id, name, description, [metadata]
            compute_embeddings: 是否立即计算embedding
        """
        print(f"Adding {len(adapters_data)} adapters to database...")
        
        # 创建LoRAModel实例
        descriptions = []
        for data in adapters_data:
            adapter = LoRAModel(
                id=data['id'],
                name=data['name'],
                description=data['description'],
                metadata=data.get('metadata', {})
            )
            self.adapters[adapter.id] = adapter
            descriptions.append(data['description'])
            self.adapter_ids.append(adapter.id)
        
        # =========================================================================
        # 计算所有adapter的embedding ϕ(a_i)
        # 这是论文中所有similarity计算的基础
        # =========================================================================
        if compute_embeddings:
            print("Computing embeddings...")
            embeddings = self.encoder.encode(
                descriptions,
                show_progress_bar=True,
                convert_to_numpy=True,
                batch_size=32
            )
            
            for i, adapter_id in enumerate(self.adapter_ids):
                self.adapters[adapter_id].embedding = embeddings[i]
            
            self.embeddings_matrix = embeddings
        
        print(f"Database now contains {len(self.adapters)} adapters")
    
    def compute_prompt_embedding(self, prompt: str) -> np.ndarray:
        """
        计算用户prompt的embedding ϕ(s)
        
        论文Section 4.2公式(2)中使用:
            F_sim(ϕ(a_i), ϕ(s))
        
        Args:
            prompt: 用户输入的完整prompt
            
        Returns:
            np.ndarray: embedding向量
        """
        return self.encoder.encode([prompt], convert_to_numpy=True)[0]
    
    def compute_concept_embedding(self, concept: str) -> np.ndarray:
        """
        计算单个concept的embedding ϕ(t)
        
        用于计算F_reward(ϕ(a_i))，即adapter embedding与concept embedding的相似度
        
        Args:
            concept: 提取的概念
            
        Returns:
            np.ndarray: embedding向量
        """
        return self.encoder.encode([concept], convert_to_numpy=True)[0]
    
    def retrieve_candidates(self, 
                           query_embedding: np.ndarray, 
                           top_k: int = 200) -> List[Tuple[str, float]]:
        """
        召回阶段: 基于cosine similarity召回top-k候选
        
        论文Section 4.2: "we first retrieve a subset relevant to the prompt using cosine similarity"
        
        Args:
            query_embedding: 查询的embedding (可以是prompt或concept)
            top_k: 召回数量，论文使用200
            
        Returns:
            List[Tuple[str, float]]: (adapter_id, similarity_score)列表
        """
        if self.embeddings_matrix is None or len(self.adapter_ids) == 0:
            raise ValueError("Database is empty. Please add adapters first.")
        
        # 计算cosine similarity
        # 公式: F_sim(ϕ(a_i), ϕ(query))
        similarities = cosine_similarity([query_embedding], self.embeddings_matrix)[0]
        
        # 获取top-k索引
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # 返回(id, similarity)列表
        return [(self.adapter_ids[i], float(similarities[i])) for i in top_indices]
    
    def build_clusters(self, candidate_ids: List[str]):
        """
        对候选adapters进行语义聚类 (论文SM §15)
        
        使用BERTopic + UMAP + HDBSCAN的组合进行聚类。
        聚类结果用于子模优化中的多样性计算。
        
        论文: "we then apply HDBSCAN on their textual embeddings via BERTopic"
        
        Args:
            candidate_ids: 候选adapter的id列表
        """
        n_candidates = len(candidate_ids)
        
        # 候选数量太少时，无法进行有效聚类
        if n_candidates < self.hdbscan_model.min_cluster_size * 2:
            print(f"Too few candidates ({n_candidates}), assigning each to its own cluster")
            for i, aid in enumerate(candidate_ids):
                self.clusters[aid] = i
            return
        
        # 提取候选adapters的embeddings和descriptions
        candidate_embeddings = []
        candidate_descriptions = []
        
        for aid in candidate_ids:
            adapter = self.adapters[aid]
            if adapter.embedding is None:
                raise ValueError(f"Adapter {aid} has no embedding")
            candidate_embeddings.append(adapter.embedding)
            candidate_descriptions.append(adapter.description)
        
        candidate_embeddings = np.array(candidate_embeddings)
        
        # 使用BERTopic进行聚类
        # BERTopic结合了UMAP降维和HDBSCAN聚类
        self.topic_model = BERTopic(
            umap_model=self.umap_model,
            hdbscan_model=self.hdbscan_model,
            embedding_model=None,  # 我们已经有embeddings了
            nr_topics=self.bertopic_n_topics,
            verbose=False
        )
        
        # 执行聚类
        topics, probs = self.topic_model.fit_transform(
            candidate_descriptions, 
            embeddings=candidate_embeddings
        )
        
        # 记录聚类结果
        # topic -1 表示离群点(outliers)
        n_clusters = len(set(topics)) - (1 if -1 in topics else 0)
        print(f"Built {n_clusters} clusters for {n_candidates} candidates")
        
        for aid, topic in zip(candidate_ids, topics):
            self.clusters[aid] = int(topic)
    
    def get_cluster_for_adapter(self, adapter_id: str) -> int:
        """
        获取adapter所属的cluster id
        
        Args:
            adapter_id: LoRA适配器ID
            
        Returns:
            int: Cluster ID (-1表示未分类或离群点)
        """
        return self.clusters.get(adapter_id, -1)
    
    def get_adapters_in_cluster(self, cluster_id: int) -> List[str]:
        """
        获取指定cluster中的所有adapter IDs
        
        Args:
            cluster_id: Cluster ID
            
        Returns:
            List[str]: Adapter ID列表
        """
        return [
            aid for aid, cid in self.clusters.items() 
            if cid == cluster_id
        ]
    
    def get_cluster_info(self) -> Dict[int, int]:
        """
        获取每个cluster中的adapter数量统计
        
        Returns:
            Dict[int, int]: cluster_id -> count
        """
        from collections import Counter
        return dict(Counter(self.clusters.values()))
    
    def save(self, filepath: str):
        """
        保存数据库到文件
        
        Args:
            filepath: 保存路径
        """
        import pickle
        
        data = {
            'adapters': self.adapters,
            'embeddings': self.embeddings_matrix,
            'adapter_ids': self.adapter_ids,
            'clusters': self.clusters
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Database saved to {filepath}")
    
    def load(self, filepath: str):
        """
        从文件加载数据库
        
        Args:
            filepath: 文件路径
        """
        import pickle
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.adapters = data['adapters']
        self.embeddings_matrix = data['embeddings']
        self.adapter_ids = data['adapter_ids']
        self.clusters = data.get('clusters', {})
        
        print(f"Database loaded from {filepath}, {len(self.adapters)} adapters")


# =========================================================================
# 单机测试代码
# =========================================================================
if __name__ == "__main__":
    # 创建模拟数据
    mock_adapters = [
        {"id": f"lora_{i:03d}", "name": f"Style_{i}", 
         "description": desc}
        for i, desc in enumerate([
            "Photorealistic portrait photography style",
            "Anime manga illustration aesthetic",
            "Watercolor painting effect soft",
            "Oil painting classical art texture",
            "Cyberpunk neon futuristic cityscape",
            "Steampunk mechanical gears brass",
            "Fantasy medieval castle landscape",
            "Sci-fi spaceship design concept",
            "Nature forest green atmospheric",
            "Urban street photography gritty",
        ] * 20)  # 模拟200个adapter
    ]
    
    # 初始化数据库
    db = AdapterDatabase()
    db.add_adapters(mock_adapters[:100])
    
    # 测试检索
    query = "A photorealistic portrait in fantasy setting"
    query_emb = db.compute_prompt_embedding(query)
    
    candidates = db.retrieve_candidates(query_emb, top_k=50)
    print(f"\nTop 5 candidates for '{query}':")
    for aid, score in candidates[:5]:
        adapter = db.adapters[aid]
        print(f"  {score:.4f} | {adapter.name}: {adapter.description[:40]}...")
    
    # 测试聚类
    candidate_ids = [aid for aid, _ in candidates]
    db.build_clusters(candidate_ids)
    
    # 查看聚类分布
    cluster_info = db.get_cluster_info()
    print(f"\nCluster distribution: {cluster_info}")
