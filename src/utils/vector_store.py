"""
简单的向量检索工具
====================

支持两种模式：
1. 基于 sentence-transformers 的稠密向量
2. 基于 TF-IDF 的稀疏检索（无依赖 fallback）

用于 RAG：把知识图谱节点描述、典型错误等文本向量化，检索相关知识。
"""
from __future__ import annotations
from typing import List, Dict, Optional, Tuple
import numpy as np
import re
import os


class SimpleVectorStore:
    """
    轻量级向量库：基于 TF-IDF + 余弦相似度。
    无外部依赖，适合冷启动与离线场景。
    """
    def __init__(self):
        self.docs: List[str] = []
        self.vectors: Optional[np.ndarray] = None
        self.vocab: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """支持中英文的简单分词"""
        # 英文按空格，中文按字符
        text = text.lower().strip()
        # 简单的 chinese split by char
        tokens = []
        for token in re.split(r"[\s,.!?;:()\[\]{}\"\']+", text):
            if not token:
                continue
            # 中文字符单独成 token
            sub_tokens = re.findall(r"[\u4e00-\u9fff]|[a-z0-9]+", token)
            tokens.extend(sub_tokens)
        return tokens

    def fit(self, docs: List[str]):
        """建立索引"""
        self.docs = docs
        # 词表
        tokenized = [self.tokenize(d) for d in docs]
        all_tokens = set(t for ts in tokenized for t in ts)
        self.vocab = {t: i for i, t in enumerate(sorted(all_tokens))}
        # IDF
        N = len(docs)
        df = {t: 0 for t in self.vocab}
        for ts in tokenized:
            for t in set(ts):
                df[t] += 1
        self.idf = {t: np.log(N / (df[t] + 1)) for t in self.vocab}
        # TF-IDF 向量
        vectors = np.zeros((N, len(self.vocab)))
        for i, ts in enumerate(tokenized):
            tf = {t: ts.count(t) for t in ts}
            for t, c in tf.items():
                vectors[i, self.vocab[t]] = (1 + np.log(c)) * self.idf.get(t, 0)
        # 归一化
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms < 1e-9, 1, norms)
        self.vectors = vectors / norms

    def query(self, text: str, top_k: int = 5) -> List[Tuple[int, float, str]]:
        """检索 top_k 相关文档"""
        if self.vectors is None:
            return []
        tokens = self.tokenize(text)
        vec = np.zeros(len(self.vocab))
        tf = {t: tokens.count(t) for t in tokens}
        for t, c in tf.items():
            if t in self.vocab:
                vec[self.vocab[t]] = (1 + np.log(c)) * self.idf.get(t, 0)
        norm = np.linalg.norm(vec)
        if norm > 1e-9:
            vec = vec / norm
        scores = self.vectors @ vec
        top_idx = np.argsort(-scores)[:top_k]
        return [(int(i), float(scores[i]), self.docs[i]) for i in top_idx if scores[i] > 0]


# ---------------------------------------------------------------------- #
# Sentence-Transformers 稠密向量（生产推荐）
# ---------------------------------------------------------------------- #
class DenseVectorStore:
    """基于 sentence-transformers 的稠密向量库"""
    def __init__(self, model_name: str = "shibing624/text2vec-base-chinese"):
        self.model_name = model_name
        self.model = None
        self.docs: List[str] = []
        self.vectors: Optional[np.ndarray] = None

    def _ensure_model(self):
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(self.model_name)
            except ImportError:
                raise ImportError("Please install sentence-transformers: pip install sentence-transformers")

    def fit(self, docs: List[str]):
        self._ensure_model()
        self.docs = docs
        self.vectors = self.model.encode(docs, normalize_embeddings=True)

    def query(self, text: str, top_k: int = 5) -> List[Tuple[int, float, str]]:
        if self.vectors is None:
            return []
        q = self.model.encode([text], normalize_embeddings=True)[0]
        scores = self.vectors @ q
        top_idx = np.argsort(-scores)[:top_k]
        return [(int(i), float(scores[i]), self.docs[i]) for i in top_idx]


def get_vector_store(mode: str = "simple", **kwargs):
    if mode == "dense":
        return DenseVectorStore(**kwargs)
    return SimpleVectorStore()


if __name__ == "__main__":
    vs = get_vector_store("simple")
    docs = [
        "线性回归是最简单的机器学习算法",
        "卷积神经网络用于图像识别",
        "Transformer 用自注意力机制处理序列",
        "支持向量机通过最大间隔分类",
    ]
    vs.fit(docs)
    results = vs.query("怎么处理图片？", top_k=2)
    for idx, score, doc in results:
        print(f"  [{score:.3f}] {doc}")
