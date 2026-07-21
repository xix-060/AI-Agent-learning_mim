"""文本 Embedding 客户端"""

import os
import numpy as np
from typing import Optional
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class Embedder:
    """文本向量化客户端"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "text-embedding-v3",
    ):
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.base_url = base_url or os.getenv("LLM_BASE_URL")
        self.model = model

        if not self.api_key:
            raise ValueError("请设置 LLM_API_KEY")

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def embed(self, texts: str | list[str]) -> np.ndarray:
        """将文本转为向量"""
        if isinstance(texts, str):
            texts = [texts]
            single = True
        else:
            single = False

        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
        )

        vectors = [item.embedding for item in response.data]
        result = np.array(vectors)

        return result[0] if single else result

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float | np.ndarray:
        """计算余弦相似度"""
        if a.ndim == 1:
            a = a.reshape(1, -1)
        if b.ndim == 1:
            b = b.reshape(1, -1)

        a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
        b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)

        return (a_norm @ b_norm.T).squeeze()


def demo():
    """演示 Embedding 和相似度计算"""
    embedder = Embedder()

    sentences = [
        "猫是一种可爱的宠物",
        "狗是人类的好朋友",
        "今天天气真好",
        "小猫喜欢在阳光下睡觉",
        "机器学习是人工智能的子领域",
    ]

    print("📦 向量化中...")
    vectors = embedder.embed(sentences)
    print(f"向量形状: {vectors.shape}")

    print("\n📊 句子间相似度矩阵：")
    sim_matrix = Embedder.cosine_similarity(vectors, vectors)
    for i, s1 in enumerate(sentences):
        for j, s2 in enumerate(sentences):
            if i < j:
                sim = sim_matrix[i, j]
                print(f"  [{sim:.3f}] '{s1}' ↔ '{s2}'")

    print("\n🔍 语义检索演示：")
    query = "猫猫好可爱"
    query_vec = embedder.embed(query)
    similarities = Embedder.cosine_similarity(query_vec, vectors)

    ranked = sorted(zip(sentences, similarities), key=lambda x: x[1], reverse=True)
    for sent, sim in ranked:
        print(f"  [{sim:.3f}] {sent}")


if __name__ == "__main__":
    demo()
