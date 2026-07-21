"""Naive RAG 系统骨架"""

import os
import sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from dataclasses import dataclass, field
from typing import Any, Optional
import numpy as np
from src.embedder import Embedder
from src.llm_client import LLMClient


@dataclass
class Document:
    """文档块"""

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None


class NaiveRAG:
    """最简 RAG 系统：用内存列表存向量"""

    def __init__(self, embedder: Embedder, llm: LLMClient, top_k: int = 3):
        self.embedder = embedder
        self.llm = llm
        self.top_k = top_k
        self.documents: list[Document] = []

    def add_documents(
        self, texts: list[str], metadata: list[dict[str, Any]] | None = None
    ):
        """添加文档到知识库"""
        if metadata is None:
            metadata = [{} for _ in texts]

        vectors = self.embedder.embed(texts)

        for text, meta, vec in zip(texts, metadata, vectors):
            doc = Document(content=text, metadata=meta, embedding=vec)
            self.documents.append(doc)

        print(f"✅ 添加 {len(texts)} 个文档，当前共 {len(self.documents)} 个")

    def retrieve(self, query: str) -> list[tuple[Document, float]]:
        """检索最相关的 Top-K 文档"""
        query_vec = self.embedder.embed(query)

        scores = []
        for doc in self.documents:
            sim = Embedder.cosine_similarity(query_vec, doc.embedding)
            scores.append((doc, float(sim)))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[: self.top_k]

    def generate(self, query: str, retrieved_docs: list[tuple[Document, float]]) -> str:
        """基于检索结果生成回答"""
        context = "\n\n".join(
            f"[文档{i+1}] {doc.content}" for i, (doc, _) in enumerate(retrieved_docs)
        )

        prompt = (
            f"请根据以下参考资料回答问题。如果资料中没有答案，请说明。\n\n"
            f"参考资料：\n{context}\n\n"
            f"问题：{query}\n\n"
            f"回答："
        )
        return self.llm.simple_chat(
            prompt, system_prompt="你是一个基于知识库回答问题的助手。"
        )

    def query(self, question: str) -> dict:
        """完整 RAG 流程：检索 + 生成"""
        retrieved = self.retrieve(question)
        answer = self.generate(question, retrieved)

        return {
            "question": question,
            "retrieved_docs": [
                {"content": doc.content[:100], "score": score, "metadata": doc.metadata}
                for doc, score in retrieved
            ],
            "answer": answer,
        }


def demo():
    """演示 RAG 基本流程"""
    embedder = Embedder()
    llm = LLMClient()
    rag = NaiveRAG(embedder, llm, top_k=3)

    docs = [
        "Python 是一种解释型、高级编程语言，由 Guido van Rossum 于 1991 年创建。",
        "Python 的设计哲学强调代码可读性和简洁语法，尤其是使用空格缩进划分代码块。",
        "Python 支持多种编程范式，包括面向对象、命令式、函数式和过程式编程。",
        "Transformers 库由 HuggingFace 开发，提供了大量预训练模型，是 NLP 领域最流行的库之一。",
        "RAG（检索增强生成）是一种结合检索和生成的技术，用于减少大模型的幻觉问题。",
        "LangChain 是一个用于开发 LLM 应用的框架，支持链式调用、Agent、记忆等功能。",
    ]

    rag.add_documents(docs)

    questions = [
        "Python 是谁发明的？",
        "RAG 有什么用？",
        "Transformers 库是什么？",
    ]

    for q in questions:
        print(f"\n{'='*60}")
        print(f"❓ 问题：{q}")
        result = rag.query(q)
        print("📚 检索到的文档：")
        for doc in result["retrieved_docs"]:
            print(f"  [{doc['score']:.3f}] {doc['content']}")
        print(f"🤖 回答：{result['answer']}")


if __name__ == "__main__":
    demo()
