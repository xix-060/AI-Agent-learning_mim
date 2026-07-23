"""完整 Naive RAG 系统"""

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
from src.naive_rag.loader import DocumentLoader, TextChunker


@dataclass
class Document:
    """文档块"""

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None


class NaiveRAG:
    """完整 RAG 系统"""

    def __init__(self, embedder: Embedder, llm: LLMClient, top_k: int = 3):
        self.embedder = embedder
        self.llm = llm
        self.top_k = top_k
        self.documents: list[Document] = []

    def load_file(
        self,
        file_path: str,
        chunk_method: str = "fixed",
        chunk_size: int = 500,
        overlap: int = 50,
    ) -> int:
        """加载文件到知识库，返回添加的块数"""
        pages = DocumentLoader.load(file_path)
        full_text = "\n\n".join(pages)

        if chunk_method == "fixed":
            chunks = TextChunker.fixed_size(full_text, chunk_size, overlap)
        elif chunk_method == "sentence":
            chunks = TextChunker.by_sentence(full_text)
        elif chunk_method == "paragraph":
            chunks = TextChunker.by_paragraph(full_text)
        else:
            chunks = TextChunker.fixed_size(full_text, chunk_size, overlap)

        metadata_list = [
            {"source": file_path, "chunk_index": i, "total_chunks": len(chunks)}
            for i in range(len(chunks))
        ]
        self.add_documents(chunks, metadata_list)
        return len(chunks)

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

    def query(self, question: str, show_context: bool = False) -> dict:
        """完整 RAG 流程"""
        retrieved = self.retrieve(question)
        answer = self.generate(question, retrieved)

        result = {
            "question": question,
            "retrieved_docs": [
                {"content": doc.content, "score": score, "metadata": doc.metadata}
                for doc, score in retrieved
            ],
            "answer": answer,
        }

        if show_context:
            print("\n📚 检索到的上下文：")
            for i, (doc, score) in enumerate(retrieved):
                print(f"  [文档{i+1} | 相似度={score:.3f}]")
                print(f"  {doc.content[:200]}...")
            print(f"\n🤖 回答：{answer}")

        return result


def demo_with_text():
    """用文本演示 RAG"""
    embedder = Embedder()
    llm = LLMClient()
    rag = NaiveRAG(embedder, llm, top_k=3)

    knowledge_base = [
        (
            "LangChain 是一个用于开发 LLM 应用的开源框架。它提供了模块化组件，包括：\n"
            "1. Models：对接各种 LLM（OpenAI、Qwen、LLaMA 等）\n"
            "2. Prompts：提示模板管理\n"
            "3. Chains：链式调用（把多个步骤串起来）\n"
            "4. Agents：自主决策的智能体\n"
            "5. Memory：对话记忆管理\n"
            "6. Retrievers：检索器（RAG 核心组件）\n"
            "LangChain 支持 Python 和 JavaScript 两个版本。"
        ),
        (
            "LangGraph 是 LangChain 团队推出的 Agent 编排框架。它基于状态机理念，\n"
            "允许开发者定义：\n"
            "- 节点（Node）：每一步的处理逻辑\n"
            "- 边（Edge）：节点之间的转移条件\n"
            "- 状态（State）：在节点间传递的数据\n"
            "LangGraph 特别适合需要复杂流程控制的 Agent，如多步推理、人机协同、\n"
            "可暂停/可恢复的工作流。它是构建生产级 Agent 的首选工具。"
        ),
        (
            "RAG（Retrieval-Augmented Generation，检索增强生成）通过三个步骤工作：\n"
            "1. 索引（Indexing）：将文档切块、向量化、存入向量库\n"
            "2. 检索（Retrieval）：用户提问后，用相同模型向量化问题，在向量库中检索 Top-K 相关文档\n"
            "3. 生成（Generation）：将检索到的文档作为上下文，和问题一起发给 LLM 生成答案\n"
            "RAG 的核心优势：减少幻觉、知识可更新、可溯源、数据隐私。"
        ),
        (
            "向量数据库是专门存储和检索向量的数据库。常见选择：\n"
            "- Chroma：轻量级，适合开发原型\n"
            "- Milvus：分布式，适合生产环境，支持十亿级向量\n"
            "- Faiss：Facebook 开源，纯库无服务，高性能\n"
            "- Pinecone：云服务，免运维\n"
            "- Weaviate：开源，支持混合检索\n"
            "选择建议：开发用 Chroma，生产用 Milvus，快速验证用 Faiss。"
        ),
    ]

    rag.add_documents(knowledge_base)
    questions = [
        "LangChain 有哪些核心组件？",
        "LangGraph 和 LangChain 有什么区别？",
        "RAG 是怎么工作的？",
        "如果要部署生产级 RAG，该选哪个向量库？",
    ]

    for q in questions:
        rag.query(q, show_context=True)
        print("-" * 60)


def demo_with_pdf(pdf_path: str):
    """用 PDF 演示 RAG"""
    embedder = Embedder()
    llm = LLMClient()
    rag = NaiveRAG(embedder, llm, top_k=3)

    print(f"📄 加载 PDF: {pdf_path}")
    num_chunks = rag.load_file(
        pdf_path, chunk_method="fixed", chunk_size=500, overlap=50
    )
    print(f"✅ 切成 {num_chunks} 个块")

    print("\n进入问答模式（输入 quit 退出）")
    while True:
        q = input("\n❓ 问题：")
        if q.lower() in ("quit", "exit", "q"):
            break
        rag.query(q, show_context=True)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # python -m src.naive_rag.rag path/to/file.pdf
        demo_with_pdf(sys.argv[1])
    else:
        demo_with_text()
