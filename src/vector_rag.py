"""基于 Chroma 向量库的 RAG"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb
from pathlib import Path

from src.embedder import Embedder
from src.llm_client import LLMClient
from src.naive_rag.loader import DocumentLoader, TextChunker


class ChromaRAG:
    """使用 Chroma 向量库的 RAG 系统"""

    def __init__(
        self,
        embedder: Embedder,
        llm: LLMClient,
        collection_name: str = "knowledge_base",
        persist_directory: str = "./data/chroma_db",
        top_k: int = 3,
    ):
        self.embedder = embedder
        self.llm = llm
        self.top_k = top_k

        # 确保目录存在
        os.makedirs(persist_directory, exist_ok=True)

        # 初始化 Chroma 客户端（持久化存储）
        self.client = chromadb.PersistentClient(path=persist_directory)

        # 获取或创建集合
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        print(f"✅ Chroma 向量库就绪，当前 {self.collection.count()} 条记录")

    def add_documents(self, texts: list[str], metadata: list[dict] | None = None):
        """添加文档到向量库"""
        if not texts:
            return

        if metadata is None:
            metadata = [{} for _ in texts]

        try:
            # 批量 Embedding
            vectors = self.embedder.embed(texts).tolist()

            # 生成 ID
            existing_count = self.collection.count()
            ids = [f"doc_{existing_count + i}" for i in range(len(texts))]

            # 添加到 Chroma
            self.collection.add(
                embeddings=vectors,
                documents=texts,
                metadatas=metadata,
                ids=ids,
            )

            print(f"✅ 添加 {len(texts)} 个文档，当前共 {self.collection.count()} 条")
        except Exception as e:
            print(f"❌ 添加文档失败: {e}")
            raise

    def load_file(
        self,
        file_path: str,
        chunk_size: int = 500,
        overlap: int = 50,
    ):
        """加载文件"""
        try:
            pages = DocumentLoader.load(file_path)
            full_text = "\n\n".join(pages)
            chunks = TextChunker.fixed_size(full_text, chunk_size, overlap)

            metadata_list = [
                {"source": file_path, "chunk_index": i} for i in range(len(chunks))
            ]
            self.add_documents(chunks, metadata_list)
            return len(chunks)
        except Exception as e:
            print(f"❌ 加载文件失败: {e}")
            raise

    def retrieve(self, query: str) -> list[dict]:
        """检索"""
        query_vec = self.embedder.embed(query).tolist()

        results = self.collection.query(
            query_embeddings=[query_vec],
            n_results=self.top_k,
        )

        retrieved = []
        for i in range(len(results["documents"][0])):
            retrieved.append(
                {
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "score": 1 - results["distances"][0][i],
                }
            )
        return retrieved

    def generate(self, query: str, retrieved_docs: list[dict]) -> str:
        """生成回答"""
        context = "\n\n".join(
            f"[文档{i + 1}] {doc['content']}" for i, doc in enumerate(retrieved_docs)
        )

        prompt = f"""请根据以下参考资料回答问题。如果资料中没有答案，请说明。

参考资料：
{context}

问题：{query}

回答："""
        return self.llm.simple_chat(
            prompt, system_prompt="你是一个基于知识库回答问题的助手。"
        )

    def query(self, question: str, show_context: bool = False) -> dict:
        """完整 RAG 查询"""
        retrieved = self.retrieve(question)
        answer = self.generate(question, retrieved)

        if show_context:
            print("\n📚 检索到的上下文：")
            for i, doc in enumerate(retrieved):
                print(f"  [文档{i + 1} | 相似度={doc['score']:.3f}]")
                print(f"  {doc['content'][:200]}...")
            print(f"\n🤖 回答：{answer}")

        return {
            "question": question,
            "retrieved_docs": retrieved,
            "answer": answer,
        }

    def clear(self):
        """清空集合"""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"},
        )
        print("✅ 已清空向量库")


def demo():
    """演示 Chroma RAG"""
    embedder = Embedder()
    llm = LLMClient()
    rag = ChromaRAG(embedder, llm, top_k=3)

    # 如果向量库为空，加载文档
    if rag.collection.count() == 0:
        try:
            text = Path("data/sample_knowledge.txt").read_text(encoding="utf-8")
            chunks = TextChunker.fixed_size(text, chunk_size=500, overlap=50)
            rag.add_documents(chunks)
        except FileNotFoundError:
            print("❌ 找不到 data/sample_knowledge.txt")
            return
        except Exception as e:
            print(f"❌ 加载文档失败: {e}")
            return

    # 问答
    questions = [
        "Transformer 是哪一年提出的？",
        "Agent 的四个核心组件是什么？",
        "MCP 是什么？",
    ]

    for q in questions:
        rag.query(q, show_context=True)
        print("-" * 60)


if __name__ == "__main__":
    demo()
