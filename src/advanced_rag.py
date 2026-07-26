"""高级 RAG：Query 改写 + HyDE + Reranker"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
from pathlib import Path

from src.embedder import Embedder
from src.llm_client import LLMClient
from src.vector_rag import ChromaRAG


class AdvancedRAG:
    """高级 RAG 系统"""

    def __init__(
        self,
        embedder: Embedder,
        llm: LLMClient,
        rag: ChromaRAG,
        retrieve_top_k: int = 10,
    ):
        self.embedder = embedder
        self.llm = llm
        self.rag = rag
        # 使用独立的 top_k，不修改传入对象
        self.retrieve_top_k = retrieve_top_k

    def _retrieve(self, query: str) -> list[dict]:
        """检索方法，临时修改 top_k 后恢复"""
        original_top_k = self.rag.top_k
        self.rag.top_k = self.retrieve_top_k
        try:
            return self.rag.retrieve(query)
        finally:
            self.rag.top_k = original_top_k

    # ========== Query 改写 ==========

    def rewrite_query(self, query: str) -> str:
        """用 LLM 改写查询，使其更适合检索"""
        prompt = f"""请将以下问题改写为更适合文档检索的查询语句。
要求：补充关键词、明确指代、去掉口语化表达。只输出改写后的查询，不要解释。

原始问题：{query}
改写后："""
        return self.llm.simple_chat(
            prompt, system_prompt="你是一个查询优化助手。"
        ).strip()

    # ========== HyDE ==========

    def hyde_generate(self, query: str) -> str:
        """HyDE：生成假设性文档用于检索"""
        prompt = f"""请为以下问题写一段可能的答案（100-200字）。不需要完全正确，只需表述接近真实文档即可。

问题：{query}
假设答案："""
        return self.llm.simple_chat(prompt).strip()

    # ========== Multi-Query ==========

    def multi_query(self, query: str, n: int = 3) -> list[str]:
        """生成多个不同角度的查询"""
        prompt = f"""请为以下问题生成{n}个不同角度的检索查询，每行一个，不要编号。

问题：{query}

{n}个检索查询："""
        result = self.llm.simple_chat(prompt)
        queries = [q.strip() for q in result.strip().split("\n") if q.strip()]
        return queries[:n]

    # ========== Reranker（简化版，用 LLM 做）==========

    def rerank(self, query: str, documents: list[dict], top_k: int = 3) -> list[dict]:
        """用 LLM 对检索结果重排序"""
        if len(documents) <= top_k:
            return documents

        # 构造排序 prompt（使用 list comprehension 避免 IDE 误报）
        docs_list = [
            f"[{i+1}] {doc['content'][:200]}" for i, doc in enumerate(documents)
        ]
        docs_text = "\n".join(docs_list)

        prompt = f"""请根据与问题的相关性，对以下文档排序。只输出排序后的文档编号，用逗号分隔，如：3,1,4,2,5

问题：{query}

文档：
{docs_text}

相关性排序（从高到低）："""
        result = self.llm.simple_chat(prompt, temperature=0.0).strip()

        # 解析排序结果
        indices = re.findall(r"\d+", result)
        indices = [int(i) - 1 for i in indices if int(i) <= len(documents)]

        # 按排序结果重排
        reranked = []
        for idx in indices:
            if idx < len(documents):
                reranked.append(documents[idx])
        # 补充未排序的
        for i, doc in enumerate(documents):
            if i not in indices:
                reranked.append(doc)

        return reranked[:top_k]

    # ========== 完整查询流程 ==========

    def query_with_rewrite(self, question: str) -> dict:
        """带 Query 改写的查询"""
        # 1. 改写
        rewritten = self.rewrite_query(question)
        print(f"  📝 改写：{question} → {rewritten}")

        # 2. 检索（临时扩大召回）
        retrieved = self._retrieve(rewritten)

        # 3. Rerank
        reranked = self.rerank(rewritten, retrieved, top_k=3)

        # 4. 生成
        answer = self.rag.generate(question, reranked)

        return {
            "question": question,
            "rewritten_query": rewritten,
            "retrieved_docs": reranked,
            "answer": answer,
        }

    def query_with_hyde(self, question: str) -> dict:
        """带 HyDE 的查询"""
        # 1. 生成假设文档
        hyde_doc = self.hyde_generate(question)
        print(f"  🔮 HyDE：{hyde_doc[:100]}...")

        # 2. 用假设文档检索（临时扩大召回）
        retrieved = self._retrieve(hyde_doc)

        # 3. Rerank
        reranked = self.rerank(question, retrieved, top_k=3)

        # 4. 生成
        answer = self.rag.generate(question, reranked)

        return {
            "question": question,
            "hyde_document": hyde_doc,
            "retrieved_docs": reranked,
            "answer": answer,
        }

    def query_with_multi(self, question: str) -> dict:
        """带 Multi-Query 的查询"""
        # 1. 生成多个查询
        queries = self.multi_query(question, n=3)
        print(f"  🌐 Multi-Query: {queries}")

        # 2. 分别检索（每次都临时扩大召回）
        all_docs = []
        for q in queries:
            docs = self._retrieve(q)
            all_docs.extend(docs)

        # 3. 去重（按内容）
        seen = set()
        unique_docs = []
        for doc in all_docs:
            if doc["content"] not in seen:
                seen.add(doc["content"])
                unique_docs.append(doc)

        # 4. Rerank
        reranked = self.rerank(question, unique_docs, top_k=3)

        # 5. 生成
        answer = self.rag.generate(question, reranked)

        return {
            "question": question,
            "multi_queries": queries,
            "retrieved_docs": reranked,
            "answer": answer,
        }


def compare_strategies():
    """对比 Naive vs Advanced RAG"""
    try:
        embedder = Embedder()
        llm = LLMClient()
        rag = ChromaRAG(embedder, llm, top_k=3)

        # 如果向量库为空，加载文档
        if rag.collection.count() == 0:
            print("📚 向量库为空，正在加载文档...")
            file_path = "data/sample_knowledge.txt"
            if Path(file_path).exists():
                rag.load_file(file_path)
            else:
                print(f"⚠️ 找不到 {file_path}，跳过文档加载")

        advanced = AdvancedRAG(embedder, llm, rag)

        questions = [
            "他发明的",  # 模糊查询，测试 Query 改写
            "MCP 是哪个公司提出的？",
            "Agent 的核心组件是什么？",
            "深度学习有哪些模型？",
        ]

        for q in questions:
            print(f"\n{'=' * 60}")
            print(f"❓ 问题：{q}")
            print(f"{'=' * 60}")

            print("\n[Naive RAG]")
            result_naive = rag.query(q)

            print("\n[Query 改写 + Reranker]")
            result_rewrite = advanced.query_with_rewrite(q)

            print("\n[HyDE + Reranker]")
            result_hyde = advanced.query_with_hyde(q)

            print("\n📋 对比：")
            print(f"  Naive: {result_naive['answer'][:80]}...")
            print(f"  Rewrite: {result_rewrite['answer'][:80]}...")
            print(f"  HyDE: {result_hyde['answer'][:80]}...")

    except Exception as e:
        print(f"❌ 运行失败: {e}")


if __name__ == "__main__":
    compare_strategies()
