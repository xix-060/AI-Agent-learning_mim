"""Agent 记忆三层架构"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
from src.embedder import Embedder
from src.llm_client import LLMClient


@dataclass
class Message:
    """单条消息"""

    role: str  # user / assistant / system
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)


# ========== 1. 短期记忆 ==========


class ShortTermMemory:
    """短期记忆：滑动窗口 + 摘要压缩"""

    def __init__(self, max_messages: int = 20, max_tokens: int = 4000):
        self.messages: list[Message] = []
        self.max_messages = max_messages
        self.max_tokens = max_tokens
        self.summary: str = ""  # 被压缩的历史摘要

    def add(self, role: str, content: str, **metadata):
        """添加消息"""
        msg = Message(role=role, content=content, metadata=metadata)
        self.messages.append(msg)
        self._compress_if_needed()

    def get_messages(self) -> list[Message]:
        """获取当前记忆中的消息"""
        return self.messages

    def get_context(self) -> str:
        """获取记忆上下文文本"""
        parts = []
        if self.summary:
            parts.append(f"[历史摘要] {self.summary}")
        for msg in self.messages:
            parts.append(f"{msg.role}: {msg.content}")
        return "\n".join(parts)

    def _compress_if_needed(self):
        """如果超出限制，压缩旧消息"""
        if len(self.messages) <= self.max_messages:
            return

        # 保留最近的消息，把旧的压缩成摘要
        old_messages = self.messages[: -self.max_messages // 2]
        self.messages = self.messages[-self.max_messages // 2 :]

        # 生成摘要（实际中用 LLM 生成，这里简化拼接）
        old_text = "\n".join(f"{m.role}: {m.content[:100]}" for m in old_messages)
        self.summary = (self.summary + "\n" + old_text)[-2000:]  # 限制摘要长度

    def clear(self):
        """清空"""
        self.messages = []
        self.summary = ""


# ========== 2. 长期记忆 ==========


class LongTermMemory:
    """长期记忆：基于向量库的语义检索"""

    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.memories: list[dict] = []  # 存储记忆
        self.embeddings: list[np.ndarray] = []  # 对应的向量

    def store(self, content: str, metadata: dict | None = None):
        """存储一条长期记忆"""
        vec = self.embedder.embed(content)
        self.memories.append(
            {
                "content": content,
                "metadata": metadata or {},
                "timestamp": datetime.now().isoformat(),
            }
        )
        self.embeddings.append(vec)
        print(f"💾 长期记忆已存储：{content[:50]}...（共 {len(self.memories)} 条）")

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        """检索相关记忆"""
        if not self.memories:
            return []

        query_vec = self.embedder.embed(query)
        similarities = [
            float(Embedder.cosine_similarity(query_vec, vec)) for vec in self.embeddings
        ]

        # 排序取 Top-K
        ranked = sorted(
            zip(self.memories, similarities),
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]

        return [{**mem, "score": score} for mem, score in ranked]

    def get_all(self) -> list[dict]:
        """获取所有记忆"""
        return self.memories


# ========== 3. 情景记忆 ==========


class EpisodicMemory:
    """情景记忆：记录完整交互情节"""

    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.episodes: list[dict] = []
        self.embeddings: list[np.ndarray] = []

    def record_episode(
        self,
        task: str,
        actions: list[str],
        result: str,
        success: bool,
        metadata: dict | None = None,
    ):
        """记录一个完整的交互情节"""
        episode = {
            "task": task,
            "actions": actions,
            "result": result,
            "success": success,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
        }

        # 向量化（用任务描述作为索引）
        vec = self.embedder.embed(task)
        self.episodes.append(episode)
        self.embeddings.append(vec)

        print(f"🎬 情节已记录：{task[:50]}...（成功={success}）")

    def retrieve_similar_episodes(self, task: str, top_k: int = 2) -> list[dict]:
        """检索类似的过往情节"""
        if not self.episodes:
            return []

        query_vec = self.embedder.embed(task)
        similarities = [
            float(Embedder.cosine_similarity(query_vec, vec)) for vec in self.embeddings
        ]

        ranked = sorted(
            zip(self.episodes, similarities),
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]

        return [{**ep, "relevance": score} for ep, score in ranked]


# ========== 4. 组合：Agent 记忆系统 ==========


class AgentMemory:
    """组合三层记忆的完整记忆系统"""

    def __init__(self, embedder: Embedder, llm: LLMClient):
        self.short_term = ShortTermMemory(max_messages=10)
        self.long_term = LongTermMemory(embedder)
        self.episodic = EpisodicMemory(embedder)
        self.llm = llm

    def add_message(self, role: str, content: str):
        """添加对话消息到短期记忆"""
        self.short_term.add(role, content)

    def store_fact(self, fact: str, metadata: dict | None = None):
        """存储事实到长期记忆"""
        self.long_term.store(fact, metadata)

    def record_episode(self, task: str, actions: list[str], result: str, success: bool):
        """记录交互情节"""
        self.episodic.record_episode(task, actions, result, success)

    def get_context_for_query(self, query: str) -> str:
        """获取用于回答 query 的完整记忆上下文"""
        parts = []

        # 1. 短期记忆（当前对话）
        short_ctx = self.short_term.get_context()
        if short_ctx:
            parts.append(f"【当前对话】\n{short_ctx}")

        # 2. 长期记忆（相关事实）
        long_results = self.long_term.retrieve(query, top_k=3)
        if long_results:
            long_text = "\n".join(f"- {r['content']}" for r in long_results)
            parts.append(f"【相关知识】\n{long_text}")

        # 3. 情景记忆（类似经历）
        episodes = self.episodic.retrieve_similar_episodes(query, top_k=2)
        if episodes:
            ep_text = "\n".join(
                f"- 任务: {ep['task']} → 结果: {ep['result'][:50]}... (成功={ep['success']})"
                for ep in episodes
            )
            parts.append(f"【过往经历】\n{ep_text}")

        return "\n\n".join(parts) if parts else "（无相关记忆）"


# ========== 演示 ==========
def demo():
    """演示三层记忆"""
    embedder = Embedder()
    llm = LLMClient()
    memory = AgentMemory(embedder, llm)

    # 1. 模拟对话（短期记忆）
    print("=" * 60)
    print("📝 短期记忆测试")
    print("=" * 60)
    memory.add_message("user", "我叫小明")
    memory.add_message("assistant", "你好，小明！")
    memory.add_message("user", "我是数据科学专业的学生")
    memory.add_message("assistant", "很好，数据科学很有前景！")

    # 2. 存储事实到长期记忆
    print("\n" + "=" * 60)
    print("💾 长期记忆测试")
    print("=" * 60)
    memory.store_fact("用户叫小明，是数据科学专业学生")
    memory.store_fact("用户正在学习 AI Agent 开发")
    memory.store_fact("用户使用 Python 3.11 和 LangChain")

    # 3. 记录情节
    print("\n" + "=" * 60)
    print("🎬 情景记忆测试")
    print("=" * 60)
    memory.record_episode(
        task="帮用户配置 LangChain 环境",
        actions=["检查 Python 版本", "安装 langchain", "验证安装"],
        result="成功配置 LangChain 0.3",
        success=True,
    )
    memory.record_episode(
        task="调试 RAG 检索问题",
        actions=["检查切块策略", "换 Embedding 模型", "加 Reranker"],
        result="检索准确率从 60% 提升到 85%",
        success=True,
    )

    # 4. 查询记忆
    print("\n" + "=" * 60)
    print("🔍 记忆检索测试")
    print("=" * 60)

    queries = [
        "我叫什么名字？",
        "我之前配置过什么环境？",
        "帮我配置一个新的开发环境",
    ]

    for q in queries:
        print(f"\n❓ 查询：{q}")
        context = memory.get_context_for_query(q)
        print(f"📚 记忆上下文：\n{context}")


if __name__ == "__main__":
    demo()
