"""记忆管理模块 - 短期记忆 + 长期记忆"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage


class ShortTermMemory:
    """短期记忆 - 保留最近 N 轮对话"""

    def __init__(self, max_messages: int = 20):
        self.max_messages = max_messages
        self.history: List[BaseMessage] = []

    def add_user_message(self, content: str) -> None:
        self.history.append(HumanMessage(content=content))
        self._trim()

    def add_ai_message(self, content: str) -> None:
        self.history.append(AIMessage(content=content))
        self._trim()

    def add_message(self, message: BaseMessage) -> None:
        self.history.append(message)
        self._trim()

    def get_messages(self) -> List[BaseMessage]:
        return self.history.copy()

    def get_recent(self, n: int = 5) -> List[BaseMessage]:
        return self.history[-n:]

    def clear(self) -> None:
        self.history = []

    def _trim(self) -> None:
        if len(self.history) > self.max_messages:
            self.history = self.history[-self.max_messages :]

    def get_summary(self) -> str:
        if not self.history:
            return "暂无对话记录"
        count = len(self.history)
        first = self.history[0]
        role = "用户" if isinstance(first, HumanMessage) else "助手"
        preview = first.content[:40]
        return f"共 {count} 条消息，最近: {role}: {preview}..."


class LongTermMemory:
    """长期记忆 - 持久化存储关键信息到 JSON 文件"""

    def __init__(self, storage_path: Optional[str] = None):
        if storage_path is None:
            from knowledge_agent.src.config import DATA_DIR

            storage_path = str(DATA_DIR / "long_term_memory.json")
        self.storage_path = Path(storage_path)
        self._data: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if self.storage_path.exists():
            try:
                self._data = json.loads(self.storage_path.read_text(encoding="utf-8"))
            except Exception:
                self._data = []

    def _save(self) -> None:
        self.storage_path.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def add(self, key: str, value: str, metadata: Optional[Dict] = None) -> None:
        """添加一条长期记忆"""
        entry = {
            "key": key,
            "value": value,
            "metadata": metadata or {},
        }
        self._data.append(entry)
        self._save()

    def search(self, keyword: str) -> List[Dict[str, Any]]:
        """搜索长期记忆"""
        keyword_lower = keyword.lower()
        return [
            entry
            for entry in self._data
            if keyword_lower in entry["key"].lower()
            or keyword_lower in entry["value"].lower()
        ]

    def get_all(self) -> List[Dict[str, Any]]:
        return self._data.copy()

    def clear(self) -> None:
        self._data = []
        self._save()


class ConversationMemory:
    """对话记忆管理器 - 整合短期 + 长期记忆"""

    def __init__(self, max_short_term: int = 20):
        self.short_term = ShortTermMemory(max_messages=max_short_term)
        self.long_term = LongTermMemory()

    def add_user_message(self, content: str) -> None:
        self.short_term.add_user_message(content)

    def add_ai_message(self, content: str) -> None:
        self.short_term.add_ai_message(content)

    def get_history(self) -> List[BaseMessage]:
        return self.short_term.get_messages()

    def get_context(self, keyword: Optional[str] = None) -> str:
        """获取记忆上下文（短期 + 长期检索）"""
        parts = []

        # 短期记忆摘要
        short_summary = self.short_term.get_summary()
        parts.append(f"[短期记忆] {short_summary}")

        # 长期记忆检索
        if keyword:
            long_results = self.long_term.search(keyword)
            if long_results:
                long_parts = [f"  - {r['key']}: {r['value']}" for r in long_results[:5]]
                parts.append("[长期记忆]\n" + "\n".join(long_parts))

        return "\n".join(parts)

    def remember(self, key: str, value: str, metadata: Optional[Dict] = None) -> None:
        """存入长期记忆"""
        self.long_term.add(key, value, metadata)

    def recall(self, keyword: str) -> List[Dict[str, Any]]:
        """从长期记忆中检索"""
        return self.long_term.search(keyword)

    def clear_short_term(self) -> None:
        """清空短期记忆"""
        self.short_term.clear()

    def clear_all(self) -> None:
        """清空所有记忆"""
        self.short_term.clear()
        self.long_term.clear()
