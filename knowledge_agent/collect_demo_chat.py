"""临时采集脚本：项目1 知识库 Agent 对话序列（GIF 数据源）

用法:
    conda activate ai-agent
    python knowledge_agent/collect_demo_chat.py
输出:
    knowledge_agent/docs/demo_chat.json
"""

import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent))  # 仓库根

from knowledge_agent.src.agent import KnowledgeAgent  # noqa: E402

# (阶段标签, 用户输入/命令, 类型)；None 表示只取系统状态
SEQUENCE = [
    ("stats", None, "stats"),
    ("chat", "AI Agent 的核心架构包含哪些组件？", "rag"),
    ("chat", "计算 15 乘以 23 等于多少？", "tool"),
    ("chat", "帮我读取 test_tool.txt 文件的内容", "tool"),
]


def main() -> None:
    """按序列采集 stats 与三轮对话（RAG + 工具调用）。"""
    agent = KnowledgeAgent()
    records = []

    stats = agent.get_stats()
    print("知识库状态:", stats)
    records.append({"kind": "stats", "stats": stats})

    for _label, user_input, kind in SEQUENCE[1:]:
        print(f"\n[您] {user_input}")
        try:
            answer = agent.chat(user_input)
        except Exception as e:  # 单轮失败不中断
            answer = f"[采集失败: {e}]"
        print(f"[Agent] {answer[:120]}")
        records.append({"kind": kind, "input": user_input, "answer": answer})

    out = BASE / "docs" / "demo_chat.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入 {out}")


if __name__ == "__main__":
    main()
