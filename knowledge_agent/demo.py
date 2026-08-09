# -*- coding: utf-8 -*-
"""knowledge_agent 演示脚本（供 OBS 录制 3-5 分钟视频）

演示流程:
  1. 项目结构展示
  2. 初始化 Agent
  3. 导入文档（PDF / TXT / Markdown）
  4. 对话演示 - 知识库检索 (RAG)
  5. 工具调用演示 - 时间 / 计算器

运行:
  conda run -n ai-agent python knowledge_agent/demo.py
"""

import os
import sys
import time
from pathlib import Path

# UTF-8 输出，避免 Windows GBK 编码错误
sys.stdout.reconfigure(encoding="utf-8")

# 添加项目父目录到 Python 路径，使 knowledge_agent 包可被导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knowledge_agent.src.agent import KnowledgeAgent  # noqa: E402
from knowledge_agent.src.config import DATA_DIR  # noqa: E402

LINE = "=" * 60


def section(num, title):
    print("\n" + LINE)
    print(f"  {num}. {title}")
    print(LINE)
    time.sleep(1.5)


def pause(s=2):
    time.sleep(s)


def show_project_structure():
    """展示 knowledge_agent 目录结构"""
    root = Path(__file__).resolve().parent
    print(f"项目根目录: {root}")
    print()
    skip_dirs = {"__pycache__", ".pytest_cache", "chroma_db", "uploads", ".git"}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in skip_dirs)
        rel = os.path.relpath(dirpath, root)
        depth = 0 if rel == "." else rel.count(os.sep) + 1
        indent = "    " * depth
        name = root.name if rel == "." else os.path.basename(dirpath)
        print(f"{indent}{name}/")
        sub_indent = "    " * (depth + 1)
        for f in sorted(filenames):
            if f.endswith((".pyc",)):
                continue
            print(f"{sub_indent}{f}")
    pause(4)


def main():
    print(LINE)
    print("  knowledge_agent 个人知识库 Agent 演示")
    print(LINE)
    pause(2)

    # 1. 项目结构
    section("1", "项目结构展示")
    show_project_structure()

    # 2. 初始化 Agent
    section("2", "初始化知识库 Agent")
    print("正在初始化 Agent（加载 LLM、向量库、工具）...")
    agent = KnowledgeAgent()
    stats = agent.get_stats()
    print(f"知识库状态: {stats}")
    pause(2)

    # 3. 导入文档
    section("3", "导入测试文档 (PDF / TXT / Markdown)")
    # 清空旧知识库，确保演示从零开始
    try:
        if stats.get("chunk_count", 0) > 0:
            print("  清空旧知识库...")
            agent.rag.clear()
            pause(1)
    except Exception as e:
        print(f"  (清空跳过: {e})")

    docs = ["ai_agent.pdf", "vector_db.txt", "prompt_engineering.md", "sample.txt"]
    for name in docs:
        path = DATA_DIR / name
        if not path.exists():
            print(f"  [跳过] {name} 不存在")
            continue
        print(f"  导入: {name} ...", end=" ", flush=True)
        try:
            result = agent.import_document(str(path))
            if result.get("success"):
                print(f"OK ({result['chunks']} 个分块)")
            else:
                print(f"失败: {result.get('message')}")
        except Exception as e:
            print(f"出错: {e}")
        pause(1)

    stats = agent.get_stats()
    print(f"\n导入完成，知识库共 {stats.get('chunk_count', 0)} 个分块")
    pause(2)

    # 4. 对话演示 - 知识库检索
    section("4", "对话演示 - 知识库检索 (RAG)")
    rag_questions = [
        "AI Agent 的核心架构包含哪些组件？",
        "向量数据库常见的距离度量有哪些？",
        "Few-shot 提示是什么意思？",
    ]
    for q in rag_questions:
        print(f"\n[用户] {q}")
        print("[Agent] ", end="", flush=True)
        try:
            answer = agent.chat(q)
        except Exception as e:
            answer = f"(出错: {e})"
        print(answer)
        pause(3)

    # 5. 工具调用演示
    section("5", "工具调用演示 (时间 / 计算器)")
    tool_questions = [
        ("现在几点了？", "get_time"),
        ("帮我计算 25 * 17 + 100", "calculator"),
    ]
    for q, tool in tool_questions:
        print(f"\n[用户] {q}")
        print(f"  (预期触发工具: {tool})")
        print("[Agent] ", end="", flush=True)
        try:
            answer = agent.chat(q)
        except Exception as e:
            answer = f"(出错: {e})"
        print(answer)
        pause(3)

    # 结束
    print("\n" + LINE)
    print("  演示结束")
    print(LINE)


if __name__ == "__main__":
    main()
