# -*- coding: utf-8 -*-
"""项目 1 评测

基于 data/ 目录下的测试文档（PDF / TXT / Markdown）设计 5 个问题，
通过关键词匹配判断 Agent 是否能正确检索并回答。

用法:
  python eval.py            # 默认从 knowledge_agent/ 目录运行
  python knowledge_agent/eval.py  # 从项目根目录运行
"""

import os
import sys

# 添加项目父目录到 Python 路径，使 knowledge_agent 包可被导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knowledge_agent.src.agent import KnowledgeAgent  # noqa: E402
from knowledge_agent.src.config import DATA_DIR  # noqa: E402


# 测试文档（位于 data/ 目录）
TEST_DOCUMENTS = [
    "ai_agent.pdf",  # PDF：AI Agent 技术白皮书
    "vector_db.txt",  # TXT：向量数据库基础知识
    "prompt_engineering.md",  # Markdown：Prompt Engineering 笔记
    "sample.txt",  # TXT：RAG 技术简介
]

EVAL_TASKS = [
    # 覆盖 PDF 文档（AI Agent 知识）
    {"question": "AI Agent 的核心架构包含哪些组件？", "expected": "感知"},
    {"question": "ReAct 框架是什么？", "expected": "推理"},
    # 覆盖 TXT 文档（向量数据库）
    {"question": "向量数据库常见的距离度量有哪些？", "expected": "余弦"},
    {"question": "Chroma 是什么？", "expected": "向量数据库"},
    # 覆盖 Markdown 文档（Prompt Engineering）
    {"question": "Few-shot 提示是什么意思？", "expected": "示例"},
]


def ensure_knowledge_base(agent: KnowledgeAgent) -> None:
    """确保知识库已导入测试文档（仅在知识库为空时导入，避免重复）"""
    stats = agent.get_stats()
    if stats.get("chunk_count", 0) > 0:
        print(f"[INFO] 知识库已有 {stats['chunk_count']} 个分块，跳过导入")
        return

    print("[INFO] 知识库为空，开始导入测试文档...")
    for name in TEST_DOCUMENTS:
        path = DATA_DIR / name
        if not path.exists():
            print(f"  [跳过] 文件不存在: {path}")
            continue
        result = agent.import_document(str(path))
        if result.get("success"):
            print(f"  [OK] {result['source']}: {result['chunks']} 个分块")
        else:
            print(f"  [FAIL] {name}: {result.get('message', '未知错误')}")


def main():
    agent = KnowledgeAgent()

    # 1. 准备知识库（导入 data/ 下的测试文档）
    ensure_knowledge_base(agent)

    # 2. 执行评测
    print("\n" + "=" * 60)
    print("开始评测...")
    print("=" * 60)

    results = []
    for task in EVAL_TASKS:
        response = agent.chat(task["question"])
        success = task["expected"] in response
        results.append(
            {
                "question": task["question"],
                "expected": task["expected"],
                "answer": response[:100],
                "success": success,
            }
        )
        status = "[PASS]" if success else "[FAIL]"
        print(f"{status} {task['question']}")

    # 3. 统计
    success_count = sum(r["success"] for r in results)
    success_rate = success_count / len(results)
    print("\n" + "=" * 60)
    print(f"成功率：{success_rate:.0%} ({success_count}/{len(results)})")
    print("=" * 60)

    # 4. 详情
    print("\n详情：")
    for i, r in enumerate(results, 1):
        print(f"\n[{i}] {r['question']}")
        print(f"    期望关键词: {r['expected']}")
        print(f"    回答片段: {r['answer']}")


if __name__ == "__main__":
    main()
