"""Agent 系统评估"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm_client import LLMClient
from src.tools.builtin import create_default_registry
from src.agent_v1 import CompleteAgent, AgentConfig


# 扩展评测集（10 个任务）
EVAL_TASKS = [
    # 工具使用类
    {
        "question": "现在几点了？",
        "expected": "202",
        "category": "时间",
        "tool": "get_current_time",
    },
    {
        "question": "计算 15 * 23 的结果",
        "expected": "345",
        "category": "计算",
        "tool": "calculator",
    },
    {
        "question": "计算 sin(pi/2) 的值",
        "expected": "1",
        "category": "计算",
        "tool": "calculator",
    },
    {
        "question": "列出当前目录文件",
        "expected": "📁",
        "category": "文件",
        "tool": "list_directory",
    },
    {
        "question": "读取 data/test_tool.txt 文件内容",
        "expected": "Hello",
        "category": "文件",
        "tool": "read_file",
    },
    {
        "question": "创建文件 eval_test.txt，内容为'评估测试'",
        "expected": "已写入",
        "category": "文件",
        "tool": "write_file",
    },
    # 多步任务类
    {
        "question": "先计算 100/4，然后告诉我现在时间",
        "expected": "25",
        "category": "多步",
        "tool": "multi",
    },
    {
        "question": "计算 2^10 然后把结果写入文件 power_result.txt",
        "expected": "已写入",
        "category": "多步",
        "tool": "multi",
    },
    # 推理类
    {
        "question": "如果每个苹果 3 元，买 15 个苹果需要多少钱？",
        "expected": "45",
        "category": "推理",
        "tool": "calculator",
    },
    {
        "question": "一个长方形长 8 米宽 5 米，面积是多少？",
        "expected": "40",
        "category": "推理",
        "tool": "calculator",
    },
]


def evaluate_agent(verbose: bool = False) -> dict:
    """完整评估"""
    llm = LLMClient()
    registry = create_default_registry()

    agent = CompleteAgent(
        llm,
        registry,
        memory=None,
        config=AgentConfig(max_steps=8, verbose=verbose),
    )

    results = []
    for i, task in enumerate(EVAL_TASKS, 1):
        print(f"\n[{i}/{len(EVAL_TASKS)}] {task['category']}: {task['question']}")

        result = agent.run(task["question"])

        # 检查是否成功（答案或观察中包含期望关键词）
        success = task["expected"] in result["answer"] or any(
            task["expected"] in str(s.get("observation", "")) for s in result["steps"]
        )

        results.append(
            {
                "category": task["category"],
                "question": task["question"],
                "expected": task["expected"],
                "answer": result["answer"][:100],
                "success": success,
                "steps": len(result["steps"]),
                "error": result.get("answer", "") if not success else None,
            }
        )

        print(f"  {'✅' if success else '❌'} 步数={len(result['steps'])}")

    # 分类统计
    categories = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"total": 0, "success": 0}
        categories[cat]["total"] += 1
        if r["success"]:
            categories[cat]["success"] += 1

    # 总结
    total = len(results)
    success_count = sum(r["success"] for r in results)
    success_rate = success_count / total

    print(f"\n{'='*60}")
    print("📊 Agent 评估报告")
    print(f"{'='*60}")
    print(f"\n总体成功率：{success_count}/{total} = {success_rate:.1%}")

    print("\n分类统计：")
    for cat, stats in categories.items():
        rate = stats["success"] / stats["total"]
        print(f"  {cat}: {stats['success']}/{stats['total']} = {rate:.0%}")

    print("\n失败任务：")
    failures = [r for r in results if not r["success"]]
    if failures:
        for f in failures:
            print(f"  ❌ [{f['category']}] {f['question']}")
            print(f"     期望: {f['expected']}")
            print(f"     实际: {f['answer'][:80]}")
    else:
        print("  （无）")

    # 保存报告
    report = {
        "total_tasks": total,
        "success_count": success_count,
        "success_rate": success_rate,
        "categories": categories,
        "details": results,
    }
    with open("docs/agent-eval-report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n📄 报告已保存到 docs/agent-eval-report.json")

    return report


if __name__ == "__main__":
    evaluate_agent(verbose=False)
