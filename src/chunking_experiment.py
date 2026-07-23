"""切块策略对比实验"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt

from pathlib import Path

from src.embedder import Embedder
from src.llm_client import LLMClient
from src.naive_rag.rag import NaiveRAG
from src.naive_rag.loader import TextChunker


def run_chunking_experiment(
    text: str, embedder: Embedder, llm: LLMClient
) -> list[dict]:
    """对比不同切块策略"""

    # 实验配置
    configs = [
        {"name": "fixed-200", "method": "fixed", "chunk_size": 200, "overlap": 20},
        {"name": "fixed-500", "method": "fixed", "chunk_size": 500, "overlap": 50},
        {"name": "fixed-1000", "method": "fixed", "chunk_size": 1000, "overlap": 100},
        {"name": "sentence-3", "method": "sentence", "sentences_per_chunk": 3},
        {"name": "sentence-5", "method": "sentence", "sentences_per_chunk": 5},
        {"name": "paragraph", "method": "paragraph"},
    ]

    # 测试问题 + 标准答案(用于评估检索准确性)
    test_cases = [
        {"question": "AI 经历了几次浪潮？", "expected_keyword": "三次"},
        {"question": "谁提出了图灵测试？", "expected_keyword": "Turing"},
        {"question": "Transformer 是哪一年提出的？", "expected_keyword": "2017"},
        {"question": "Agent 的四个核心组件是什么？", "expected_keyword": "规划"},
        {"question": "MCP 是哪个公司提出的？", "expected_keyword": "Anthropic"},
    ]

    results = []
    for config in configs:
        print(f"\n{'=' * 60}")
        print(f"实验配置：{config['name']}")
        print(f"{'=' * 60}")

        # 切块
        if config["method"] == "fixed":
            chunks = TextChunker.fixed_size(
                text, config["chunk_size"], config["overlap"]
            )
        elif config["method"] == "sentence":
            chunks = TextChunker.by_sentence(text, config["sentences_per_chunk"])
        elif config["method"] == "paragraph":
            chunks = TextChunker.by_paragraph(text)
        else:
            continue

        avg_len = sum(len(c) for c in chunks) / len(chunks)
        print(f"  切块数：{len(chunks)}")
        print(f"  平均块长：{avg_len:.0f} 字符")

        rag = NaiveRAG(embedder, llm, top_k=3)
        rag.add_documents(chunks)

        correct = 0
        for tc in test_cases:
            retrieved = rag.retrieve(tc["question"])
            top1_content = retrieved[0][0].content
            hit = tc["expected_keyword"] in top1_content
            if hit:
                correct += 1
            status = "✅" if hit else "❌"
            print(f"  {status} Q: {tc['question']} | 期望含 '{tc['expected_keyword']}'")

        accuracy = correct / len(test_cases)
        print(f"  检索准确率：{correct}/{len(test_cases)} = {accuracy:.1%}")

        results.append(
            {
                "config": config["name"],
                "num_chunks": len(chunks),
                "avg_chunk_length": avg_len,
                "retrieval_accuracy": accuracy,
                "correct": correct,
                "total": len(test_cases),
            }
        )

    return results


def plot_results(results: list[dict]):
    """绘制对比图"""
    import matplotlib

    matplotlib.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    names = [r["config"] for r in results]
    accuracies = [r["retrieval_accuracy"] for r in results]
    num_chunks = [r["num_chunks"] for r in results]

    # 准确率对比
    bars = axes[0].bar(names, accuracies, color="steelblue")
    axes[0].set_ylabel("检索准确率")
    axes[0].set_title("切块策略 vs 检索准确率")
    axes[0].set_ylim(0, 1.1)
    for bar, acc in zip(bars, accuracies):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{acc:.0%}",
            ha="center",
        )

    # 块数对比
    axes[1].bar(names, num_chunks, color="coral")
    axes[1].set_ylabel("切块数量")
    axes[1].set_title("切块策略 vs 块数")

    plt.tight_layout()
    plt.savefig("docs/chunking-experiment.png", dpi=150)
    print("✅ 图表保存到 docs/chunking-experiment.png")


def main():
    os.makedirs("docs", exist_ok=True)

    try:
        # 加载样本数据
        text = Path("data/sample_knowledge.txt").read_text(encoding="utf-8")
    except FileNotFoundError:
        print(
            "❌ 找不到 data/sample_knowledge.txt，请先运行 python scripts/gen_sample_data.py"
        )
        return

    print(f"📄 样本数据：{len(text)} 字符")

    embedder = Embedder()
    llm = LLMClient()

    # 运行实验
    results = run_chunking_experiment(text, embedder, llm)

    try:
        # 保存实验结果
        with open("docs/chunking-results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print("✅ 结果保存到 docs/chunking-results.json")
    except OSError as e:
        print(f"❌ 保存 JSON 失败：{e}")

    # 绘制对比图
    plot_results(results)

    # 实验总结
    print(f"\n{'=' * 60}")
    print("📊 实验总结")
    print(f"{'=' * 60}")

    best = max(results, key=lambda x: x["retrieval_accuracy"])
    print(f"最佳策略：{best['config']}（准确率 {best['retrieval_accuracy']:.1%}）")
    print(f"  块数：{best['num_chunks']}")
    print(f"  平均块长：{best['avg_chunk_length']:.0f} 字符")


if __name__ == "__main__":
    main()
