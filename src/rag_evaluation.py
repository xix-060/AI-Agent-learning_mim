"""用 RAGAS 评估 RAG 系统质量"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

from datasets import Dataset
from src.embedder import Embedder
from src.llm_client import LLMClient
from src.vector_rag import ChromaRAG


# ====== 评估数据集 ======
# 每个测试用例包含：问题、标准答案、相关上下文（Ground Truth）
EVAL_DATASET = [
    {
        "question": "AI 经历了几次浪潮？",
        "ground_truth": "AI 经历了三次浪潮：第一次（1956-1974）以符号主义为代表，第二次（1980-1987）以专家系统和连接主义为代表，第三次（2006年至今）以深度学习为代表。",
        "contexts": [],
        "answer": "",
    },
    {
        "question": "谁提出了图灵测试？",
        "ground_truth": "Alan Turing 提出了图灵测试。",
        "contexts": [],
        "answer": "",
    },
    {
        "question": "Transformer 是哪一年提出的？",
        "ground_truth": "Transformer 由 Google 团队在 2017 年提出。",
        "contexts": [],
        "answer": "",
    },
    {
        "question": "Agent 的四个核心组件是什么？",
        "ground_truth": "Agent 的四个核心组件是：规划（Planning）、记忆（Memory）、工具使用（Tool Use）和行动执行（Action）。",
        "contexts": [],
        "answer": "",
    },
    {
        "question": "MCP 是哪个公司提出的？",
        "ground_truth": "MCP 是 Anthropic 在 2024 年提出的开放协议。",
        "contexts": [],
        "answer": "",
    },
]


def generate_rag_responses(rag: ChromaRAG, dataset: list[dict]) -> list[dict]:
    """用 RAG 系统生成回答，填充 contexts 和 answer"""
    for item in dataset:
        result = rag.query(item["question"])
        item["contexts"] = [doc["content"] for doc in result["retrieved_docs"]]
        item["answer"] = result["answer"]
        print(f"Q: {item['question']}")
        print(f"A: {item['answer'][:100]}...")
        print()

    return dataset


def evaluate_with_ragas(dataset: list[dict]):
    """用 RAGAS 评估"""
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )

    # 转换为 HuggingFace Dataset
    eval_data = {
        "question": [d["question"] for d in dataset],
        "answer": [d["answer"] for d in dataset],
        "contexts": [d["contexts"] for d in dataset],
        "ground_truth": [d["ground_truth"] for d in dataset],
    }
    hf_dataset = Dataset.from_dict(eval_data)

    # 运行评估
    eval_result = evaluate(
        dataset=hf_dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
    )

    return eval_result


def manual_evaluation(dataset: list[dict]) -> dict:
    """手动评估（RAGAS 依赖较重时的替代方案）"""
    results = []
    for item in dataset:
        # 1. 答案相关性：答案是否非空且与问题相关
        answer_relevant = len(item["answer"]) > 10

        # 2. 上下文命中率：检索到的上下文是否包含标准答案关键词
        gt_keywords = extract_keywords(item["ground_truth"])
        context_hit = any(
            any(kw in ctx for kw in gt_keywords) for ctx in item["contexts"]
        )

        # 3. 忠实度：答案是否来自上下文
        answer_keywords = extract_keywords(item["answer"])
        faithfulness_score = sum(
            1 for kw in answer_keywords if any(kw in ctx for ctx in item["contexts"])
        ) / max(len(answer_keywords), 1)

        results.append(
            {
                "question": item["question"],
                "answer_relevant": answer_relevant,
                "context_hit": context_hit,
                "faithfulness": faithfulness_score,
            }
        )

    # 汇总
    n = len(results)
    summary = {
        "total": n,
        "answer_relevancy": sum(r["answer_relevant"] for r in results) / n,
        "context_precision": sum(r["context_hit"] for r in results) / n,
        "faithfulness": sum(r["faithfulness"] for r in results) / n,
        "details": results,
    }
    return summary


def extract_keywords(text: str) -> list[str]:
    """简单提取关键词（人名、年份、专有名词）"""
    import re

    keywords = []
    # 年份
    keywords.extend(re.findall(r"\d{4}", text))
    # 英文单词
    keywords.extend(re.findall(r"[A-Z][a-zA-Z]+", text))
    return list(set(keywords))


def main():
    """完整评估流程"""
    os.makedirs("docs", exist_ok=True)

    print("📊 RAG 系统评估")
    print("=" * 60)

    # 1. 初始化 RAG
    embedder = Embedder()
    llm = LLMClient()
    rag = ChromaRAG(embedder, llm, top_k=3)

    # 加载知识库
    if rag.collection.count() == 0:
        from src.naive_rag.loader import TextChunker

        try:
            text = Path("data/sample_knowledge.txt").read_text(encoding="utf-8")
            chunks = TextChunker.fixed_size(text, chunk_size=500, overlap=50)
            rag.add_documents(chunks)
        except FileNotFoundError:
            print("❌ 找不到 data/sample_knowledge.txt")
            return
        except Exception as e:
            print(f"❌ 加载知识库失败: {e}")
            return

    # 2. 生成 RAG 回答
    print("\n📝 生成 RAG 回答...")
    dataset = [d.copy() for d in EVAL_DATASET]
    dataset = generate_rag_responses(rag, dataset)

    # 3. 评估
    print("\n📊 运行评估...")

    # 先做手动评估，用于后续对比
    manual_result = manual_evaluation(dataset)

    # 尝试用 RAGAS
    try:
        ragas_result = evaluate_with_ragas(dataset)
        print("\n✅ RAGAS 评估结果：")
        print(f"   Faithfulness（忠实度）: {ragas_result['faithfulness']:.3f}")
        print(
            f"   Answer Relevancy（答案相关性）: {ragas_result['answer_relevancy']:.3f}"
        )
        print(
            f"   Context Precision（上下文精确率）: {ragas_result['context_precision']:.3f}"
        )
        print(
            f"   Context Recall（上下文召回率）: {ragas_result['context_recall']:.3f}"
        )

        # 保存
        with open("docs/ragas-results.json", "w", encoding="utf-8") as f:
            json.dump(dict(ragas_result), f, ensure_ascii=False, indent=2)

        # 使用 RAGAS 的 faithfulness 作为最终结果
        faithfulness_score = ragas_result.get("faithfulness", 0)

    except ImportError:
        print("\n⚠️ RAGAS 未安装，使用手动评估")
        print("\n📋 手动评估结果：")
        print(f"   Answer Relevancy: {manual_result['answer_relevancy']:.3f}")
        print(f"   Context Precision: {manual_result['context_precision']:.3f}")
        print(f"   Faithfulness: {manual_result['faithfulness']:.3f}")

        with open("docs/rag-eval-manual.json", "w", encoding="utf-8") as f:
            json.dump(manual_result, f, ensure_ascii=False, indent=2)

        # 使用手动评估的 faithfulness 作为最终结果
        faithfulness_score = manual_result.get("faithfulness", 0)

    except Exception as e:
        print(f"\n⚠️ RAGAS 评估失败（{e}），使用手动评估")
        print("\n📋 手动评估结果：")
        print(f"   Answer Relevancy: {manual_result['answer_relevancy']:.3f}")
        print(f"   Context Precision: {manual_result['context_precision']:.3f}")
        print(f"   Faithfulness: {manual_result['faithfulness']:.3f}")

        with open("docs/rag-eval-manual.json", "w", encoding="utf-8") as f:
            json.dump(manual_result, f, ensure_ascii=False, indent=2)

        # 使用手动评估的 faithfulness 作为最终结果
        faithfulness_score = manual_result.get("faithfulness", 0)

    # 4. 达标检查
    print("\n" + "=" * 60)
    if faithfulness_score >= 0.85:
        print(
            f"🎉 恭喜！Naive RAG faithfulness = {faithfulness_score:.3f} ≥ 0.85，达标！"
        )
    else:
        print(f"⚠️ Naive RAG faithfulness = {faithfulness_score:.3f} < 0.85，需要优化")

    # 5. Advanced RAG 对比评估
    print("\n" + "=" * 60)
    print("📊 Advanced RAG 对比评估")
    print("=" * 60)

    from src.advanced_rag import AdvancedRAG

    print("\n📝 生成 Advanced RAG 回答...")
    advanced = AdvancedRAG(embedder, llm, rag)
    advanced_dataset = [d.copy() for d in EVAL_DATASET]
    for item in advanced_dataset:
        result = advanced.query_with_rewrite(item["question"])
        item["contexts"] = [doc["content"] for doc in result["retrieved_docs"]]
        item["answer"] = result["answer"]
        print(f"  Q: {item['question']}")
        print(f"  A: {item['answer'][:100]}...")
        print()

    # 评估 Advanced RAG
    print("\n📊 评估 Advanced RAG...")
    advanced_result = manual_evaluation(advanced_dataset)
    print(f"  Answer Relevancy: {advanced_result['answer_relevancy']:.3f}")
    print(f"  Context Precision: {advanced_result['context_precision']:.3f}")
    print(f"  Faithfulness: {advanced_result['faithfulness']:.3f}")

    # 保存 Advanced RAG 结果
    with open("docs/rag-advanced-eval.json", "w", encoding="utf-8") as f:
        json.dump(advanced_result, f, ensure_ascii=False, indent=2)

    # 6. 对比结果
    print("\n" + "=" * 60)
    print("📈 Naive vs Advanced RAG 对比")
    print("=" * 60)

    naive_score = faithfulness_score
    advanced_score = advanced_result["faithfulness"]

    print(f"\n  {'指标':<20} {'Naive':<10} {'Advanced':<10} {'提升':<10}")
    print(f"  {'-'*50}")
    print(
        f"  {'Faithfulness':<20} {naive_score:<10.3f} {advanced_score:<10.3f} {advanced_score-naive_score:+.3f}"
    )
    print(
        f"  {'Answer Rel':<20} {manual_result['answer_relevancy']:<10.3f} {advanced_result['answer_relevancy']:<10.3f} {advanced_result['answer_relevancy']-manual_result['answer_relevancy']:+.3f}"
    )
    print(
        f"  {'Context Prec':<20} {manual_result['context_precision']:<10.3f} {advanced_result['context_precision']:<10.3f} {advanced_result['context_precision']-manual_result['context_precision']:+.3f}"
    )

    if advanced_score > naive_score:
        print(
            f"\n  🎉 Advanced RAG 在 Faithfulness 上提升了 {(advanced_score-naive_score)*100:.1f}%"
        )
    elif advanced_score < naive_score:
        print(
            f"\n  ⚠️ Advanced RAG 在 Faithfulness 上下降了 {(naive_score-advanced_score)*100:.1f}%"
        )
    else:
        print("\n  📊 两者在 Faithfulness 上持平")


if __name__ == "__main__":
    main()
