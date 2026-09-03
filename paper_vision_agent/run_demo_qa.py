"""临时脚本：三张测试图各问 3 个问题，收集真实问答（GIF 素材数据源）

用法:
    conda activate ai-agent
    python paper_vision_agent/run_demo_qa.py

输出:
    paper_vision_agent/test_images/qa_results.json
"""

import json
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "src"))

from agent import PaperVisionAgent  # noqa: E402

QA_PLAN: dict = {
    "1_line_chart.png": [
        "这张图对比了哪些方法？分别用什么颜色表示？",
        "在 540b 模型下，哪种方法表现最好？大约多少分？",
        "ReAct 这种方法出自哪篇论文？它和 Attention 有什么关系？",
    ],
    "2_architecture_transformer.png": [
        "这张图展示的是什么模型的整体架构？",
        "图中左右两个堆叠结构分别是什么？各包含哪些组件？",
        "Transformer 架构的核心创新是什么？知识库里有相关论文吗？",
    ],
    "3_table.png": [
        "这张表格的行和列分别代表什么？",
        "哪个模型在 EN-DE 上的 BLEU 分数最高？具体是多少？",
        "Transformer (base model) 比之前最强的 ConvS2S Ensemble 好在哪里？",
    ],
}


def main() -> None:
    """逐图逐问采集，结果写 JSON。支持 `--only 图片名` 只重采单图（与已有结果合并）。"""
    only = None
    if len(sys.argv) > 2 and sys.argv[1] == "--only":
        only = sys.argv[2]

    results: list = []
    for img, questions in QA_PLAN.items():
        if only and only not in img:
            continue
        agent = PaperVisionAgent(str(BASE / "test_images" / img))
        print(f"\n===== {img} 视觉分析完成，开始问答 =====")
        for q in questions:
            t0 = time.time()
            try:
                a = agent.ask(q)
            except Exception as e:  # 单问失败不中断整体采集
                a = f"[采集失败: {e}]"
            dt = time.time() - t0
            results.append(
                {"image": img, "question": q, "answer": a, "seconds": round(dt, 1)}
            )
            print(f"[{dt:.0f}s] Q: {q}\n    A: {a[:80]}...")

    out = BASE / "test_images" / "qa_results.json"
    if only:  # 单图重采：替换已有结果中该图的条目
        old = json.loads(out.read_text(encoding="utf-8")) if out.exists() else []
        results = [r for r in old if only not in r["image"]] + results
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n完成 {len(results)} 条问答 → {out}")


if __name__ == "__main__":
    main()
