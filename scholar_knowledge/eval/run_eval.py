"""评测 runner：计算 GraphRAG/HybridRAG 在评测集上的关键词命中率。

用法：
    python run_eval.py --mode graph --limit 30      # 快速抽样
    python run_eval.py --mode graph                 # 全量 100 条
    python run_eval.py --mode hybrid --limit 20     # 混合模式（需网络 embedding）

命中率定义：每题命中关键词数 / 该题关键词总数，全集取平均（与 hybrid-analysis.md 口径一致）。
"""

import argparse
import json
import sys
import time
from pathlib import Path

# 项目记忆：跨目录 import 需 sys.path 配置 + # noqa: E402
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))  # noqa: E402

from graph_builder import ScholarGraph  # noqa: E402
from graph_rag import GraphRAG  # noqa: E402
from src.llm_client import LLMClient  # noqa: E402

_EVAL_PATH = Path(__file__).resolve().parent.parent / "data" / "eval_set.json"
_REPORT_PATH = Path(__file__).resolve().parent.parent / "data" / "eval_report.json"


def run_eval(mode: str, limit: int | None) -> dict:
    """运行评测并返回汇总指标。

    Args:
        mode: "graph"（纯图谱）或 "hybrid"（向量+图谱）
        limit: 抽样条数，None 表示全量

    Returns:
        汇总 dict：{mode, n, avg_rate, full_hit_rate, elapsed_s}
    """
    items = json.loads(_EVAL_PATH.read_text(encoding="utf-8"))
    if limit:
        items = items[:limit]

    graph = ScholarGraph()
    llm = LLMClient()
    rag: GraphRAG
    if mode == "hybrid":
        from hybrid_rag import HybridRAG  # 延迟导入：graph 模式不依赖 embedding
        from src.embedder import Embedder

        rag = HybridRAG(graph, llm, Embedder())
    else:
        rag = GraphRAG(graph, llm)

    start = time.time()
    details: list[dict] = []
    total_rate = 0.0
    full_hits = 0
    for i, item in enumerate(items, 1):
        q, kws = item["q"], item["kw"]
        try:
            answer = rag.query(q)
        except Exception as e:
            answer = f"ERROR: {e}"
        hits = [k for k in kws if k in answer]
        rate = len(hits) / len(kws) if kws else 0.0
        total_rate += rate
        if kws and len(hits) == len(kws):
            full_hits += 1
        details.append(
            {
                "q": q,
                "kw": kws,
                "hits": hits,
                "rate": round(rate, 3),
                "answer": answer[:200],
            }
        )
        if i % 10 == 0:
            print(f"  进度 {i}/{len(items)}，当前累计命中率 {total_rate / i:.1%}")

    summary = {
        "mode": mode,
        "n": len(items),
        "avg_rate": round(total_rate / len(items), 4),
        "full_hit_rate": round(full_hits / len(items), 4),
        "elapsed_s": round(time.time() - start, 1),
    }

    # 报告落盘：全量覆盖 summary，details 保留最近一次
    report: dict = {}
    if _REPORT_PATH.exists():
        try:
            report = json.loads(_REPORT_PATH.read_text(encoding="utf-8"))
        except Exception:
            report = {}
    report[f"{mode}_{'all' if not limit else limit}"] = summary
    report["last_details"] = details
    _REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(
        f"\n📊 [{mode}] n={summary['n']} 平均命中率 {summary['avg_rate']:.1%}"
        f"（全命中题占比 {summary['full_hit_rate']:.1%}）耗时 {summary['elapsed_s']}s"
        f" → {_REPORT_PATH}"
    )
    return summary


def main() -> None:
    """命令行入口。"""
    parser = argparse.ArgumentParser(description="学术 GraphRAG 评测 runner")
    parser.add_argument(
        "--mode", choices=["graph", "hybrid"], default="graph", help="评测模式"
    )
    parser.add_argument("--limit", type=int, default=None, help="抽样条数（默认全量）")
    args = parser.parse_args()
    run_eval(args.mode, args.limit)


if __name__ == "__main__":
    main()
