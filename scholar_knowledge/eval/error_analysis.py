"""
错误分析：把评测的 bad case 分类归因，输出 markdown 报告
用法:
    cd scholar_knowledge
    python eval/error_analysis.py --result data/eval_report.json --out eval/bad_cases.md

数据来源（适配 run_eval.py 输出）:
    data/eval_report.json 的 last_details，字段映射:
        q   -> question
        kw  -> expected（list[str]，期望关键词）
        rate-> hit（rate >= 1.0 视为命中，与 full_hit_rate 口径一致）
    run_eval 未保存检索证据，contexts 由当前代码对 bad case 现场重建
    （graph 模式纯图遍历不调用 LLM，零 token 成本；hybrid 模式重建向量证据走 embedding API）

分类逻辑:
    数据失败: 召回上下文为空或过短（图谱里没数据）
    生成失败: 期望关键词全部出现在上下文中（检索到了，模型没用上）
    检索失败: 其余（上下文里没有全部期望关键词）
"""

import json
import argparse
import sys
from pathlib import Path

# 项目记忆：跨目录 import 需 sys.path 配置 + # noqa: E402
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))  # noqa: E402

from graph_builder import ScholarGraph  # noqa: E402
from graph_rag import GraphRAG  # noqa: E402


def load_results(path: str) -> tuple[list[dict], int]:
    """读取 run_eval.py 的输出（data/eval_report.json 的 last_details）。

    Args:
        path: eval_report.json 路径

    Returns:
        (results, missing_evidence)：
        results 为题目级列表，字段统一为
        {"question", "expected"(list[str]), "answer", "hit", "rate"}；
        missing_evidence 为重建证据失败的题数（这些题 contexts 为空，只能粗归因）
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"评测结果不存在: {p}\n先运行评测: python eval/run_eval.py --mode graph"
        )
    data = json.loads(p.read_text(encoding="utf-8"))
    # 兼容两种顶层结构：{"last_details": [...]}（run_eval 原生）或裸 list
    if isinstance(data, dict):
        details = data.get("last_details")
        if not isinstance(details, list):
            raise ValueError(
                f"{p} 中没有 last_details 字段，请先运行 run_eval.py 生成评测结果"
            )
    else:
        details = data

    results: list[dict] = []
    missing_evidence = 0
    for d in details:
        expected = d.get("kw") or []
        if isinstance(expected, str):  # 容错：单关键词写成字符串
            expected = [expected]
        rate = float(d.get("rate", 0.0))
        contexts = d.get("contexts")  # 若未来 run_eval 保存了证据则直接用
        if contexts is None:
            missing_evidence += 1
        results.append(
            {
                "question": d.get("q", ""),
                "expected": expected,
                "answer": str(d.get("answer", "")),
                "hit": rate >= 1.0,
                "rate": rate,
                "contexts": contexts,
            }
        )
    results.sort(key=lambda c: c["rate"])  # 命中率低的排前面，便于看最差 case
    return results, missing_evidence


def rebuild_contexts(results: list[dict], mode: str) -> None:
    """对没有 contexts 的题目现场重建检索证据（不调用 LLM，零 token 成本）。

    Args:
        results: load_results 返回的题目列表（原地补全 contexts）
        mode: "graph"（纯图谱证据）或 "hybrid"（图谱 + 向量证据，需 embedding）
    """
    todo = [c for c in results if c["contexts"] is None]
    if not todo:
        return

    graph = ScholarGraph()
    rag = GraphRAG(graph, llm=None)  # 检索为纯图遍历，不用 llm
    hybrid = None
    if mode == "hybrid":
        try:
            from hybrid_rag import HybridRAG  # noqa: E402
            from src.embedder import Embedder  # noqa: E402

            hybrid = HybridRAG(graph, rag.llm, Embedder())
            if not hybrid.has_vector_index():
                print("[WARN] 向量索引未构建，hybrid 模式降级为纯图谱证据")
                hybrid = None
        except Exception as e:
            print(f"[WARN] hybrid 证据重建不可用（{e}），降级为纯图谱证据")

    print(f"[INFO] 正在为 {len(todo)} 条题目重建检索证据（用于归因，不调用 LLM）...")
    for c in todo:
        try:
            if hybrid is not None:
                graph_ev, vector_ev = hybrid.hybrid_search(c["question"])
                c["contexts"] = graph_ev + vector_ev
            else:
                c["contexts"] = rag.retrieve_graph_evidence(c["question"])
        except Exception as e:
            print(f"  [WARN] 证据重建失败: {c['question'][:30]}... → {e}")
            c["contexts"] = []


def classify_error(case: dict) -> str:
    """按证据与期望关键词的关系归因单条 bad case。

    Args:
        case: 含 expected(list[str]) 与 contexts(list[str]) 的题目 dict

    Returns:
        "数据失败" | "生成失败" | "检索失败"
    """
    contexts = " ".join(case.get("contexts", []) or [])
    expected = case.get("expected", []) or []
    kws = [str(k).replace(" ", "").lower() for k in expected if str(k).strip()]

    # 1) 上下文为空 → 图谱/向量库里没东西，属于数据问题
    if len(contexts.strip()) < 10:
        return "数据失败"

    # 2) 期望关键词全部出现在上下文里 → 检索成功了，模型没用上
    norm = contexts.replace(" ", "").lower()
    if kws and all(k in norm for k in kws):
        return "生成失败"

    # 3) 其余 → 检索没召回全
    return "检索失败"


def main():
    parser = argparse.ArgumentParser(description="评测 bad case 归因分析")
    parser.add_argument(
        "--result", default="data/eval_report.json", help="run_eval 输出路径"
    )
    parser.add_argument("--out", default="eval/bad_cases.md", help="报告输出路径")
    parser.add_argument(
        "--mode",
        choices=["graph", "hybrid"],
        default="graph",
        help="与 run_eval 时一致的评测模式（决定证据重建方式）",
    )
    args = parser.parse_args()

    try:
        results, missing_evidence = load_results(args.result)
    except (FileNotFoundError, ValueError) as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    if not results:
        print("[WARN] 评测结果为空，无 bad case 可分析")
        sys.exit(0)

    if missing_evidence:
        rebuild_contexts(results, args.mode)

    bad = [c for c in results if not c.get("hit")]
    good = [c for c in results if c.get("hit")]
    total = len(results) or 1

    buckets = {"检索失败": [], "生成失败": [], "数据失败": []}
    for c in bad:
        buckets[classify_error(c)].append(c)

    n_ret, n_gen, n_data = (
        len(buckets[k]) for k in ("检索失败", "生成失败", "数据失败")
    )
    fix_map = {
        "检索失败": "调混合权重 / 同义词扩展 / 加查询改写",
        "生成失败": "prompt 强制引用证据 / temperature 降为 0",
        "数据失败": "补论文数据 / 补关系边",
    }

    lines = [
        "# Bad Case 分析报告\n",
        f"- 总样本: {len(results)}",
        f"- 命中: {len(good)}（{len(good) / total * 100:.1f}%，口径 rate>=1.0）",
        f"- 未命中: {len(bad)}",
        f"- 证据来源: run_eval 未保存证据，{missing_evidence} 条由当前代码重建（"
        f"{args.mode} 模式）\n",
        "## 错误分布\n",
        "| 类型 | 数量 | 占错误比 | 修复手段 |",
        "|------|------|---------|---------|",
    ]
    for k in ("检索失败", "生成失败", "数据失败"):
        pct = len(buckets[k]) / len(bad) * 100 if bad else 0
        lines.append(f"| {k} | {len(buckets[k])} | {pct:.0f}% | {fix_map[k]} |")
    lines.append("")

    for k, cases in buckets.items():
        lines.append(f"## {k}（{len(cases)} 条）\n")
        for i, c in enumerate(cases[:20], 1):
            hits = "、".join(str(h) for h in c.get("expected", []))
            lines.append(f"{i}. **Q**: {c.get('question', '')}")
            lines.append(f"   - 期望: {hits}")
            lines.append(f"   - 实际: {c.get('answer', '')[:80]}")
        lines.append("")

    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print(f"报告已写入 {args.out}")
    print(f"当前命中率: {len(good) / total * 100:.1f}%（rate>=1.0 口径）")
    print(f"错误分布 → 检索失败 {n_ret} | 生成失败 {n_gen} | 数据失败 {n_data}")


if __name__ == "__main__":
    main()
