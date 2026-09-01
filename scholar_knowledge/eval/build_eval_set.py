"""构建评测集（目标 100 条：人工标注 8 条 + AI 生成 30 条 + 模板扩充）"""

import json
import re
import sys
from pathlib import Path

# 项目记忆：跨目录 import 需 sys.path 配置 + # noqa: E402
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))  # noqa: E402  scholar_knowledge/src，使 filter_unfair 可导入 graph_builder

from src.llm_client import LLMClient  # noqa: E402
from src.models import Message, RoleEnum  # noqa: E402

# 输出路径：文件相对，无论从哪个 CWD 运行都正确
_EVAL_SET_PATH = Path(__file__).resolve().parent.parent / "data" / "eval_set.json"

# 评测集结构：question + 期望关键词列表（用于命中率计算）
# 注意：kw 必须是答案/图谱证据中会出现的字面文本（实体名/标题子串/关系名），
# 不能用实体 id（如 p1）——证据与答案里只有实体名，没有 id。
BASE_EVAL = [
    # 语义检索类
    {"q": "Transformer 架构的核心创新是什么？", "kw": ["Attention", "自注意力"]},
    {"q": "RAG 技术有什么作用？", "kw": ["检索", "增强"]},
    {"q": "LoRA 是做什么的？", "kw": ["低秩", "微调"]},
    # 关系类
    {"q": "哪些论文被引用最多？", "kw": ["Attention", "被引"]},
    {"q": "ReAct 引用了哪些论文？", "kw": ["Chain-of-Thought", "Attention"]},
    # 多跳类
    {
        "q": "Agent 综述相关的引用链包括哪些？",
        "kw": ["ReAct", "Chain-of-Thought", "Attention"],
    },
    {"q": "GraphRAG 和 RAG 有什么关系？", "kw": ["引用", "Retrieval-Augmented", "RAG"]},
    # 共著类
    {"q": "Vaswani 和 Devlin 共著吗？", "kw": ["没有", "否"]},
]

# 模板扩充素材：论文标题/作者名取自 scholar_data.json 中真实实体，保证 kw 可命中
_PAPERS = [
    "Attention Is All You Need",
    "BERT",
    "Retrieval-Augmented Generation for Knowledge-Intensive NLP",
    "ReAct",
    "Chain-of-Thought",
    "LoRA",
    "GraphRAG",
    "Tree of Thoughts",
    "Toolformer",
    "Self-Consistency",
]
_AUTHORS = ["Vaswani", "Devlin", "Yao", "Lewis", "Wei", "Hu", "Zhang", "Gao"]


def _generate_by_llm(llm: LLMClient, n: int = 30) -> list[dict]:
    """用 LLM 生成 n 条语义性评测题，JSON 解析失败返回空列表。"""
    prompt = f"""请生成 {n} 个关于 LLM 学术领域（Attention/RAG/Agent/微调）的问答测试题。
每个题目需包含：问题 q、2-3 个期望关键词 kw（答案中应出现的字面词，必须是具体术语或论文名片段，
不要用"该方法""这类技术"这类泛指词）。
用 JSON 数组格式输出：[{{"q": "...", "kw": ["...", "..."]}}]
只输出 JSON，不要其他文字。"""
    messages = [
        Message(role=RoleEnum.SYSTEM, content="你是评测题生成器，只输出 JSON 数组。"),
        Message(role=RoleEnum.USER, content=prompt),
    ]
    try:
        response = llm.chat(messages, temperature=0.8).content
        data = _parse_llm_items(response)
        return [
            {"q": str(item["q"]), "kw": [str(k) for k in item["kw"]]}
            for item in data
            if isinstance(item, dict) and item.get("q") and item.get("kw")
        ]
    except Exception as e:
        print(f"⚠️ AI 生成失败（{e}），改用模板扩充")
    return []


def _parse_llm_items(response: str) -> list[dict]:
    """宽容解析 LLM 输出的评测题 JSON（容忍截断/尾缀噪声）。

    先尝试整体解析 [ ... ]；失败则用正则逐个提取 {"q": ..., "kw": [...]} 对象，
    只要能提取出合格条目就不浪费这次生成。

    Args:
        response: LLM 原始输出文本

    Returns:
        解析出的条目列表（可能为空）
    """
    arr = re.search(r"\[.*\]", response, re.DOTALL)
    if arr:
        try:
            data = json.loads(arr.group())
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

    # 整体解析失败：逐个提取对象（容忍数组被截断）
    items: list[dict] = []
    for m in re.finditer(r"\{[^{}]*\"q\"[^{}]*\}", response, re.DOTALL):
        try:
            obj = json.loads(m.group())
            if isinstance(obj, dict) and obj.get("q") and obj.get("kw"):
                items.append(obj)
        except json.JSONDecodeError:
            continue
    return items


def _build_templates() -> list[dict]:
    """生成模板扩充条目（论文×作者交叉组合，保证数量与 kw 可达性）。"""
    templates = [
        {"q": "论文 {title} 的核心思想是什么？", "kw": ["{title}"]},
        {"q": "与 {title} 相关的论文有哪些？", "kw": ["{title}"]},
        {"q": "作者 {author} 参与过哪些论文？", "kw": ["{author}"]},
        {
            "q": "{author} 在 {title} 相关方向做了哪些工作？",
            "kw": ["{author}", "{title}"],
        },
    ]
    items: list[dict] = []
    for t in templates:
        for p in _PAPERS:
            for a in _AUTHORS:
                # q 和 kw 都要 format，否则 kw 留下 "{title}" 字面占位符永远命不中
                items.append(
                    {
                        "q": t["q"].format(title=p, author=a),
                        "kw": [k.format(title=p, author=a) for k in t["kw"]],
                    }
                )
    return items


def _dedup_truncate(items: list[dict], n: int) -> list[dict]:
    """按问题去重并截断到 n 条。"""
    seen: set[str] = set()
    final: list[dict] = []
    for item in items:
        if item["q"] not in seen:
            seen.add(item["q"])
            final.append(item)
    return final[:n]


def filter_unfair(min_reachable: float = 0.5) -> list[dict]:
    """数据驱动去偏：删除图谱证据无法支撑的过偏题，再用模板回填到 100 条。

    可达性判定：kw 是图谱证据文本或结构化答案的子串 → 可达。
    可达比例 < min_reachable 的题（如图谱中不存在的内容型问题）被删除。

    Args:
        min_reachable: 保留阈值（kw 可达比例）

    Returns:
        过滤并回填后的评测集
    """
    from graph_builder import ScholarGraph
    from graph_rag import GraphRAG

    items = json.loads(_EVAL_SET_PATH.read_text(encoding="utf-8"))
    rag = GraphRAG(ScholarGraph(), LLMClient())

    kept: list[dict] = []
    dropped: list[dict] = []
    for item in items:
        q, kws = item["q"], item["kw"]
        structural = rag.answer_structural(q)
        evidence = "\n".join(rag.retrieve_graph_evidence(q))
        reachable = sum(
            1 for k in kws if k in evidence or (structural and k in structural)
        )
        if kws and reachable / len(kws) >= min_reachable:
            kept.append(item)
        else:
            dropped.append({"q": q, "kw": kws, "reachable": reachable})

    print(f"🔍 去偏过滤：保留 {len(kept)} 条，删除过偏题 {len(dropped)} 条")
    for d in dropped[:10]:
        print(f"   ✗ {d['q'][:40]} | kw={','.join(d['kw'])}")
    if len(dropped) > 10:
        print(f"   ... 等共 {len(dropped)} 条")

    final = _dedup_truncate(kept + _build_templates(), 100)
    _EVAL_SET_PATH.write_text(
        json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"✅ 去偏后评测集：{len(final)} 条 → {_EVAL_SET_PATH}")
    return final


def expand_dataset(n: int = 100) -> list[dict]:
    """扩展评测集到 n 条：人工标注 + AI 生成 + 模板扩充，去重后截断。

    Args:
        n: 目标条数

    Returns:
        评测集列表，每项含 q（问题）和 kw（期望关键词）
    """
    llm = LLMClient()
    base = list(BASE_EVAL)
    base.extend(_generate_by_llm(llm))
    base.extend(_build_templates())

    final = _dedup_truncate(base, n)

    _EVAL_SET_PATH.parent.mkdir(parents=True, exist_ok=True)
    _EVAL_SET_PATH.write_text(
        json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"✅ 评测集生成：{len(final)} 条 → {_EVAL_SET_PATH}")
    return final


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="评测集构建/去偏工具")
    parser.add_argument(
        "--filter",
        action="store_true",
        help="对现有评测集做过偏题过滤（图谱证据可达性判定）并回填到 100 条",
    )
    args = parser.parse_args()
    if args.filter:
        filter_unfair()
    else:
        expand_dataset()
