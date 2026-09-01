"""学术数据采集：论文/作者/引用（内置精简版 + 可选 Semantic Scholar API 扩充）"""

import json
import time
from pathlib import Path

# ===== 精简版学术数据（AI/LLM/Agent 领域演示用）=====
# 论文：title, authors, keywords, venue, cited_count
PAPERS = [
    {
        "id": "p1",
        "title": "Attention Is All You Need",
        "authors": ["Vaswani"],
        "keywords": ["Transformer", "NLP"],
        "venue": "NeurIPS",
        "cited": 100000,
    },
    {
        "id": "p2",
        "title": "BERT: Pre-training of Deep Bidirectional Transformers",
        "authors": ["Devlin"],
        "keywords": ["Transformer", "预训练"],
        "venue": "NAACL",
        "cited": 90000,
    },
    {
        "id": "p3",
        "title": "ReAct: Synergizing Reasoning and Acting in Language Models",
        "authors": ["Yao"],
        "keywords": ["Agent", "ReAct"],
        "venue": "ICLR",
        "cited": 15000,
    },
    {
        "id": "p4",
        "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP",
        "authors": ["Lewis"],
        "keywords": ["RAG", "检索增强"],
        "venue": "NeurIPS",
        "cited": 30000,
    },
    {
        "id": "p5",
        "title": "Chain-of-Thought Prompting Elicits Reasoning in LLMs",
        "authors": ["Wei"],
        "keywords": ["CoT", "推理"],
        "venue": "NeurIPS",
        "cited": 20000,
    },
    {
        "id": "p6",
        "title": "A Survey on Large Language Model based Autonomous Agents",
        "authors": ["Wang"],
        "keywords": ["Agent", "综述"],
        "venue": "Frontiers",
        "cited": 5000,
    },
    {
        "id": "p7",
        "title": "GraphRAG: Unifying Text and Structure for Reasoning over Graphs",
        "authors": ["Edge"],
        "keywords": ["GraphRAG", "知识图谱"],
        "venue": "ICLR",
        "cited": 3000,
    },
    {
        "id": "p8",
        "title": "LoRA: Low-Rank Adaptation of Large Language Models",
        "authors": ["Hu"],
        "keywords": ["LoRA", "微调"],
        "venue": "ICLR",
        "cited": 25000,
    },
]

# 引用关系：p_x 引用了 p_y（箭头被引）
CITATIONS = [
    ("p2", "p1"),
    ("p3", "p1"),
    ("p4", "p1"),
    ("p5", "p1"),  # 都引用了 Attention
    ("p3", "p5"),  # ReAct 引用 CoT
    ("p4", "p1"),  # RAG 引用 Attention
    ("p6", "p3"),
    ("p6", "p5"),  # Agent 综述引用
    ("p7", "p4"),
    ("p7", "p1"),  # GraphRAG 引用
    ("p8", "p1"),  # LoRA 引用
]

# 共著关系（简化：同作者即共著）
CO_AUTHORS = []

# 论文摘要文本（供向量检索 + LLM 生成）
PAPER_TEXTS = {
    "p1": "Attention Is All You Need 提出 Transformer 架构，用自注意力机制取代 RNN，成为大模型的基础。",
    "p2": "BERT 是双向 Transformer 预训练模型，通过掩码语言模型任务学习，在 NLP 任务上大幅刷新记录。",
    "p3": "ReAct 提出推理与行动协同框架，通过 Thought-Action-Observation 循环增强 LLM 的工具使用能力。",
    "p4": "RAG 将参数化生成模型与检索结合，从外部知识库检索文档增强生成，减少幻觉。",
    "p5": "CoT 通过思维链提示，让 LLM 分步推理，显著提升复杂推理任务表现。",
    "p6": "本文系统综述基于 LLM 的自主 Agent，总结其架构、应用与挑战。",
    "p7": "GraphRAG 结合文本与图结构推理，在涉及多跳关系的任务上优于纯文本 RAG。",
    "p8": "LoRA 提出低秩适应方法，冻结原始权重只训练低秩矩阵，实现参数高效微调。",
}


def build_graph_data(
    papers: list[dict] | None = None,
    citations: list[tuple[str, str]] | None = None,
    texts: dict[str, str] | None = None,
) -> dict:
    """构建图谱数据：节点 + 边。

    Args:
        papers: 论文列表，默认使用内置 PAPERS
        citations: 引用关系列表 [(src_id, dst_id)]，默认使用内置 CITATIONS
        texts: 论文摘要文本，默认使用内置 PAPER_TEXTS
    """
    papers = papers if papers is not None else PAPERS
    citations = citations if citations is not None else CITATIONS
    texts = texts if texts is not None else PAPER_TEXTS

    entities: list[dict] = []
    relations: list[tuple[str, str, str]] = []

    # 论文节点
    for p in papers:
        entities.append({"id": p["id"], "type": "论文", "name": p["title"]})
        # 作者节点
        for a in p["authors"]:
            aid = f"author_{a.lower()}"
            entities.append({"id": aid, "type": "作者", "name": a})
            relations.append((p["id"], "作者", aid))  # 论文-作者
        # 关键词节点
        for kw in p["keywords"]:
            kid = f"kw_{kw.lower()}"
            entities.append({"id": kid, "type": "关键词", "name": kw})
            relations.append((p["id"], "关键词", kid))
        # 会议节点
        vid = f"venue_{p['venue'].lower()}"
        entities.append({"id": vid, "type": "会议", "name": p["venue"]})
        relations.append((p["id"], "发表于", vid))

    # 引用关系（论文引用论文）
    for src, dst in citations:
        relations.append((src, "引用", dst))

    # 共著关系（同论文作者间）
    for p in papers:
        for i in range(len(p["authors"])):
            for j in range(i + 1, len(p["authors"])):
                relations.append(
                    (
                        f"author_{p['authors'][i].lower()}",
                        "共著",
                        f"author_{p['authors'][j].lower()}",
                    )
                )

    # 去重实体
    seen: set[str] = set()
    unique_entities: list[dict] = []
    for e in entities:
        if e["id"] not in seen:
            seen.add(e["id"])
            unique_entities.append(e)

    return {"entities": unique_entities, "relations": relations, "texts": texts}


def save_scholar_data(
    output_path: str | Path = Path(__file__).resolve().parent.parent
    / "data"
    / "scholar_data.json",
    enrich: bool = False,
) -> dict:
    """保存图谱数据到 JSON。

    Args:
        output_path: 输出文件路径
        enrich: True 时调用 Semantic Scholar API 扩充数据；False 仅用种子数据
    """
    data = enrich_from_api() if enrich else build_graph_data()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"\n✅ 保存 {len(data['entities'])} 个实体, {len(data['relations'])} 条关系 → {output_path}"
    )

    from collections import Counter

    types = Counter(e["type"] for e in data["entities"])
    print(f"实体类型：{dict(types)}")
    rel_types = Counter(r[1] for r in data["relations"])
    print(f"关系类型：{dict(rel_types)}")
    return data


def collect_from_openalex(query: str, limit: int = 10) -> list[dict]:
    """从 OpenAlex API 检索论文（含摘要/作者/会议/引用列表）。

    OpenAlex 完全免费且无需 API key，限额宽松（100k req/day）。
    一次 search 请求即可拿到 referenced_works，无需二次查询引用。

    Args:
        query: 检索关键词
        limit: 返回论文数上限

    Returns:
        论文原始数据列表；失败返回空列表并打印警告
    """
    import requests

    url = "https://api.openalex.org/works"
    select = (
        "id,title,abstract_inverted_index,authorships,publication_year,"
        "primary_location,cited_by_count,concepts,referenced_works"
    )
    params = {"search": query, "per_page": limit, "select": select}
    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, timeout=15)
            if resp.status_code == 429:
                wait = 2 ** (attempt + 2)
                print(f"⚠️ 限流，{wait}s 后重试（第 {attempt + 1}/3 次）")
                time.sleep(wait)
                continue
            if resp.status_code != 200:
                print(f"⚠️ API 返回 {resp.status_code}：{resp.text[:200]}")
                return []
            works = resp.json().get("results", []) or []
            print(f"✅ 获取 {len(works)} 篇论文（query={query!r}）")
            for w in works[:3]:
                print(
                    f"  {w.get('publication_year')} | {w.get('title', '')[:50]} | 被引{w.get('cited_by_count', 0)}"
                )
            return works
        except requests.RequestException as e:
            print(f"⚠️ 请求异常：{e}")
            time.sleep(2 ** (attempt + 1))
    print("⚠️ 三次重试均失败，返回空列表")
    return []


def _restore_abstract(inverted_index: dict | None) -> str:
    """将 OpenAlex 倒排索引还原为连续文本。

    Args:
        inverted_index: {"word": [position1, position2, ...]} 字典

    Returns:
        还原后的摘要文本；输入为空返回空串
    """
    if not inverted_index:
        return ""
    positions: list[tuple[int, str]] = []
    for word, idxs in inverted_index.items():
        for i in idxs:
            positions.append((i, word))
    positions.sort()
    return " ".join(w for _, w in positions)


KEYWORD_LEXICON = [
    "Transformer",
    "NLP",
    "预训练",
    "Agent",
    "ReAct",
    "RAG",
    "检索增强",
    "CoT",
    "推理",
    "综述",
    "GraphRAG",
    "知识图谱",
    "LoRA",
    "微调",
    "GPT",
    "BERT",
    "Attention",
    "embedding",
    "prompt",
    "fine-tuning",
    "instruction",
    "chain-of-thought",
    "tool use",
    "planning",
    "memory",
    "multimodal",
    "RLHF",
    "alignment",
    "scaling",
    "reasoning",
    "hallucination",
    "LLM",
    "language model",
    "self-attention",
    "encoder",
    "decoder",
]


def _extract_keywords(text: str) -> list[str]:
    """从 title/abstract 文本中按词典匹配关键词。"""
    if not text:
        return []
    text_lower = text.lower()
    matched: list[str] = []
    for kw in KEYWORD_LEXICON:
        if kw.lower() in text_lower and kw not in matched:
            matched.append(kw)
    return matched


def enrich_from_api(extra_queries: list[str] | None = None) -> dict:
    """扩充图谱：调用 OpenAlex API 拿真实论文 + 真实引用关系。

    流程：
      1. 多关键词检索 LLM/Agent 领域论文，按 OpenAlex work ID 去重
      2. 每篇提取作者/关键词/会议节点 + 摘要文本（倒排索引还原）
      3. 用 search 返回的 referenced_works 建立真实引用边（仅在采集集合内）
      4. 合并原 8 篇种子数据，去重

    Args:
        extra_queries: 自定义检索词列表；默认 LLM Agent 领域三组词

    Returns:
        合并后的图谱数据（entities/relations/texts）；API 全失败时降级为种子数据
    """
    queries = extra_queries or [
        "large language model agent",
        "retrieval augmented generation",
        "LLM reasoning chain of thought",
    ]
    print("🔍 调用 OpenAlex API 扩充数据 ...")

    raw_works: list[dict] = []
    seen_wids: set[str] = set()
    for q in queries:
        for w in collect_from_openalex(q, limit=10):
            wid = w.get("id")
            if not wid or wid in seen_wids:
                continue
            seen_wids.add(wid)
            title = (w.get("title") or "").strip()
            if not title:
                continue
            raw_works.append(w)
        time.sleep(2)  # 查询间防限流

    if not raw_works:
        print("⚠️ API 无返回，降级到种子数据")
        return build_graph_data()

    # 转换为内部 PAPERS 格式：id 用 p9, p10, ...；OpenAlex work ID 末段作去重键
    api_papers: list[dict] = []
    wid_to_internal: dict[str, str] = {}
    next_idx = 9
    for w in raw_works:
        wid = w["id"]
        internal_id = f"p{next_idx}"
        next_idx += 1
        wid_to_internal[wid] = internal_id
        abstract = _restore_abstract(w.get("abstract_inverted_index"))
        # venue: primary_location.source.display_name
        venue = "Unknown"
        pl = w.get("primary_location") or {}
        source = pl.get("source") or {}
        venue = (source.get("display_name") or "Unknown").strip() or "Unknown"
        # 作者取姓氏（最后一段），最多 3 位
        authors: list[str] = []
        for a in (w.get("authorships") or [])[:3]:
            name = ((a.get("author") or {}).get("display_name") or "").strip()
            if name:
                authors.append(name.split()[-1])
        authors = [a for a in authors if a]
        if not authors:
            authors = ["Unknown"]
        # 关键词：用 concepts（取 score>0.3 的 display_name），失败回退到词典匹配
        concepts = [
            (c.get("display_name") or "").strip()
            for c in (w.get("concepts") or [])
            if (c.get("score") or 0) > 0.3 and c.get("display_name")
        ]
        concepts = [c for c in concepts if c][:3]
        keywords = concepts or _extract_keywords(
            (w.get("title") or "") + " " + abstract
        )
        if not keywords:
            keywords = ["LLM", "Agent"]
        api_papers.append(
            {
                "id": internal_id,
                "work_id": wid,
                "title": w.get("title", ""),
                "abstract": abstract,
                "authors": authors,
                "keywords": keywords,
                "venue": venue,
                "year": w.get("publication_year"),
                "cited": w.get("cited_by_count", 0),
                "referenced_works": w.get("referenced_works") or [],
            }
        )

    print(
        f"✅ 去重后采集 {len(api_papers)} 篇 API 论文（合并种子 8 篇 → 共 {8 + len(api_papers)} 篇）"
    )

    # 用 search 返回的 referenced_works 直接建引用边（无需二次查询）
    api_citations: list[tuple[str, str]] = []
    for p in api_papers:
        hit = 0
        for ref_wid in p["referenced_works"]:
            dst_internal = wid_to_internal.get(ref_wid)
            if dst_internal and dst_internal != p["id"]:
                api_citations.append((p["id"], dst_internal))
                hit += 1
        print(
            f"  📖 [{p['id']}] {p['title'][:40]} → {len(p['referenced_works'])} 条引用，命中集合内 {hit} 条"
        )

    # 合并到种子数据
    merged_papers = list(PAPERS) + [
        {
            "id": p["id"],
            "title": p["title"],
            "authors": p["authors"],
            "keywords": p["keywords"],
            "venue": p["venue"],
            "cited": p["cited"],
        }
        for p in api_papers
    ]
    merged_citations = list(CITATIONS) + api_citations
    merged_texts = dict(PAPER_TEXTS)
    for p in api_papers:
        merged_texts[p["id"]] = p["abstract"] or p["title"]

    print(f"📊 合并后：{len(merged_papers)} 篇论文，{len(merged_citations)} 条引用边")
    return build_graph_data(merged_papers, merged_citations, merged_texts)


if __name__ == "__main__":
    import sys

    enrich = "--enrich" in sys.argv
    save_scholar_data(enrich=enrich)
    if not enrich:
        print(
            "\n提示：运行 `python data_collector.py --enrich` 调用 Semantic Scholar API 扩充数据"
        )
