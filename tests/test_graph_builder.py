"""graph_builder.ScholarGraph 查询能力测试。

覆盖 9 个查询方法 + 1 个 bug 回归，确保图谱构建后的所有查询接口可用。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scholar_knowledge.src.graph_builder import ScholarGraph  # noqa: E402

# 复用提交的种子+API 扩充数据（190 实体/364 边）
GRAPH = ScholarGraph(data_path="scholar_knowledge/data/scholar_data.json")


# ===== 实体名查询 =====
def test_get_entity_name_returns_known_name():
    """已知 id 返回对应实体名。"""
    assert GRAPH.get_entity_name("p1") == "Attention Is All You Need"


def test_get_entity_name_unknown_id_fallback():
    """未知 id 回退为 id 本身（不抛 KeyError）。"""
    assert GRAPH.get_entity_name("not_exist") == "not_exist"


def test_find_entity_by_name_fuzzy_match():
    """模糊匹配返回所有命中 id。"""
    hits = GRAPH.find_entity_by_name("attention")
    assert "p1" in hits
    assert len(hits) >= 1


def test_find_entity_by_name_no_match():
    """无命中返回空列表。"""
    assert GRAPH.find_entity_by_name("zzz_no_such_entity_zzz") == []


# ===== 邻居查询 =====
def test_get_neighbors_all_relations():
    """不限定关系时返回所有出边邻居。"""
    neighbors = GRAPH.get_neighbors("p1")
    # p1 至少有 作者/关键词/发表于 三类出边
    assert len(neighbors) >= 3
    assert all(isinstance(n, tuple) and len(n) == 2 for n in neighbors)


def test_get_neighbors_filter_by_relation():
    """限定关系类型时只返回该类邻居。"""
    kw_neighbors = GRAPH.get_neighbors("p1", relation="关键词")
    assert len(kw_neighbors) >= 1
    assert all(rel == "关键词" for _, rel in kw_neighbors)


def test_get_neighbors_no_out_edges():
    """孤立节点（仅被引）出边查询返回空列表。"""
    # p1 在种子数据里只被引、不引用别人，但仍有关键词/作者出边
    # 找一个真正无出边的节点：构造性测试用不存在 id
    assert GRAPH.get_neighbors("not_exist") == []


# ===== 多跳路径 =====
def test_multi_hop_returns_paths():
    """两跳内能找到 p6 → p3 → p1 这类路径。"""
    paths = GRAPH.multi_hop("p6", max_hops=2)
    assert len(paths) > 0
    # 至少一条路径长度=3（起+中+终）
    assert any(len(p) == 3 for p in paths)


def test_multi_hop_self_skipped():
    """起点自身不出现在路径终点。"""
    paths = GRAPH.multi_hop("p1", max_hops=1)
    assert all("p1" not in path[1:] for path in paths)


# ===== 共著查询（修复回归） =====
def test_co_author_no_crash_on_different_authors():
    """不同作者查询不抛 TypeError（bug#1 回归：has_edge relation= 参数）。

    修复前：MultiDiGraph.has_edge(u,v,relation=...) 直接抛 TypeError。
    """
    # 任意两个作者名都不应抛异常
    result = GRAPH.co_author("Vaswani", "Devlin")
    assert isinstance(result, bool)


def test_co_author_unknown_author_returns_false():
    """未知作者返回 False（不抛 IndexError）。"""
    assert GRAPH.co_author("Zzz_Unknown", "Vaswani") is False


# ===== 被引统计 =====
def test_cited_count_p1_attention_highly_cited():
    """p1 (Attention) 被引次数 ≥ 7（种子数据 + API 扩充）。"""
    cnt = GRAPH.cited_count("p1")
    assert cnt >= 7


def test_cited_count_only_counts_citation_edges():
    """cited_count 只统计 relation='引用' 的入边，不含其他关系。

    bug#2 回归：修复前统计所有入边（包括 作者/关键词 等）。
    """
    # 找一个有非"引用"入边的论文，确保不会被误算
    # p1 有大量"引用"入边，但它的非引用入边应来自其他论文的"引用"以外的关系
    # 论文节点本身通常只被"引用"，所以这里验证逻辑：
    # cited_count(p1) 应等于 get_cited_by(p1) 的长度
    assert GRAPH.cited_count("p1") == len(GRAPH.get_cited_by("p1"))


def test_cited_count_unknown_paper_zero():
    """未知论文被引次数为 0。"""
    assert GRAPH.cited_count("not_exist") == 0


# ===== get_cited_by（新增方法） =====
def test_get_cited_by_returns_citing_papers():
    """get_cited_by 返回引用了 p1 的论文 id 列表。"""
    citers = GRAPH.get_cited_by("p1")
    assert "p2" in citers  # BERT 引用 Attention
    assert "p3" in citers  # ReAct 引用 Attention
    assert all(c.startswith("p") for c in citers)


def test_get_cited_by_unknown_paper_empty():
    """未知论文的 get_cited_by 返回空列表。"""
    assert GRAPH.get_cited_by("not_exist") == []


# ===== 被引最多 =====
def test_most_cited_papers_returns_sorted_desc():
    """most_cited_papers 按被引次数降序排列。"""
    top = GRAPH.most_cited_papers(top_n=5)
    assert len(top) == 5
    counts = [c for _, c in top]
    assert counts == sorted(counts, reverse=True)


def test_most_cited_papers_p1_at_top():
    """p1 (Attention) 应在被引榜首（种子数据里被引最多）。"""
    top = GRAPH.most_cited_papers(top_n=1)
    assert top[0][0] == "p1"
    assert top[0][1] >= 7


def test_most_cited_papers_all_paper_type():
    """返回的 id 都应为论文类型（bug#3 回归：KeyError on nodes[pid]['type']）。"""
    top = GRAPH.most_cited_papers(top_n=10)
    for pid, _ in top:
        assert GRAPH.G.nodes[pid].get("type") == "论文"


# ===== 作者论文查询 =====
def test_author_papers_returns_papers():
    """author_papers 返回该作者参与的所有论文 id。"""
    # Vaswani 是 p1 的作者
    vaswani_id = "author_vaswani"
    papers = GRAPH.author_papers(vaswani_id)
    assert "p1" in papers


def test_author_papers_unknown_author_empty():
    """未知作者返回空列表。"""
    assert GRAPH.author_papers("not_exist") == []


# ===== 图谱整体健康度 =====
def test_graph_has_expected_scale():
    """图谱规模符合 SPEC 验收：≥100 实体、≥150 边。"""
    assert GRAPH.G.number_of_nodes() >= 100
    assert GRAPH.G.number_of_edges() >= 150


def test_graph_has_four_entity_types():
    """图谱含 4 类节点：论文/作者/关键词/会议。"""
    types = {attrs.get("type") for _, attrs in GRAPH.G.nodes(data=True)}
    assert {"论文", "作者", "关键词", "会议"}.issubset(types)
