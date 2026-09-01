"""GraphRAG：图谱检索 + LLM 生成"""

import re
import sys
from pathlib import Path

# 项目记忆：跨目录 import 需 sys.path 配置 + # noqa: E402
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))  # 项目根，使 src.* 可导入
sys.path.insert(
    0, str(Path(__file__).resolve().parent)
)  # scholar_knowledge/src，使 graph_builder 可导入

from src.llm_client import LLMClient  # noqa: E402
from src.models import Message, RoleEnum  # noqa: E402
from graph_builder import ScholarGraph  # noqa: E402


class GraphRAG:
    """基于图谱的问答"""

    def __init__(self, graph: ScholarGraph, llm: LLMClient):
        self.graph = graph
        self.llm = llm
        # 预缓存图谱所有实体名，避免每个 query 重算
        self._entity_names: list[tuple[str, str]] = [
            (nid, attrs.get("name", ""))
            for nid, attrs in self.graph.G.nodes(data=True)
            if attrs.get("name")
        ]

    # ===== 图谱检索 =====

    def retrieve_graph_evidence(self, query: str, max_evidence: int = 20) -> list[str]:
        """从图谱检索证据（实体定位 + 邻居扩展）。

        改进：
          - 动态扫描图谱所有实体名做匹配，不依赖硬编码关键词列表
          - 同名实体优先论文类型（避免"ReAct"命中关键词节点而非论文节点）
          - evidence 同时取出边（出）和入边（被引/被 authored），
            覆盖"XX 论文引用了哪些论文"等入边问题
          - evidence 标注实体类型，让 LLM 区分同名不同类型实体
        """
        evidence: list[str] = []
        query_lower = query.lower()

        # 1. 定位实体：动态匹配图谱所有实体名，按类型优先级排序（论文 > 作者 > 关键词 > 会议）
        type_priority = {"论文": 0, "作者": 1, "关键词": 2, "会议": 3}
        matched: list[tuple[int, str]] = []
        for nid, name in self._entity_names:
            if name and name.lower() in query_lower:
                node_type = self.graph.G.nodes[nid].get("type", "")
                matched.append((type_priority.get(node_type, 9), nid))
        matched.sort()
        # 同名只保留最高优先级类型（如"ReAct"同时命中 p3 和 kw_react，只留 p3）
        seen_names: set[str] = set()
        kept_ids: list[str] = []
        for _, nid in matched:
            name = self.graph.G.nodes[nid].get("name", "")
            if name not in seen_names:
                seen_names.add(name)
                kept_ids.append(nid)

        # 2. 从每个实体扩展邻居（同时取出边 + 入边，覆盖"被引"等问题）
        for eid in kept_ids:
            if eid not in self.graph.G:
                continue
            node_type = self.graph.G.nodes[eid].get("type", "")
            evidence.append(
                f"实体[{node_type}: {self.graph.get_entity_name(eid)}] 的关系："
            )
            # 出边：u -rel-> v
            for _, v, attrs in self.graph.G.out_edges(eid, data=True):
                rel = attrs.get("relation", "")
                v_type = self.graph.G.nodes[v].get("type", "")
                evidence.append(
                    f"  -{rel}-> [{v_type}] {self.graph.get_entity_name(v)}"
                )
            # 入边：u -rel-> eid（用户问"XX 引用了谁"时，被引方向是入边）
            for u, _, attrs in self.graph.G.in_edges(eid, data=True):
                rel = attrs.get("relation", "")
                u_type = self.graph.G.nodes[u].get("type", "")
                evidence.append(
                    f"  <-{rel}- [{u_type}] {self.graph.get_entity_name(u)}"
                )

        return evidence[:max_evidence]

    # ===== 结构化查询（把问题转成图谱操作）=====

    def answer_structural(self, query: str) -> str | None:
        """尝试用图谱直接回答（不走 LLM）。"""
        # 被引最多
        if "被引" in query and ("最多" in query or "top" in query.lower()):
            top = self.graph.most_cited_papers(5)
            return "被引最多的论文：\n" + "\n".join(
                f"  {self.graph.get_entity_name(pid)}（{cnt} 次）" for pid, cnt in top
            )

        # 共著判断：匹配问题中的英文姓氏词（首字母大写）
        m = re.findall(r"([A-Z][a-z]+)", query)
        if "共著" in query and len(m) >= 2:
            if self.graph.co_author(m[0], m[1]):
                return f"{m[0]} 和 {m[1]} 有共著关系。"
            return f"{m[0]} 和 {m[1]} 没有直接的共著关系。"

        return None

    # ===== 完整问答 =====

    def query(self, question: str, verbose: bool = False) -> str:
        """完整 GraphRAG 问答流程"""
        # 1. 先尝试结构化回答
        structural = self.answer_structural(question)
        if structural:
            if verbose:
                print("  [图谱结构化查询]")
            return structural

        # 2. 图谱检索证据
        graph_evidence = self.retrieve_graph_evidence(question)
        if verbose:
            print(f"  [图谱证据 {len(graph_evidence)} 条]")

        # 3. 构造 prompt 给 LLM
        prompt = f"""你是一个学术知识图谱问答助手。请基于以下从知识图谱检索到的结构化证据回答问题。
如果证据不足，明确说明"图谱中缺乏相关信息"。

【图谱证据】
{chr(10).join(graph_evidence) if graph_evidence else "（无直接图谱证据）"}

【问题】
{question}

请用简洁的中文回答，并说明依据的图谱关系。"""

        # simple_chat 不支持 temperature 参数，改用 chat() 直接构造 messages
        messages = [
            Message(
                role=RoleEnum.SYSTEM,
                content="你是学术知识图谱问答助手，基于图谱证据回答。",
            ),
            Message(role=RoleEnum.USER, content=prompt),
        ]
        try:
            return self.llm.chat(messages, temperature=0.2).content
        except Exception as e:
            return f"⚠️ LLM 调用失败：{e}\n图谱证据（前 5 条）：\n" + "\n".join(
                graph_evidence[:5]
            )

    def demo_questions(self):
        """演示问题集"""
        questions = [
            "哪些论文被引用最多？",
            "ReAct 论文引用了哪些论文？",
            "RAG 领域有哪些关键词？",
            "Vaswani 和 Devlin 有共著关系吗？",
            "GraphRAG 与哪些论文有引用关联？",
            "Agent 相关的研究涉及哪些关键词？",
        ]
        for q in questions:
            print(f"\n❓ {q}")
            print(f"🤖 {self.query(q, verbose=True)}")
            print("-" * 50)


if __name__ == "__main__":
    graph = ScholarGraph()
    llm = LLMClient()
    rag = GraphRAG(graph, llm)
    rag.demo_questions()
