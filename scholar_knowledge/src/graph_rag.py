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

    def retrieve_graph_evidence(self, query: str, max_evidence: int = 40) -> list[str]:
        """从图谱检索证据（实体定位 + 邻居扩展 + 引用 2 跳扩展）。

        改进：
          - 动态扫描图谱所有实体名做匹配，不依赖硬编码关键词列表
          - 同名实体优先论文类型（避免"ReAct"命中关键词节点而非论文节点）
          - evidence 同时取出边（出）和入边（被引/被 authored），
            覆盖"XX 论文引用了哪些论文"等入边问题
          - 论文实体额外做引用 2 跳扩展（p6→p3→p1），支撑"引用链"类多跳问题
          - evidence 标注实体类型，让 LLM 区分同名不同类型实体
        """
        evidence: list[str] = []
        query_lower = query.lower()

        # 1. 定位实体：动态匹配图谱所有实体名，按类型优先级排序（论文 > 作者 > 关键词 > 会议）
        #    双向匹配：正向"实体名出现在问题里" + 反向"问题英文短词命中实体名首词"
        #    （"BERT" → "BERT: Pre-training..."），解决评测模板用短名、
        #    图谱存全标题导致的匹配盲区（bad_cases.md 数据失败 6 条的根因）
        type_priority = {"论文": 0, "作者": 1, "关键词": 2, "会议": 3}
        matched: list[tuple[int, str]] = []
        # 提取问题中的英文 token（含连字符，≥4 字符），用于反向前缀匹配
        q_tokens = set(re.findall(r"[a-zA-Z][a-zA-Z-]{3,}", query_lower))
        for nid, name in self._entity_names:
            if not name:
                continue
            name_lower = name.lower()
            hit = name_lower in query_lower
            if not hit and q_tokens:
                # 反向：实体名首个单词与问题 token 做前缀匹配
                first_word = re.sub(r"[^\w-]", "", name_lower.split()[0])
                hit = any(first_word.startswith(t) for t in q_tokens)
            if hit:
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
        hop2_done: set[str] = set()  # 已做 2 跳扩展的论文，避免重复
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
                # 引用 2 跳扩展：论文 A -引用-> B 时，补充 B 的引用对象（A→B→C）
                if (
                    rel == "引用"
                    and v_type == "论文"
                    and v not in hop2_done
                    and v not in kept_ids
                ):
                    hop2_done.add(v)
                    for _, v2, attrs2 in self.graph.G.out_edges(v, data=True):
                        if attrs2.get("relation") == "引用":
                            evidence.append(
                                f"  -引用(2跳)-> [{self.graph.G.nodes[v2].get('type', '')}]"
                                f" {self.graph.get_entity_name(v2)}"
                            )
            # 入边：u -rel-> eid（用户问"XX 引用了谁"时，被引方向是入边）
            for u, _, attrs in self.graph.G.in_edges(eid, data=True):
                rel = attrs.get("relation", "")
                u_type = self.graph.G.nodes[u].get("type", "")
                evidence.append(
                    f"  <-{rel}- [{u_type}] {self.graph.get_entity_name(u)}"
                )
                # 关键词实体的入边论文做 2 跳扩展：kw_react ← p3 时，
                # 补充 p3 的引用/关键词边（问题里通常只有短词，论文全名匹配不上，
                # 若不扩展则 p3 的引用链永远进不了证据）
                if (
                    node_type == "关键词"
                    and u_type == "论文"
                    and u not in hop2_done
                    and u not in kept_ids
                ):
                    hop2_done.add(u)
                    for _, v2, attrs2 in self.graph.G.out_edges(u, data=True):
                        rel2 = attrs2.get("relation", "")
                        if rel2 in ("引用", "关键词"):
                            evidence.append(
                                f"  -{rel2}(2跳)-> [{self.graph.G.nodes[v2].get('type', '')}]"
                                f" {self.graph.get_entity_name(v2)}"
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

        # 3. 构造 prompt 给 LLM（严格基于证据 + 强制引用原文实体名/关系名）
        prompt = f"""你是学术知识图谱问答助手。请严格遵守以下规则回答问题：
1. 只使用【图谱证据】中出现的信息，禁止使用你自己的知识补充图谱中没有的论文、作者或关系。
2. 回答中必须**逐字引用**证据里出现的实体名（论文标题/作者名/关键词原文）和关系名（如"引用"、"作者"、"关键词"）。
3. 问"哪些/有哪些/涉及哪些"时，把证据中列出的相关实体**全部列出**，不要遗漏、不要缩写（用证据中的完整名称）。
4. 如果证据不足以回答，明确说明"图谱中缺乏相关信息"。

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
