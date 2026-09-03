"""
图谱桥接：从图表描述中提取实体名，调用项目 B（scholar_knowledge）的图谱查询
作用：图表问答时附带图谱佐证，如"图中提到的 ResNet，其作者何恺明的其他论文还有…"
"""

import re
import sys
from pathlib import Path

# 指向项目 B 的 src 目录（仓库根/scholar_knowledge/src）
SCHOLAR_SRC = Path(__file__).resolve().parents[2] / "scholar_knowledge" / "src"
sys.path.insert(0, str(SCHOLAR_SRC))

from graph_builder import ScholarGraph  # noqa: E402


class GraphBridge:
    def __init__(self):
        try:
            # ScholarGraph 构造时自动加载 scholar_data.json 并建图，无单独 load() 方法
            self.graph = ScholarGraph()
            print("🔗 已连接学术知识图谱")
        except Exception as e:
            self.graph = None
            print(f"⚠️ 图谱连接失败（将降级为纯视觉问答）: {e}")

    def extract_entities(self, description: str) -> list:
        """从描述中提取实体名：优先【实体】标签行，缺失时兜底全文匹配图谱节点名

        GLM-4V 输出格式不稳定，有时【实体】标签会丢失（如输出成裸的 "4. A, B, C"），
        此时直接把图谱节点名（及其首词）拿去和描述文本做包含匹配。
        """
        m = re.search(r"【实体】(.+)", description)
        if m and "无" not in m.group(1):
            raw = m.group(1)
            # 按 / 、 , ； 等常见分隔符切分
            return [
                x.strip()
                for x in re.split(r"[/、,;；]", raw)
                if x.strip() and len(x.strip()) > 1
            ]
        return self._match_graph_nodes(description)

    def _match_graph_nodes(self, text: str, limit: int = 3) -> list:
        """用图谱节点名反向扫描描述文本（兜底实体提取）"""
        if not self.graph:
            return []
        t = text.replace(" ", "").lower()
        found: list = []
        try:
            for nid in self.graph.G.nodes:
                name = self.graph.get_entity_name(nid)
                # 节点名本身或其首词（适配长论文标题）出现在文本中即命中
                for cand in (name, name.split(":")[0].strip()):
                    key = cand.replace(" ", "").lower()
                    if len(key) > 1 and key in t:
                        found.append(name)
                        break
                if len(found) >= limit:
                    break
        except Exception:
            return []
        return found

    def enrich(self, description: str, top_k: int = 3) -> str:
        """返回图谱补充上下文（找不到就返回空字符串）"""
        if not self.graph:
            return ""
        entities = self.extract_entities(description)[:top_k]
        if not entities:
            return ""

        facts, seen = [], set()
        for name in entities:
            try:
                # 真实接口：find_entity_by_name 模糊匹配返回 id 列表
                ids = self.graph.find_entity_by_name(name)
                if not ids:
                    continue
                nid = ids[0]  # 模糊匹配取第一个（最相近）
                # get_neighbors 返回 (邻居id, 关系) 列表
                nbrs = self.graph.get_neighbors(nid)[:5]
                if nbrs:
                    for v, rel in nbrs:
                        line = (
                            f"- 图谱佐证: {self.graph.get_entity_name(nid)} —[{rel}]→ "
                            f"{self.graph.get_entity_name(v)}"
                        )
                        if line not in seen:
                            seen.add(line)
                            facts.append(line)
                else:
                    # 关键词/会议等节点无出边，查反向边（如"某论文 —[包含]→ 该关键词"）
                    for v in list(self.graph.G.predecessors(nid))[:5]:
                        attrs = self.graph.G.get_edge_data(v, nid)
                        rel = (
                            next(iter(attrs.values())).get("relation", "关联")
                            if attrs
                            else "关联"
                        )
                        line = (
                            f"- 图谱佐证: {self.graph.get_entity_name(v)} —[{rel}]→ "
                            f"{self.graph.get_entity_name(nid)}"
                        )
                        if line not in seen:
                            seen.add(line)
                            facts.append(line)
            except Exception:
                continue
        return "\n".join(facts) if facts else ""


if __name__ == "__main__":
    bridge = GraphBridge()

    # 案例 1：命中路径——实体来自 test_fig3.png 的真实视觉识别结果，ReAct/CoT 在图谱中
    hit_desc = """1. 【类型】柱状图
2. 【内容】HotpotQA EM 分数：8b/62b/540b 三档模型下对比 Standard/CoT/Act/ReAct 四种方法
3. 【结论】ReAct 在 540b 微调后达到约 32 EM
4. 【实体】HotPotQA、ReAct、CoT、Act"""
    print("实体提取:", bridge.extract_entities(hit_desc))
    print("图谱补充:")
    print(bridge.enrich(hit_desc) or "（图谱中无相关实体）")

    # 案例 2：未命中路径——ResNet/ImageNet 不在图谱覆盖内，应诚实降级
    miss_desc = """1. 【类型】折线图
2. 【内容】ResNet-50 在 ImageNet 上达到 76%
4. 【实体】ResNet、ImageNet"""
    print("\n实体提取:", bridge.extract_entities(miss_desc))
    print("图谱补充:", bridge.enrich(miss_desc) or "（图谱中无相关实体）")
