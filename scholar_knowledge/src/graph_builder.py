"""学术知识图谱构建（NetworkX）"""

import json
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt


class ScholarGraph:
    """学术知识图谱"""

    def __init__(self, data_path="scholar_knowledge/data/scholar_data.json"):
        self.data = json.loads(Path(data_path).read_text(encoding="utf-8"))
        self.G = nx.MultiDiGraph()
        self._build()

    def _build(self):
        """构建图谱"""
        # 添加节点
        for e in self.data["entities"]:
            self.G.add_node(e["id"], type=e["type"], name=e["name"])
        # 添加边
        for u, rel, v in self.data["relations"]:
            self.G.add_edge(u, v, relation=rel)
        print(
            f"✅ 图谱构建完成：{self.G.number_of_nodes()} 节点, {self.G.number_of_edges()} 边"
        )

    # ===== 查询能力 =====

    def get_entity_name(self, entity_id: str) -> str:
        """根据 id 取实体名"""
        return self.G.nodes[entity_id].get("name", entity_id)

    def find_entity_by_name(self, name: str) -> list[str]:
        """按名字找实体 id（模糊匹配）"""
        return [
            nid
            for nid, attrs in self.G.nodes(data=True)
            if name.lower() in attrs["name"].lower()
        ]

    def get_neighbors(
        self, entity_id: str, relation: str | None = None
    ) -> list[tuple[str, str]]:
        """获取邻居（可限定关系类型），返回 (邻居id, 关系)"""
        result = []
        for _, v, attrs in self.G.out_edges(entity_id, data=True):
            if relation is None or attrs.get("relation") == relation:
                result.append((v, attrs.get("relation")))
        return result

    def multi_hop(self, start_id: str, max_hops: int = 2) -> list[list[str]]:
        """多跳路径搜索，返回所有长度 ≤ max_hops 的路径"""
        paths = []
        for target in self.G.nodes():
            if target == start_id:
                continue
            for path in nx.all_simple_paths(self.G, start_id, target, cutoff=max_hops):
                paths.append(path)
        return paths

    def co_author(self, author1: str, author2: str) -> bool:
        """两个作者是否共著。

        MultiDiGraph 同一对节点可能有多条平行边（不同 relation），
        需遍历所有 key 检查 relation 属性，不能直接用 has_edge(relation=...)。
        """
        a1 = self.find_entity_by_name(author1)
        a2 = self.find_entity_by_name(author2)
        if not a1 or not a2:
            return False
        for u, v in [(a1[0], a2[0]), (a2[0], a1[0])]:
            edges = self.G.get_edge_data(u, v, default={})
            if any(attrs.get("relation") == "共著" for attrs in edges.values()):
                return True
        return False

    def cited_count(self, paper_id: str) -> int:
        """某论文被引次数（仅统计 relation='引用' 的入边）"""
        return sum(
            1
            for _, _, attrs in self.G.in_edges(paper_id, data=True)
            if attrs.get("relation") == "引用"
        )

    def get_cited_by(self, paper_id: str) -> list[str]:
        """引用了 paper_id 的论文列表（入边方向，relation='引用'）"""
        return [
            u
            for u, _, attrs in self.G.in_edges(paper_id, data=True)
            if attrs.get("relation") == "引用"
        ]

    def most_cited_papers(self, top_n: int = 5) -> list[tuple[str, int]]:
        """被引最多的论文"""
        counts = [
            (pid, self.cited_count(pid))
            for pid in self.G.nodes()
            if self.G.nodes[pid].get("type") == "论文"
        ]
        return sorted(counts, key=lambda x: x[1], reverse=True)[:top_n]

    def author_papers(self, author_id: str) -> list[str]:
        """某作者的所有论文"""
        papers = []
        for src, _, attrs in self.G.in_edges(author_id, data=True):
            if attrs.get("relation") == "作者":
                papers.append(src)
        return papers

    # ===== 可视化 =====

    def visualize(
        self,
        output_path: str = "scholar_knowledge/docs/graph.png",
        max_nodes: int = 50,
        k: float = 2.0,
        figsize: tuple[float, float] = (16, 12),
        node_size: int = 1000,
        label_font_size: int = 7,
        dpi: int = 150,
        seed: int | None = 42,
    ) -> str:
        """图谱可视化（限制节点数避免太乱）。

        Args:
            output_path: 输出 PNG 路径
            max_nodes: 取连接度最高的前 N 个节点构图
            k: spring_layout 最优节点距离；越大越散开，越小越紧凑
            figsize: 画布大小 (宽, 英寸)
            node_size: 节点像素大小
            label_font_size: 节点标签字号；节点多时建议调小
            dpi: 输出分辨率
            seed: 布局随机种子，固定后图布局可复现
        """
        # 中文字体：matplotlib 默认无 CJK 字形，title/label 中文会显示为方框
        plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False

        # 取高连接度子图
        nodes = sorted(self.G.degree, key=lambda x: x[1], reverse=True)[:max_nodes]
        sub = self.G.subgraph([n for n, _ in nodes])

        plt.figure(figsize=figsize)
        # 按类型着色
        color_map = {
            "论文": "#FF6B6B",
            "作者": "#4ECDC4",
            "关键词": "#FFE66D",
            "会议": "#95E1D3",
        }
        colors = [
            color_map.get(self.G.nodes[n].get("type", ""), "#AAAAAA")
            for n in sub.nodes()
        ]
        pos = nx.spring_layout(sub, k=k, iterations=80, seed=seed)
        nx.draw_networkx(
            sub,
            pos,
            node_color=colors,
            node_size=node_size,
            font_size=label_font_size,
            font_color="black",
            arrows=True,
            alpha=0.85,
        )
        plt.title(f"学术知识图谱（Top {max_nodes} 核心节点 / k={k}）", fontsize=13)
        plt.axis("off")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close()
        print(f"✅ 图谱可视化已保存到 {output_path}（max_nodes={max_nodes}, k={k}）")
        return output_path


# ===== 演示 =====
def demo():
    graph = ScholarGraph()

    print("\n=== 基础查询演示 ===")
    # 1. 论文被引（入边方向；p1=Attention 在种子数据里只被引、不引用别人）
    print("论文 p1 (Attention) 被哪些论文引用：")
    for nid in graph.get_cited_by("p1"):
        print(f"   ← {graph.get_entity_name(nid)}")

    # 2. 作者共著
    print("\n作者 Vaswani 是否与 Devlin 共著？", graph.co_author("Vaswani", "Devlin"))

    # 3. 被引最多
    print("\n被引最多的论文：")
    for pid, cnt in graph.most_cited_papers(5):
        print(f"   {graph.get_entity_name(pid)} 被引 {cnt} 次")

    # 4. 多跳路径（两跳：p6 → p3 → p5 或 p6 → p5）
    print("\n从 p6 (Agent 综述) 出发的两跳路径示例：")
    paths = graph.multi_hop("p6", max_hops=2)
    for path in paths[:5]:
        names = [graph.get_entity_name(n) for n in path]
        print(f"   {' → '.join(names)}")

    # 5. 可视化
    graph.visualize()


if __name__ == "__main__":
    demo()
