# 学术图谱设计 + NetworkX

## NetworkX 核心
- Graph：无向图；DiGraph：有向图；MultiDiGraph：多重有向边
- add_node(id, **attrs) / add_edge(u, v, **attrs)
- neighbors(G, n) / shortest_path / degree

## 本项目图谱存储
- 节点：论文/作者/关键词/会议（Node attrs 存 type/name）
- 边：作者/关键词/发表于/引用/共著（Edge attrs 存 relation 类型）
- 用 MultiDiGraph 支持多重关系（如一篇论文既是作者又有引用）

## 和 Neo4j 的关系（README 讲生产）
| | NetworkX（本项目） | Neo4j（生产） |
|---|---|---|
| 存储 | 内存 | 磁盘持久化 |
| 查询 | Python API | Cypher（声明式） |
| 规模 | 适合万级节点 | 十亿级 |
| 可视化 | matplotlib/networkx | Neo4j Bloom |
| 适合 | 原型/演示/学习 | 生产 |

## 图谱查询能力（本项目要实现的）
1. 邻居查询：某论文的**作者/关键词/引用**都是谁
2. 多跳：A 引用 B，B 引用 C → A 到 C 是两跳
3. 聚合：某作者论文总数、被引总量（从引用边统计）
4. 共著：两个作者是否有共同论文
