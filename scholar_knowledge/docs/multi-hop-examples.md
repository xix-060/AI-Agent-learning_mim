# 多跳问答示例（演示 + 面试讲）

> 所有路径均已通过 `ScholarGraph.multi_hop()` 实测验证。
> Q1-Q5 原版保留，Q6 为新增有答案问题。

## 问题 1：ReAct 通过什么间接关联到 Transformer/Attention？（两跳）

**图谱路径**：
```
p3 (ReAct) -引用-> p5 (CoT) -引用-> p1 (Attention Is All You Need)
```

**答案**：ReAct 引用 CoT，CoT 引用 Attention，所以 ReAct 间接建立在 Transformer 架构之上。

**验证输出**：
```
ReAct: Synergizing Reasoning and Acting in Language Models
  -> Chain-of-Thought Prompting Elicits Reasoning in LLMs
  -> Attention Is All You Need
```

---

## 问题 2：Agent 综述引用链最长到哪？（三跳）

**图谱路径**：
```
p6 (Agent 综述) -引用-> p3 (ReAct) -引用-> p5 (CoT) -引用-> p1 (Attention)
```

**答案**：从 Agent 综述出发，最长可达 3 跳到 Attention Is All You Need，途经 ReAct 和 CoT。

**验证输出**：
```
A Survey on Large Language Model based Autonomous Agents
  -> ReAct: Synergizing Reasoning and Acting in Language Models
  -> Chain-of-Thought Prompting Elicits Reasoning in LLMs
  -> Attention Is All You Need
```

---

## 问题 3：共著网络中心人物是谁？

**查询方式**：按 `relation="共著"` 边的 degree 排序。

**答案**（Top 5）：
| 排名 | 作者 | 共著边数 |
|---|---|---|
| 1 | Zhang | 12 |
| 2 | Gao | 8 |
| 3 | Wang | 6 |
| 4 | Chen | 6 |
| 5 | Li | 6 |

**说明**：共 88 条共著边。Zhang 是共著网络中心人物——OpenAlex 扩充的 30 篇 API 论文里 Zhang 姓作者出现最多，是 LLM Agent 领域的高产合作者。

**复跑命令**：
```python
from collections import Counter
deg = Counter()
for u, v, a in g.G.edges(data=True):
    if a.get("relation") == "共著":
        deg[u] += 1; deg[v] += 1
print(deg.most_common(5))
```

---

## 问题 4：同时涉及 RAG 和 Transformer 的论文有哪些？（无答案案例）

**查询方式**：遍历所有论文节点，找关键词邻居同时含 `RAG` 和 `Transformer` 的论文。

**答案**：**图谱中无同时含这两个关键词的论文**。

**原因**：
- `p4` (RAG 论文) 关键词：`{RAG, 检索增强}` —— 不含 Transformer
- `p1/p2` (Transformer/BERT 论文) 关键词：含 Transformer 但不含 RAG
- 扩充后的 30 篇 API 论文里，concepts 也没有同时命中两词的

**相关替代问题**：ReAct 论文（p3）同时关联 `Agent` 和 `推理` 两个关键词，可作为"一论文跨多主题"的演示。

**复跑命令**：
```python
for pid in [f"p{i}" for i in range(1, 39)]:
    if pid not in g.G: continue
    kws = {g.get_entity_name(v).lower()
           for _, v, a in g.G.out_edges(pid, data=True)
           if a.get("relation") == "关键词"}
    if "rag" in kws and "transformer" in kws:
        print(pid, kws)
```

---

## 问题 5：GraphRAG 引用了 RAG 吗？

**图谱路径**：
```
p7 (GraphRAG) -引用-> p4 (RAG: Retrieval-Augmented Generation for Knowledge-Intensive NLP)
```

**答案**：是的，GraphRAG 论文直接引用了 RAG 论文。同时 p7 还引用了 p1 (Attention)。

**验证输出**：
```
p7 引用 -> p4 (Retrieval-Augmented Generation for Knowledge-Intensive NLP)
p7 引用 -> p1 (Attention Is All You Need)
```

---

## 问题 6：ReAct 和 RAG 共同引用了哪些论文？

**查询方式**：分别取 ReAct(p3) 和 RAG(p4) 的出边引用集合，求交集。

**图谱路径**：
```
p3 (ReAct) -引用-> p1 (Attention)  ← 共同
p4 (RAG)   -引用-> p1 (Attention)  ← 共同
```

**答案**：ReAct 和 RAG **共同引用了 p1 (Attention Is All You Need)**。两条独立的技术路线都建立在 Transformer 架构之上。

**验证输出**：
```
p3 (ReAct) 引用集合: {p5 (CoT), p1 (Attention)}
p4 (RAG)   引用集合: {p1 (Attention)}
共同引用: {p1 (Attention Is All You Need)}
```

**面试讲法**：向量检索只能找"语义相似的论文"，但"两条独立路线汇聚到同一基础论文"这种**汇聚分析**是图算法特有的——它揭示的是研究领域的"共同根源"。

**复跑命令**：
```python
p3_refs = {v for _, v, a in g.G.out_edges("p3", data=True) if a.get("relation") == "引用"}
p4_refs = {v for _, v, a in g.G.out_edges("p4", data=True) if a.get("relation") == "引用"}
common = p3_refs & p4_refs  # {'p1'}
```

---

## 面试讲法建议

1. **从问题 1/2 引出"多跳"概念**：单跳查询（如"ReAct 的作者是谁"）用 SQL/向量库都能做，但"ReAct 间接建立在什么之上"需要图遍历——这就是 GraphRAG 的价值。
2. **问题 3 展示图统计能力**：节点度数排序是图算法的入门级应用，引出后续可扩展 PageRank/社区发现。
3. **问题 4 的"无答案"反而是亮点**：演示了图谱的诚实——不像 LLM 容易幻觉编造，图谱查不到就是查不到。
4. **问题 5 收尾**：GraphRAG 引用 RAG 是技术传承的直接证据，比"读论文读出来的"更有说服力。
5. **问题 6 展示"共同引用"语义**：两条独立路径汇聚到同一基础论文，是图算法特有的"汇聚分析"，向量检索做不到。
