# 第 3 周复盘

📅 **日期**: 2026-07-20 ~ 2026-07-26
📊 **主题**: RAG（检索增强生成）从零到进阶

---

## 1. 本周学到的最重要的 3 个概念

### 1.1 RAG 完整链路（索引→检索→生成）

```
┌─────────────────────────────────────────────────────────────────┐
│                        RAG 完整流程                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  📚 索引阶段                                                       │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐    │
│  │ 文档加载     │───▶│ 文本切分     │───▶│ Embedding + 向量存储  │    │
│  │ (PDF/TXT/...)│    │ (固定/递归)  │    │ (Chroma/Faiss/...)   │    │
│  └─────────────┘    └─────────────┘    └─────────────────────┘    │
│                                                                   │
│  🔍 检索阶段                                                       │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐    │
│  │ Query 处理   │───▶│ 相似度匹配   │───▶│ 返回 Top-K 文档块    │    │
│  │ (改写/HyDE)  │    │ (Cosine/...) │    │                     │    │
│  └─────────────┘    └─────────────┘    └─────────────────────┘    │
│                                                                   │
│  ✨ 生成阶段                                                       │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │ Prompt = System + Context + Question                      │     │
│  │ Response = LLM(Prompt)                                    │     │
│  └─────────────────────────────────────────────────────────┘     │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

**关键理解**：
- RAG 不是"外挂"，而是 LLM 的 **实时知识增强**
- 索引是离线的（可以预先构建），检索和生成是在线的
- 每个环节都有优化空间：切分策略、检索算法、Prompt 设计

### 1.2 切块策略对检索效果的影响

| 策略 | 描述 | 优点 | 缺点 |
|-----|------|-----|------|
| **固定大小** | 500字/块，50字重叠 | 简单可控 | 切断语义 |
| **句子切分** | 按句号分割 | 语义完整 | 块数多 |
| **段落切分** | 按空行分割 | 保留结构 | 块大小不均 |
| **递归切分** | 优先段落→句子→词 | 智能适配 | 实现复杂 |

**实验发现**：
- 太小（200字）：丢失上下文，检索率低
- 太大（1000字）：噪声多，占用 Token
- **最佳范围**：400-600字 + 10-20% 重叠

### 1.3 高级 RAG 优化（Query 改写 / HyDE / Reranker）

```
基础 RAG 流程：
Query → 检索 → 生成

高级 RAG 流程：
Query ──┬──▶ 改写 ──┐
        ├──▶ HyDE ──┼──▶ 多路检索 ──▶ Rerank ──▶ 生成
        └──▶ Multi ─┘
```

**三种优化技术**：

| 技术 | 原理 | 适用场景 |
|-----|------|---------|
| **Query 改写** | LLM 重写模糊查询 | "他发明的" → "爱迪生发明了什么" |
| **HyDE** | 生成假设答案用于检索 | 知识库结构不明确时 |
| **Reranker** | 对召回结果二次排序 | 提高 Top-K 质量 |

---

## 2. RAG 系统的难点在哪？

### 2.1 各环节难度分析

```
难度排行（主观）：
检索策略 ⭐⭐⭐⭐⭐  ← 最难！
生成控制 ⭐⭐⭐⭐
索引构建 ⭐⭐⭐
评估体系 ⭐⭐⭐⭐
```

### 2.2 检索为什么最难？

| 挑战 | 说明 | 我的解决方案 |
|-----|------|-------------|
| **语义鸿沟** | 用户用词 ≠ 文档用词 | Query 改写、HyDE |
| **局部最优** | 找到局部相关但整体不相关 | MMR、多路召回 |
| **无答案情况** | 知识库没有答案 | 需要检测机制 |
| **粒度不匹配** | Query 太细/太粗 | 调整 chunk_size |

### 2.3 生成的难点

| 挑战 | 说明 | 我的解决方案 |
|-----|------|-------------|
| **幻觉** | 编造不存在的内容 | System Prompt + 引用来源 |
| **上下文冲突** | 多个文档矛盾 | Rerank 提高优先级 |
| **信息压缩** | Token 限制 | 上下文压缩 |

### 2.4 评估的难点

| 挑战 | 说明 | 我的解决方案 |
|-----|------|-------------|
| **指标定义** | 什么是"好"答案？ | Faithfulness + Relevance |
| **Ground Truth** | 需要标注数据 | 手动构建测试集 |
| **自动化** | 每次手动测 | RAGAS / 自定义评估 |

---

## 3. 我的 RAG 系统表现如何？

### 3.1 评估指标

基于 `docs/rag-eval-manual.json` 和 `docs/rag-advanced-eval.json`：

| 指标 | Naive RAG | Advanced RAG | 备注 |
|-----|-----------|--------------|------|
| **Faithfulness** | 1.000 | 1.000 | 都很优秀 |
| **Answer Relevance** | 1.000 | 1.000 | 答案都相关 |
| **Context Precision** | 1.000 | 1.000 | 检索都准确 |

### 3.2 为什么分数这么高？

**原因分析**：
1. **测试集太小**：只有 5 个问题，不够有区分度
2. **知识库匹配**：所有问题在知识库中都有明确答案
3. **问题设计**：都是"标准答案"类型，没有歧义

### 3.3 真实场景会怎样？

| 场景 | Naive | Advanced | 说明 |
|-----|-------|----------|------|
| 精确匹配问题 | 好 | 好 | "Transformer 哪年提出？" |
| 模糊查询 | 差 | 好 | "谁发明的？" ← Query 改写有效 |
| 多跳推理 | 差 | 中 | "XX 的同事是谁？" |
| 无答案问题 | 差 | 差 | 需要检测机制 |

### 3.4 Naive vs Advanced 对比

```
Naive RAG（基础）：
Query → Embedding → Chroma.retrieve → LLM.generate
✅ 简单直接
❌ 模糊查询表现差

Advanced RAG（优化）：
Query → [改写/HyDE/Multi] → 多路检索 → Rerank → LLM.generate
✅ 模糊查询表现好
❌ 延迟增加（多调用几次 LLM）
❌ 成本更高
```

**结论**：当前测试集无法拉开差距，但 Advanced RAG 在真实场景（模糊查询）中会有明显优势。

---

## 4. 本周代码统计

### 4.1 Commit 记录

| 日期 | Commit | 说明 |
|-----|--------|------|
| 07-26 | be0cb0e | feat: 高级 RAG（Query 改写 + HyDE + Reranker） |
| 07-26 | 4b679eb | feat: Chroma 向量库 + RAGAS 评估 |
| 07-25 | bb9e08b | feat: 切块策略对比实验 |
| 07-24 | eeebb95 | feat: Embedding 客户端 + Naive RAG 骨架 |

**总计**：4 个 Commit

### 4.2 代码量统计

```
src/ 目录 Python 文件：
├── advanced_rag.py      (228行)  ← 本周新增
├── vector_rag.py         (193行)  ← 本周新增
├── rag_evaluation.py    (285行)  ← 本周新增
├── chunking_experiment.py (130行) ← 本周新增
├── embedder.py            (97行)  ← 本周新增
├── llm_client.py          (80行)  ← 上周
├── ...其他文件

总计：2178 行 Python 代码
本周新增：约 933 行
```

### 4.3 文件统计

| 类别 | 文件数 | 说明 |
|-----|--------|------|
| 核心模块 | 4 | embedder, llm_client, vector_rag, advanced_rag |
| 实验脚本 | 2 | chunking_experiment, rag_evaluation |
| 工具类 | 1 | naive_rag |
| **总计** | **7** | 本周新增 |

---

## 5. 下周（Agent 核心）计划

### 5.1 学习目标

基于本周 RAG 的基础，下周重点学习 Agent：

| 主题 | 目标 | 难度 |
|-----|------|-----|
| **ReAct 模式** | Thought → Action → Observation 循环 | ⭐⭐⭐ |
| **Tool Use** | 让 LLM 调用外部工具 | ⭐⭐⭐⭐ |
| **Function Calling** | OpenAI/Alibaba 的函数调用机制 | ⭐⭐⭐ |
| **Agent 框架** | LangChain Agent / AutoGen | ⭐⭐⭐⭐ |
| **记忆机制** | Short-term / Long-term memory | ⭐⭐⭐ |

### 5.2 具体任务

#### 任务 1：实现 ReAct Agent
```python
# 伪代码
class ReActAgent:
    def run(self, query):
        while not done:
            thought = llm.think(query, observations)
            action = llm.decide_action(thought)
            observation = tools.execute(action)
            observations.append(observation)
        return llm.final_answer()
```

#### 任务 2：集成到现有 RAG
```python
# Agent + RAG = 更强的问答系统
class RAGAgent(Agent):
    def __init__(self, rag_system):
        self.tools = [
            Tool(name="SearchKB", func=rag.retrieve),
            Tool(name="WebSearch", func=web.search),
            Tool(name="Calculator", func=calc),
        ]
```

#### 任务 3：多 Agent 协作（进阶）
```python
# 多 Agent 分工
ResearcherAgent → 收集资料
WriterAgent     → 撰写文章
ReviewerAgent   → 审核修改
```

### 5.3 推荐资源

| 资源 | 类型 | 链接 |
|-----|------|------|
| LangChain Agent 教程 | 文档 | https://python.langchain.com/docs/tutorials/agents/ |
| ReAct 论文 | 论文 | https://arxiv.org/abs/2210.03629 |
| LLM Agent 综述 | 博客 | https://lilianweng.github.io/posts/2023-06-23-agent/ |
| AutoGen | 框架 | https://github.com/microsoft/autogen |

### 5.4 里程碑检查点

- [ ] **W3-Mon**: 理解 ReAct 模式，实现简单 Agent
- [ ] **W3-Wed**: 集成 Tool Use，支持 2-3 个工具
- [ ] **W3-Fri**: 与 RAG 结合，实现 RAG-Agent
- [ ] **W3-Sun**: 多 Agent 协作 Demo（如果时间够）

---

## 📝 本周感悟

### 做得好的地方 ✅
1. **从 0 到 1 完整实现了 RAG**，不是只会用框架
2. **做了对比实验**，有数据支撑（虽然样本小）
3. **代码质量不错**，都有错误处理和文档
4. **学习了官方实现**，知道差距在哪

### 可以改进的地方 ⚠️
1. **测试集太小**：应该扩充到 20-30 个问题
2. **缺少真实评估**：应该用 RAGAS 而不是手动
3. **没有 Profiling**：不知道每个环节耗时多少
4. **缺少监控**：应该记录每次查询的效果

### 最大收获 🎯
> **手写一遍 RAG，比看 10 篇文章都有用。**
>
> 只有自己实现了，才知道"就这？"的地方其实暗藏玄机——为什么 chunk_size 是 500 不是 300？为什么要 overlap？为什么用 Cosine 不是 Euclidean？
>
> 这就是 **Tinker** 的价值。

---

## 📊 下周目标

```
Week 3: 理解 RAG 原理 → ✅ 已完成
Week 4: 掌握 Agent 核心 → 🔄 进行中

Week 4 目标：
├── ReAct 模式 ✅→ 🔄
├── Tool Use ✅→ 🔄
├── RAG + Agent ✅→ 🔄
└── Multi-Agent ✅→ ⏳
```

加油！继续保持每天学习的节奏 💪
