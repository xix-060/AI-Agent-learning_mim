# 第 2 周复盘

📅 **周期**: Attention → 本地推理 → Function Calling → AI Coding
📊 **代码统计**: 7 个 Commit | 2,796 行代码 | 20 个 Python 文件

---

## 1. 本周学到的最重要的 3 个概念

### 1️⃣ Attention 机制（Q/K/V）

**核心公式**:
```
Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V
```

**理解要点**:
- **Q (Query)**: 当前词"想找什么" → 类比"搜索词"
- **K (Key)**: 每个词"我是什么" → 类比"标签"
- **V (Value)**: 每个词"我携带什么信息" → 类比"内容"
- **缩放因子 √d_k**: 防止点积值过大导致 softmax 梯度消失

**我的实现** ([attention_numpy.py](file:///e:/git/AI-Agent-learning_mim/src/attention_numpy.py)):
```python
def scaled_dot_product_attention(Q, K, V):
    # Q·K^T / √d_k
    scores = np.matmul(Q, K.T) / np.sqrt(K.shape[-1])
    # softmax
    weights = softmax(scores)
    # 加权求和
    output = np.matmul(weights, V)
    return output, weights
```

---

### 2️⃣ Function Calling 完整链路

**6 步调用流程** (我自己总结的):

```
Step 1: 定义工具 Schema (JSON)
        ↓
Step 2: LLM 返回 Tool Call (function_call)
        ↓
Step 3: 解析参数 (JSON Parse)
        ↓
Step 4: 执行工具 (Python 函数)
        ↓
Step 5: 返回结果给 LLM (tool role message)
        ↓
Step 6: LLM 生成最终回答 (基于工具结果)
```

**核心代码** ([function_calling.py](file:///e:/git/AI-Agent-learning_mim/src/function_calling.py)):
```python
# Step 1: 定义工具
tools = [{
    "type": "function",
    "function": {
        "name": "calculator",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string"}
            }
        }
    }
}]

# Step 2: LLM 返回 tool_call
response = llm.chat(messages, tools=tools)

# Step 3-4: 解析并执行
if response.tool_calls:
    tool_call = response.tool_calls[0]
    args = json.loads(tool_call.function.arguments)
    result = calculator(args["expression"])
```

**关键洞察**: Function Calling 本质是 **结构化输出** + **程序执行**，让 LLM 从"只会说话"变成"能动手做事"。

---

### 3️⃣ 采样参数对生成的影响

**三个核心参数**:

| 参数 | 作用 | 高值效果 | 低值效果 |
|------|------|---------|---------|
| **Temperature** | 控制随机性 | 更多样化 | 更确定性 |
| **Top-P** | 核采样 | 保留更多概率质量 | 只选概率最高的 |
| **Top-K** | 限制候选数 | 更多选择 | 只选 K 个最高 |

**实验结论** ([temperature_experiment.py](file:///e:/git/AI-Agent-learning_mim/src/temperature_experiment.py)):
```
创意写作: temperature=0.8, top_p=0.9 → 多样性好
代码生成: temperature=0.2, top_p=0.95 → 准确性高
问答任务: temperature=0.0 → 最确定性
```

**我的推荐**: 默认 temperature=0.7，根据任务调整

---

## 2. 最难理解的部分

### Multi-Head Attention 的维度变换

**问题**: 为什么要拆成多头？维度怎么变的？

```python
# 单头 vs 多头
# 输入: [batch, seq_len, d_model]  d_model=512

# 单头: 直接线性变换
Q = W_q · X  # [batch, seq_len, d_model]

# 多头: 拆分后分别计算
# 假设 num_heads=8, head_dim=64
X_reshaped = reshape(X, [batch, seq_len, 8, 64])
Q = W_q · X_reshaped  # [batch, seq_len, 8, 64]
# 每个头独立做 attention
output_per_head = attention(Q, K, V)  # [batch, seq_len, 8, 64]
# 拼接回去
output = concat(output_per_head)  # [batch, seq_len, 512]
```

**我的理解**:
- **多头让模型关注不同方面**: 有的头关注语法，有的头关注语义
- **不是并行计算**: 是参数不共享，每个头学自己的特征
- **维度变换核心**: `d_model = num_heads × head_dim`

**现在能讲清吗?** ✅ 能，但需要画图辅助

---

## 3. Function Calling 你理解了吗？

### ✅ 我能用自己的话描述 6 步链路

```
用户: "计算 (15 + 27) × 3"
  │
  ▼
[Step 1] 定义工具 Schema
  │ 我注册了一个 calculator 工具，接受 expression 参数
  │
  ▼
[Step 2] LLM 返回 Tool Call
  │ LLM 看到问题，决定调用 calculator
  │ 返回: {"name": "calculator", "arguments": {"expression": "(15+27)*3"}}
  │
  ▼
[Step 3] 解析参数
  │ 把 JSON 字符串转成 Python dict
  │ args = {"expression": "(15+27)*3"}
  │
  ▼
[Step 4] 执行工具
  │ 调用 calculator("(15+27)*3")
  │ 返回结果: "126"
  │
  ▼
[Step 5] 返回结果给 LLM
  │ 添加一条 tool role message
  │ {"role": "tool", "content": "126"}
  │
  ▼
[Step 6] LLM 生成最终回答
  │ LLM 看到工具结果，组织语言
  │ 返回: "计算结果是 126"
```

### 关键代码片段

```python
# 完整的 Function Calling 循环
while True:
    # Step 1-2: LLM 决定是否调用工具
    response = llm.chat(messages, tools=tools)

    if not response.tool_calls:
        # Step 6: 没有工具调用，直接返回回答
        return response.content

    # Step 3-5: 执行工具并返回结果
    for tool_call in response.tool_calls:
        # Step 3: 解析参数
        args = json.loads(tool_call.function.arguments)

        # Step 4: 执行工具
        result = execute_tool(tool_call.function.name, args)

        # Step 5: 添加结果到消息
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": str(result)
        })
```

### 和 ReAct 的对比

| 维度 | ReAct (文本格式) | Function Calling (JSON) |
|------|-----------------|------------------------|
| **格式** | `Action[xxx]` | `{"name": "xxx", "args": {...}}` |
| **解析** | 正则匹配 | 标准 JSON |
| **可靠性** | 容易出错 | 模型训练过，更可靠 |
| **本质** | 让 LLM 输出结构化指令 | 让 LLM 输出可执行代码 |

**一句话总结**: Function Calling = ReAct 的工业化实现

---

## 4. 本周代码统计

### 📊 提交记录（7 个 Commit）

| Date | Commit | 核心产出 |
|------|--------|---------|
| 07-14 | 4dd1767 | Attention 论文精读 + numpy 实现 |
| 07-15 | 4bf8a4b | Transformer 架构 + Qwen2.5 报告 |
| 07-16 | 66b9232 | 本地加载 Qwen2.5-0.5B |
| 07-17 | 609e117 | 采样参数实验 + 图表 |
| 07-18 | 14578e7 | Function Calling 实战 |
| 07-19 | 1a695a2 | Cursor AI Coding - HN 爬虫 |
| 07-20 | 23903d3 | 上下文工程 + 本周复盘 |

### 📁 新增文件

```
src/
├── attention_numpy.py      # Attention numpy 实现
├── chat_local.py           # 本地模型推理
├── temperature_experiment.py # 采样参数实验
├── function_calling.py     # Function Calling 实现
├── crawler.py              # AI Coding 爬虫
├── context_experiment.py   # 上下文工程实验
└── prompt_patterns.py      # 提示工程模式

docs/
├── attention-notes.md      # Attention 笔记
├── transformer-notes.md   # Transformer 笔记
├── temperature-report.md   # 采样参数报告
├── function-calling-notes.md # FC 笔记
└── week2-review.md          # 本文档
```

### 📈 代码量统计

```
src/     : 1,847 行
scripts/ :   29 行
tests/   :  -
─────────────────────
总计     : 2,796 行（含其他周代码）
本周新增 : 约 1,200 行
```

---

## 5. 下周（RAG）想重点关注什么？

### 🎯 核心目标

| 优先级 | 任务 | 预期产出 |
|--------|------|---------|
| ⭐⭐⭐ | **RAG 完整链路** | 从文档加载到回答生成的全流程代码 |
| ⭐⭐⭐ | **向量数据库基础** | Chroma/Pinecone 存储和检索 |
| ⭐⭐ | **Embedding 原理** | 文本向量化的数学直觉 |
| ⭐⭐ | **Chunking 策略** | 切块大小和重叠的影响 |
| ⭐ | **RAG 评估** | Faithfulness、Relevancy 指标 |

### 📚 学习路线

```
Day 1-2: Embedding + 向量检索基础
  ↓ 理解 Cosine Similarity、向量空间
Day 3-4: 完整 RAG Pipeline
  ↓ 文档加载 → 切块 → 向量化 → 存储 → 检索 → 生成
Day 5: 高级 RAG (Query 改写、HyDE)
  ↓ 解决"检索不到"的问题
Day 6-7: RAG 评估 + 工程实践
  ↓ 搭建评测集，量化效果
```

### 💡 想解决的问题

1. **为什么不能直接把文档全塞进 Prompt?**
   → Context Window 有限，需要检索相关片段

2. **Chunk 大小怎么选?**
   → 太小缺上下文，太大会包含噪声

3. **向量检索一定准吗?**
   → 不一定，需要 Query 改写、Rerank 等优化

4. **RAG 和 Fine-tune 怎么选?**
   → RAG 适合新知识，Fine-tune 适合风格/格式

### 🔗 前置知识（本周已掌握）

| 下周需要 | 本周学的 | 关联 |
|---------|---------|------|
| Embedding | Attention 的投影变换 | 都是线性变换 |
| 向量空间 | Q/K/V 的点积 | 都是相似度计算 |
| Prompt 工程 | 提示工程 6 模式 | RAG 的 Prompt 设计 |
| Function Calling | 工具调用 | RAG 作为 Agent 的 Tool |

---

## 📝 本周反思

### ✅ 做得好的地方

1. **手敲代码而非抄库**: Attention 用 numpy 实现，理解更深刻
2. **实验驱动学习**: 采样参数做了对比实验，有数据支撑
3. **AI Coding 实战**: 用 Cursor 写爬虫，体验了 AI 提效

### ⚠️ 需要改进

1. **Multi-Head Attention 理解还不够深**: 维度变换有点绕，下周再画一次图
2. **本地模型推理**: 只跑了 0.5B 小模型，下周试试更大的
3. **笔记可以更结构化**: 参考吴恩达的笔记风格

### 🎯 下周期待

> RAG 是把 LLM 从"通用知识问答"变成"专家系统"的关键技术。
>
> 本周学的 Function Calling，下周可以让 RAG 变成 Agent 的一个 Tool！

---

**文档生成**: 2026-07-20
**下周日标**: RAG Complete Pipeline + Evaluation
