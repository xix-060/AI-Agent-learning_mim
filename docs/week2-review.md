# 第 2 周复盘

> 周期：2026-07-13 \~ 2026-07-19
>
> 上周计划：Transformer 基础、Attention 论文、为大模型/Agent 打下基础

***

## 1. 本周学到的最重要的 3 个概念

### 1. Attention 机制（Q/K/V）

Attention 是 Transformer 的灵魂，也是几乎所有现代 LLM 的核心组件。我理解的关键突破点：

- **为什么需要除以 √d\_k**：不是玄学，而是为了控制方差。维度越大，Q·K 点积值越大，softmax 进入梯度消失区域。除以 √d\_k 相当于"归一化"，让梯度健康流动。
- **Q/K/V 分别是什么**：Query（我想找什么）、Key（我是什么标签）、Value（我的内容）。图书馆的类比帮我彻底搞懂了——Q 是搜索词，K 是分类标签，V 是书的内容。
- **为什么用 Multi-Head**：单头只能学一种关注模式（比如语法依赖），多头可以并行捕捉多种关系（语法、语义、指代、情感等），然后拼接融合。

**亲手实现**：用 numpy 从零实现了 Scaled Dot-Product Attention 和 Multi-Head Attention（[attention\_numpy.py](file:///e:/git/AI-Agent-learning_mim/src/attention_numpy.py)），验证了 Q·K^T / √d\_k 的计算流程。

### 2. Function Calling 完整链路

这是从"会用 API"到"能做 Agent"的关键跃迁。6 步链路我可以用自己的话复述：

```
用户提问 → 构造 [问题 + 工具列表(JSON Schema)] → 发给 LLM
    ↓
LLM 返回 {"name": "工具名", "arguments": {...}}
    ↓
解析 JSON → 本地执行工具函数 → 把结果发回 LLM → LLM 生成自然语言回复
```

**核心理解**：LLM 不执行代码，它只输出"调用意图"（JSON），真正的执行在你的 Python 代码里。这就是为什么需要用 JSON Schema 精确描述每个工具的签名——LLM 根据 Schema 来生成正确格式的调用指令。

**踩过的坑**：

- `tool_choice="auto"` 下 LLM 有时会"自作主张"不调用工具，直接编造答案（比如编造"当前时间"）。解决方法是在 system prompt 中**明确要求**必须用工具。
- `else` 分支的缩进错误导致 `SyntaxError`——`if msg.tool_calls:` 的结束位置搞错了。

### 3. 采样参数对生成的影响

通过控制变量实验（[temperature\_experiment.py](file:///e:/git/AI-Agent-learning_mim/src/temperature_experiment.py)），实测了 Temperature 和 Top-P 对生成多样性的影响：

- **Top-P 是主旋钮**：缩小 Top-P（0.1\~0.5）比调 Temperature 更能有效提升多样性。实验中 Top-P=0.1 时多样性最高（40%）。
- **Temperature 不是越大越多样**：呈"倒 U 型"分布，0.3 附近反而比 0.7\~1.5 更有多样性。
- **两者的协同关系**：高 Top-P（1.0）会抵消 Temperature 的随机性，低 Top-P 即使在低 Temperature 下也能产生多样输出。

***

## 2. 最难理解的部分

### Multi-Head Attention 的维度变换细节

虽然理解了"多头并行 → 拼接 → 线性融合"的宏观流程，但具体到维度变换时容易混乱：

```
输入 X: (seq_len, d_model) = (10, 512)

Q = X·W_Q: (10, 512) × (512, 512) = (10, 512)  # 先做线性变换
K = X·W_K: (10, 512) × (512, 512) = (10, 512)
V = X·W_V: (10, 512) × (512, 512) = (10, 512)

拆分: (10, 512) → (10, 8, 64)  # 8 个头，每个 64 维
       reshape: (10, 8, 64) → (8, 10, 64)  # 头移到前面方便并行

每个头做 Attention: (8, 10, 64) → (8, 10, 64)
拼接回去: (8, 10, 64) → (10, 512)

最终输出 = concat(所有头) · W_O: (10, 512) × (512, 512) = (10, 512)
```

容易搞错的地方：

- **什么时候 reshape**：线性变换之后、Attention 计算之前才拆分；Attention 之后再拼回去。
- **是否影响其他层**：Embedding 层和 FFN 的维度不变，只有 Attention 内部拆分再拼回。
- **为什么要拼回 d\_model 而不是保持多维度**：因为残差连接要求输入输出维度相同，否则无法 `X + Attention(X)`。

另外还有一个疑惑：为什么论文选 8 个头而不是 4 或 16？目前理解是工程权衡——头太多每个头维度太小（信息容量不足），头太少又捕捉不够丰富的模式。但这个数值是实验经验值还是有理论依据，还没搞清楚。

***

## 3. Function Calling 你理解了吗？

**用自己的话描述 6 步链路：**

### 第 1 步：构造 Prompt + 工具描述

把用户的问题和工具列表（用 JSON Schema 描述的函数签名）拼成 messages 发给 LLM。工具描述告诉 LLM："你有这几个工具可用，每个工具的名字、用途和参数格式如下。"

```python
messages = [
    {"role": "system", "content": "你是可以调用工具的助手..."},
    {"role": "user", "content": "现在几点了？"},
]
# 同时附带 tools 参数：JSON Schema 格式的工具列表
```

### 第 2 步：LLM 判断是否需要调用工具

LLM 收到问题后，有两种选择：

- 如果判断需要调用工具 → 返回 `tool_calls` 字段，里面包含要调的工具名和参数
- 如果判断不需要（比如闲聊）→ 直接返回自然语言回复

```
返回内容可能是：
{
  "content": null,
  "tool_calls": [{
    "id": "call_abc123",
    "type": "function",
    "function": {
      "name": "get_current_time",
      "arguments": "{}"    ← JSON 字符串，需要 parse
    }
  }]
}
```

### 第 3 步：解析并本地执行工具

从 LLM 返回的 `tool_calls` 中提取函数名和参数，在本地用 Python 执行：

```python
func_name = tool_call.function.name       # "get_current_time"
func_args = json.loads(tool_call.function.arguments)  # {}
result = TOOL_MAP[func_name](**func_args)  # 调用真实函数
```

**关键点**：LLM 不执行代码，你在本地执行。

### 第 4 步：把工具结果返回给 LLM

构造一个 `role: tool` 的 message，把执行结果放进去，连同之前的对话历史一起发回 LLM：

```python
messages.append({
    "role": "tool",
    "tool_call_id": "call_abc123",
    "content": "2026年07月18日 10:30:00 星期五"
})
```

### 第 5 步：LLM 基于工具结果生成自然语言回复

这是第二次调用 LLM。这次 LLM 已经拿到了工具的执行结果，它需要根据结果生成人能理解的话：

```
LLM 返回: "现在是 2026 年 7 月 18 日星期五上午 10:30。"
```

### 第 6 步：输出最终结果给用户

把 LLM 的自然语言回复返回给用户，一次 Function Calling 完成。如果工具可以链式调用（比如"查天气 → 根据天气推荐穿搭"），则需要重复第 2\~5 步。

### 一句话总结

**LLM 是"大脑"（决定调什么工具、怎么组织语言），你的代码是"手脚"（真正执行工具、传递结果）。Function Calling 就是大脑指挥手脚的完整闭环。**

***

## 4. 本周代码统计

### Commit 数：6 次

| # | Commit    | 内容                              |
| - | --------- | ------------------------------- |
| 1 | `4dd1767` | Attention 论文精读 + numpy 实现       |
| 2 | `4bf8a4b` | Transformer 架构 + Qwen2.5 技术报告笔记 |
| 3 | `66b9232` | 本地加载 Qwen2.5-0.5B 实现多轮对话        |
| 4 | `609e117` | 采样参数实验 + 图表分析                   |
| 5 | `14578e7` | Function Calling 实战（Agent 基石）   |
| 6 | `1a695a2` | Cursor AI Coding 实战 - HN 爬虫     |

### 代码行数：约 1,180 行新增 Python

| 分类               | 文件                          | 行数        | 说明                       |
| ---------------- | --------------------------- | --------- | ------------------------ |
| Attention        | `attention_numpy.py`        | 102       | numpy 从零实现 Attention     |
| Transformer      | `chat_local.py`             | 127       | 本地加载 Qwen2.5 实现多轮对话      |
| 采样实验             | `temperature_experiment.py` | 111       | Temperature/Top-P 控制变量实验 |
| 工具函数             | `tools.py`                  | 119       | 可被 LLM 调用的 3 个工具         |
| Function Calling | `function_calling.py`       | 109       | 完整的工具调用 Agent            |
| 爬虫               | `crawler.py`                | 328       | HN 爬虫 + 域名分析             |
| 图表               | `plot_sampling.py`          | 39        | 采样实验结果可视化                |
| 测试               | 3 个 test\_\*.py             | 243       | 单元测试                     |
| **合计**           | **10 个新文件**                 | **1,180** | <br />                   |

### 新增文件：约 30 个

| 类型          | 数量 | 代表文件                                                                  |
| ----------- | -- | --------------------------------------------------------------------- |
| Python 源码   | 10 | attention\_numpy.py, chat\_local.py, crawler.py, function\_calling.py |
| Python 测试   | 3  | test\_crawler.py, test\_function\_calling.py, test\_chat\_local.py    |
| Markdown 笔记 | 16 | Attention 论文笔记、Transformer 架构、采样实验报告、Function Calling 笔记              |
| 数据/图片       | 3  | sampling-experiment-results.json, 2 张架构图                              |

***

## 5. 下周（RAG）想重点关注什么？

### 核心问题

1. **RAG 和 Fine-Tuning 的边界**：什么时候该用检索增强，什么时候该微调模型？两者的效果对比和适用场景是什么？
2. **向量数据库选型**：Chroma / FAISS / Milvus / Pinecone 各有什么特点？本地轻量级场景选哪个？
3. **Embedding 模型**：向量质量如何影响 RAG 效果？本地小模型（如 bge-small）和云端 API（如 text-embedding-3-small）差距多大？

### 具体动手计划

1. 搭建最小可用的 RAG pipeline：文档加载 → 分块 → Embedding → 向量存储 → 检索 → 生成
2. 测试不同 chunk size（128 / 256 / 512 tokens）对检索质量的影响
3. 尝试本地 Embedding 模型 vs 云端 API 的对比
4. 结合 Function Calling，做一个能"查文档 + 回答问题"的 Agent

### 想提前搞清楚的概念

- Embedding 的本质：为什么相近意思的文本在向量空间里距离近？余弦相似度和欧氏距离的区别
- 分块策略：固定长度 vs 语义段落 vs 滑动窗口，各自的优缺点
- Rerank（重排序）：为什么检索后还需要再排序？Cross-Encoder 和 Bi-Encoder 的区别

***

## 附录：本周修复的 Bug

| 文件                          | 问题                             | 修复                       |
| --------------------------- | ------------------------------ | ------------------------ |
| `function_calling.py`       | `if/else` 缩进错误导致 `SyntaxError` | 把步骤 3 放回 `if` 块内         |
| `function_calling.py`       | 缺少 `sys.path` 设置               | 添加项目根目录路径注入              |
| `tools.py`                  | `get_current_time()` 返回值拼接错误   | 正确映射 `weekday()` 到中文     |
| `llm_client.py`             | `chat()` 不支持 `top_p` 参数        | 添加可选参数支持                 |
| `temperature_experiment.py` | 实验 3 结果未保存到 JSON               | 收集 `prompt_results` 一起写入 |
| `models.py`                 | `content` 长度限制 10,000 过小       | 改为 100,000               |
| `context_experiment.py`     | 缺少 `sys.path` 设置               | 添加路径注入                   |
