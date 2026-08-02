# Toolformer 论文笔记

**论文**: Toolformer: Language Models Can Teach Themselves to Use Tools
**作者**: Timo Schick, Jane Dwivedi-Yu, Roberto Dessi, et al. (Meta AI)
**发表**: NeurIPS 2023 (Oral)
**arXiv**: https://arxiv.org/abs/2302.04761

---

## 1. 核心思想：让模型在训练阶段学会调用工具

### 问题背景
- LLM 在基本任务上表现差：计算、查资料、翻译
- 已有方案问题：
  - **人工标注成本高**: 需要大量人工标注工具调用
  - **任务特定**: 每个任务需要不同的工具设计

### Toolformer 的创新
**自监督学习**: 让 LLM 自己生成工具调用数据，然后学习使用

```
传统方法: 人工标注 → 微调模型
Toolformer: LLM生成候选 → 过滤有用的 → 自训练
```

### 关键区别：训练时 vs 推理时

| 特性 | Function Calling | Toolformer |
|------|------------------|------------|
| **学习方式** | Prompt 指令 | 模型权重学习 |
| **训练成本** | 零 (推理时) | 需要预训练 |
| **灵活性** | 高 (随时换工具) | 低 (固定工具集) |
| **泛化性** | 依赖 Prompt | 依赖训练数据 |

---

## 2. 方法：3 步自监督训练

### Step 1: 生成候选工具调用

给定原始文本，让 LLM 预测可能需要的工具调用：

```python
# 原始文本
"New York is the capital of [?]"

# LLM 生成候选
candidates = [
    QA("What is the capital of New York?"),
    Search("New York capital"),
    Calculator("...")
]
```

### Step 2: 过滤有用的调用

**核心指标**: 工具调用是否降低了后续 token 的预测损失？

```python
# 假设有无工具调用的损失
loss_without = model.loss("New York is the capital of [?]")
loss_with_QA = model.loss("New York is the capital of [QA(...)] [answer]")

# 如果 loss_with_QA < loss_without，则保留
useful = (loss_with_QA < loss_without)
```

### Step 3: 微调整模型

将有用的工具调用嵌入训练数据，继续预训练：

```python
# 增强后的数据
"New York is the capital of [QA(\"What is the capital of NY?\") → Albany] Albany"

# 训练目标
loss = -log P("Albany" | "New York is the capital of [QA(...)] → ")
```

---

## 3. 支持的工具

| 工具 | 功能 | 示例调用 |
|------|------|----------|
| **Calculator** | 数学计算 | `Calculator(15 * 23)` → 345 |
| **QA** | 问答系统 | `QA("Python 发明者")` → Guido van Rossum |
| **Search** | 搜索引擎 | `Search("AI news 2024")` → [结果列表] |
| **MT** | 机器翻译 | `MT("hello", "zh")` → 你好 |
| **Calendar** | 日期计算 | `Calendar("2024-01-01 + 30 days")` → 2024-01-31 |
| **Wiki** | 维基百科 | `Wiki("Python")` → [摘要] |

### 工具调用格式
```
[ToolName(input) → output]
```

---

## 4. 实验结果

### 模型规模
- 基础模型: **GPT-J 6.7B**
- 对比: GPT-3 (175B), OPT (13B)

### 主要结果

| 任务 | GPT-J | Toolformer | GPT-3 (175B) |
|------|-------|------------|--------------|
| **数学计算** | 30% | **70%** | 60% |
| **事实问答** | 40% | **65%** | 55% |
| **翻译** | 50% | **75%** | 70% |

### 关键发现
1. **小模型变强**: 6.7B Toolformer 超过 175B GPT-3
2. **零样本提升**: 不需要为每个任务设计 Prompt
3. **保留原能力**: 不损失原有的语言生成能力

---

## 5. 和 Function Calling 的对比

### Function Calling (如 ChatGPT)
```python
# 推理时，通过 Prompt 告诉模型可用工具
system_prompt = """
你可以使用以下工具:
- calculator(expression): 计算数学表达式
- search(query): 搜索网络

当你需要使用工具时，输出:
{"name": "calculator", "arguments": {"expression": "2+2"}}
"""

# 模型输出 JSON → 解析 → 执行 → 返回结果
```

### Toolformer
```python
# 训练时，模型学会在特定位置插入工具调用
# 推理时，直接生成工具调用 token
# 不需要特殊 Prompt

# 模型内部已编码工具知识
"结果是 [Calculator(2+2) → 4] 4"
```

### 优劣势对比

| 维度 | Function Calling | Toolformer |
|------|------------------|------------|
| **开发成本** | 低 (Prompt 工程) | 高 (需要训练) |
| **工具灵活性** | 高 (随时添加) | 低 (固定工具集) |
| **推理延迟** | 高 (多次往返) | 低 (单次生成) |
| **学习能力** | 依赖 Prompt | 依赖训练数据 |
| **适用场景** | 快速原型、多工具 | 单一场景、高吞吐 |

---

## 6. 与本项目的关系

### 我们目前的方案
- 用 **Function Calling** 思路 (ReAct 格式)
- 推理时解析 `Thought/Action/Observation`
- 灵活性高，但格式解析脆弱

### Toolformer 思路的启示
1. **训练时学习格式**: 如果有足够数据，可以让模型学会工具调用格式
2. **损失函数设计**: 用"是否帮助预测"作为过滤标准
3. **离线工具库**: 预计算常用工具结果

### 可能的改进方向
```python
# 当前: 推理时解析
# 未来: 训练更稳定的格式

class StableToolformer(CompleteAgent):
    def __init__(self, ...):
        # 预训练时已编码工具知识
        self.tool_embeddings = self.load_tool_embeddings()

    def generate(self, prompt):
        # 模型直接生成带工具调用的文本
        output = self.llm.generate_with_tools(prompt)
        return self.parse_embedded_tools(output)
```

---

## 7. 总结

### Toolformer 的核心贡献
1. **自监督工具学习**: 不需要人工标注
2. **格式嵌入**: 工具调用作为特殊 token
3. **小模型高效**: 6.7B 超过 175B

### 适用场景
- 高吞吐推理服务 (低延迟需求)
- 固定工具集场景
- 有足够训练数据

### 不适用场景
- 需要快速迭代新工具
- 动态工具选择
- 零样本场景

### 参考资源
- 代码: https://github.com/lucidrains/toolformer-pytorch (非官方)
- 论文: https://arxiv.org/abs/2302.04761
- 演示: https://eagle705.github.io/Toolformer/
