# ReAct 论文精读笔记

📄 **论文**: ReAct: Synergizing Reasoning and Acting in Language Models\
🔗 **链接**: <https://arxiv.org/abs/2210.03629>\
📅 **发表**: 2022年10月

***

## 1. 解决什么问题？

### 问题背景

大型语言模型（LLM）有两种主要工作方式，但都存在缺陷：

| 方式                  | 优点       | 缺点          |
| ------------------- | -------- | ----------- |
| **纯推理 (Reasoning)** | 逻辑清晰，可解释 | 容易幻觉，依赖内部知识 |
| **纯行动 (Acting)**    | 能获取外部信息  | 缺乏规划，盲目行动   |

### ReAct 的解决方案

```
ReAct = Reasoning + Acting (循环交替)

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   推理 (Thought)  ──▶  行动 (Action)                        │
│                                                             │
│        ▲                    │                                │
│        │                    ▼                                │
│        └───── 观察 (Observation) ───┘                        │
│                                                             │
│   推理指导行动，行动反馈观察，观察再指导推理                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**核心思想**：让 LLM 像人类一样"思考-行动-观察"循环工作，而不是一次性输出答案。

***

## 2. 核心循环（必须掌握）

### ReAct 循环流程图

```
                    ┌─────────────┐
                    │   Start     │
                    │   (Question)│
                    └──────┬──────┘
                           │
                           ▼
                  ┌─────────────────┐
         ┌──────▶│   Thought        │◀───────┐
         │       │   (LLM 思考)     │        │
         │       └────────┬──────────┘        │
         │                │                   │
         │                ▼                   │
         │       ┌─────────────────┐         │
         │       │   Action        │         │
         │       │   (执行操作)    │         │
         │       └────────┬──────────┘         │
         │                │                   │
         │                ▼                   │
         │       ┌─────────────────┐         │
         │       │   Observation   │         │
         │       │   (获取结果)    │         │
         │       └────────┬──────────┘         │
         │                │                   │
         │                ▼                   │
         │       ┌─────────────────┐         │
         │       │ 继续?            │         │
         │       │ ├─ Yes: 回到    │         │
         │       │ │  Thought      │         │
         │       │ └─ No: Finish   │         │
         │       └────────┬──────────┘         │
         │                │                   │
         │                ▼                   │
         │       ┌─────────────────┐         │
         │       │   Final Answer  │         │
         │       └─────────────────┘         │
         │                                    │
         └────────────────────────────────────┘
```

### 真实示例

**问题**：科罗拉多造山运动延伸到的区域的海拔范围是多少？

```
Thought 1: 我需要先了解科罗拉多造山运动是什么，以及它延伸到哪些区域。
Action 1: Search["科罗拉多造山运动"]
┌─────────────────────────────────────────────────────────┐
│ Observation 1:                                          │
│ 科罗拉多造山运动（Laramide orogeny）是一次地质过程，      │
│ 形成了北美西部的山脉，包括落基山脉和大平原地区。          │
│ 它从加拿大延伸到墨西哥。                                  │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
Thought 2: 造山运动延伸到了大平原地区。我需要搜索大平原的海拔范围。
Action 2: Search["大平原 海拔范围"]
┌─────────────────────────────────────────────────────────┐
│ Observation 2:                                          │
│ 大平原（Great Plains）是北美中部的广阔平原，             │
│ 海拔约 1,800-2,400 米（6,000-8,000 英尺）。              │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
Thought 3: 根据搜索结果，科罗拉多造山运动延伸到的大平原地区海拔是 1,800-2,400 米。
Action 3: Finish["科罗拉多造山运动延伸到的区域海拔范围约为 1,800-2,400 米。"]
```

### 循环终止条件

| 终止方式          | 说明                            |
| ------------- | ----------------------------- |
| **Finish 动作** | LLM 判断已经有足够信息，输出 Final Answer |
| **达到最大步数**    | 防止无限循环（通常 5-10 步）             |
| **工具执行失败**    | 连续失败时应该终止或跳过                  |

***

## 3. 为什么 ReAct 比纯 CoT 好？

### CoT (Chain of Thought) 是什么？

```python
# CoT 示例
Question: 小明有 5 个苹果，给了小红 3 个，又买了 2 个，现在有几个？

# CoT Prompt:
Let's think step by step:
1. 小明开始有 5 个苹果
2. 给了小红 3 个: 5 - 3 = 2
3. 又买了 2 个: 2 + 2 = 4
Answer: 4
```

### CoT vs ReAct 对比

| 维度       | 纯 CoT    | ReAct         |
| -------- | -------- | ------------- |
| **知识来源** | 仅模型内部知识  | 内部知识 + 外部工具   |
| **处理能力** | 只能处理已有知识 | 能搜索、计算、调用 API |
| **错误处理** | 无法纠正幻觉   | 观察结果能纠正错误     |
| **可解释性** | 推理过程可解释  | 推理 + 行动过程都可解释 |
| **适用场景** | 数学、逻辑推理  | 开放式问题、需要实时信息  |

### ReAct 的优势

```python
# 示例：需要实时信息的问题
Question: "今天北京的空气质量如何？"

# 纯 CoT（失败）:
Let me think... 北京的空气质量通常...（模型训练数据可能过时）
❌ 无法给出准确答案

# ReAct（成功）:
Thought: 我需要查询今天北京的空气质量数据
Action: Search["北京空气质量 2024-01-15"]
Observation: 北京今日 AQI 85，良，PM2.5 浓度 35μg/m³
Thought: 已经获取到准确数据
Action: Finish["今天北京空气质量良，AQI 85"]
✅ 成功获取实时信息
```

### 核心差异

| 问题             | 纯推理 (CoT)   | ReAct |
| -------------- | ----------- | ----- |
| "地球是平的吗？"      | 模型可能记错，无法验证 | 搜索验证  |
| "当前油价多少？"      | 模型数据可能过时    | 实时查询  |
| "2024年奥运会在哪开？" | 依赖训练截止日期    | 搜索确认  |

***

## 4. Prompt 模板（关键！）

### 完整的 ReAct Prompt 结构

```python
# Prompt 模板 = 4 个部分

REACT_PROMPT = """
# 1. 任务描述
你是一个智能助手，可以使用外部工具来回答问题。
请按照 Thought -> Action -> Observation 的格式来思考和行动。

# 2. 可用工具列表
可用工具:
- Search[query]: 搜索引擎，用于查询信息
- Calculator[expression]: 计算器，用于数学运算
- Weather[city]: 查询城市天气

工具使用格式:
Action: <工具名>[参数]

# 3. Few-shot 示例
示例 1:
Question: 北京明天的天气？
Thought 1: 我需要查询北京明天的天气
Action 1: Weather[北京]
Observation 1: 北京明天晴，气温 15-25°C
Thought 2: 已经获取到天气信息
Action 2: Finish[北京明天晴，气温 15-25°C]

示例 2:
Question: 123 * 456 等于多少？
Thought 1: 这是一个数学计算问题
Action 1: Calculator[123 * 456]
Observation 1: 56088
Thought 2: 计算完成
Action 2: Finish[56088]

# 4. 当前问题
Question: {user_question}
"""
```

### 格式细节

| 元素              | 格式                          | 说明           |
| --------------- | --------------------------- | ------------ |
| **Thought**     | `Thought N: ...`            | LLM 的思考过程    |
| **Action**      | `Action N: ToolName[param]` | 调用哪个工具，参数是什么 |
| **Observation** | `Observation N: ...`        | 工具返回的结果      |
| **Finish**      | `Action N: Finish[answer]`  | 终止并输出最终答案    |

### 解析器逻辑

```python
def parse_llm_output(output: str) -> dict:
    """解析 LLM 输出"""
    lines = output.strip().split('\n')

    for line in lines:
        if line.startswith('Thought'):
            return {'type': 'thought', 'content': line.split(':')[1].strip()}
        elif line.startswith('Action'):
            # 提取工具名和参数
            match = re.match(r'Action \d+: (\w+)\[(.+)\]', line)
            if match:
                return {
                    'type': 'action',
                    'tool': match.group(1),
                    'params': match.group(2)
                }
            # Finish 动作
            if 'Finish[' in line:
                answer = line.split('Finish[')[1].rstrip(']')
                return {'type': 'finish', 'answer': answer}

    return {'type': 'unknown'}
```

***

## 5. 和 Function Calling 的关系

### 发展历程

```
ReAct 论文 (2022.10)     Function Calling (2023.06)
    │                          │
    ▼                          ▼
┌─────────────┐          ┌─────────────┐
│ 纯文本解析   │          │ JSON 结构化  │
│ ToolName[x] │          │ {"name":    │
│             │          │  "ToolName",│
│ 脆弱:       │          │  "args": {} │
│ - 需要正则  │          │ }           │
│ - 容易出错  │          │             │
└─────────────┘          │ 可靠:       │
                         │ - 标准格式  │
                         │ - 易于解析  │
                         └─────────────┘
```

### 对比分析

| 维度           | ReAct (文本格式) | Function Calling (JSON) |
| ------------ | ------------ | ----------------------- |
| **输出格式**     | 自然语言 + 标记    | 结构化 JSON                |
| **解析难度**     | 高（需要正则）      | 低（标准 JSON 解析）           |
| **可靠性**      | 低（格式容易错）     | 高（模型训练过）                |
| **灵活性**      | 高（任何工具都能表示）  | 中（需要预定义 schema）         |
| **Token 效率** | 低（输出冗余）      | 高（紧凑格式）                 |

### 本质相同

```python
# ReAct 格式
output = """
Thought 1: 需要查询天气
Action 1: Search[北京天气]
"""

# Function Calling 格式
output = {
    "role": "assistant",
    "content": None,
    "tool_calls": [{
        "id": "call_123",
        "type": "function",
        "function": {
            "name": "Search",
            "arguments": '{"query": "北京天气"}'
        }
    }]
}

# 本质都是：LLM 输出"调用意图" → 代码执行 → 结果回传
```

### 实际应用

```python
# 简化的 Function Calling 实现
class ReActAgent:
    def __init__(self, tools):
        self.tools = tools

    def run(self, question: str):
        messages = [{"role": "user", "content": question}]

        for step in range(5):  # 最多 5 步
            response = llm.chat(
                messages=messages,
                tools=self.tools_schema  # 工具定义
            )

            # 如果有 tool_calls，执行
            if response.tool_calls:
                for call in response.tool_calls:
                    result = self.execute_tool(call)
                    messages.append({"role": "tool", "content": result})
            else:
                # 返回最终答案
                return response.content
```

***

## 6. 实验结果（论文摘要）

### 主要发现

| 任务类型      | CoT | Act-only | ReAct  | 提升      |
| --------- | --- | -------- | ------ | ------- |
| **知识密集型** | 好   | 中        | **最好** | +20-30% |
| **推理密集型** | 好   | 差        | **好**  | +5-10%  |
| **开放式问题** | 差   | 中        | **最好** | +40-50% |

### 关键洞察

1. **ReAct 对复杂问题效果最好**：多步骤、需要工具的任务
2. **简单任务 CoT 足够**：数学计算、简单逻辑
3. **主要瓶颈在工具质量**：工具好用，ReAct 效果好

***

## 7. 总结

### ReAct 核心要点

```
1. 循环: Thought → Action → Observation → ... → Finish
2. Prompt: 任务描述 + 工具列表 + Few-shot 示例 + 当前问题
3. 优势: 解决幻觉问题，支持外部工具，可解释性强
4. 进化: Function Calling 是 ReAct 的工业化版本
```

### 实战建议

| 建议               | 原因                |
| ---------------- | ----------------- |
| **限制最大步数**       | 防止无限循环，通常 5-10 步  |
| **提供良好工具**       | 工具质量决定 ReAct 上限   |
| **清晰的 Few-shot** | 格式示例比长篇说明更有效      |
| **错误处理**         | 工具调用失败时给 LLM 重试机会 |

***

## 8. 相关资源

- 📄 论文原文: <https://arxiv.org/abs/2210.03629>
- 🎥 论文解读: <https://www.youtube.com/watch?v=EigM5nLq0wA>
- 🐍 实现参考: <https://github.com/ysymyth/ReAct>
- 📚 扩展阅读: 《LLM Agent》by Lilian Weng
