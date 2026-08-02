# 第 4 周复盘

## 1. 本周最重要的 3 个收获

### 1. ReAct 循环（Agent 的心脏）

- **核心公式**: Thought → Action → Observation → ... → Final Answer
- **实现关键**:
  - 解析 LLM 输出的结构化格式（用正则匹配 `Thought:`、`Action:`、`Action Input:`）
  - 执行工具后将结果作为 `Observation` 返回
  - 循环直到 LLM 输出 `Final Answer` 或达到最大步数
- **实际意义**: 这是让 Agent "会思考、会行动" 的基础框架

### 2. Memory 三层架构

- **短期记忆 (Short-term)**: 当前对话窗口，直接拼接在 Prompt 里
- **长期记忆 (Long-term)**: 存储用户信息、知识，用向量检索
- **情景记忆 (Episodic)**: 存储历史任务、成功经验，支持 "之前怎么做成的" 查询
- **关键设计**: 记忆检索要去重、限制数量、按相关性排序

### 3. 完整的 Agent 系统

- **组件集成**: LLM + Tools + Memory + Parser
- **评测闭环**: 设计 10 个测试任务 → 运行 Agent → 统计成功率
- **迭代优化**: 根据失败案例调整 Prompt 和工具描述

***

## 2. Agent 最难的部分是什么？

### 难度排序（个人感受）

| 难度    | 模块               | 原因                      |
| ----- | ---------------- | ----------------------- |
| ⭐⭐⭐⭐⭐ | **解析器 (Parser)** | LLM 输出格式不稳定，正则匹配脆弱      |
| ⭐⭐⭐⭐  | **Prompt 工程**    | 既要指令清晰，又不能冗余，还要给例子      |
| ⭐⭐⭐   | **工具描述**         | 描述不清楚导致 LLM 乱选工具或参数错    |
| ⭐⭐    | **记忆管理**         | 设计合理的检索和存储策略            |
| ⭐     | **循环控制**         | max\_steps 限制、检测循环、提前终止 |

### 最大痛点：解析器

```python
# 这个正则太脆弱了！
pattern = r'Thought: (.+?)\nAction: (\w+)\nAction Input: (.+?)\n'

# LLM 可能输出：
# - 空格/换行不一致
# - 用引号包裹
# - JSON 格式
# - 不按格式来...
```

**解决思路**:

1. 多写几个正则 fallback
2. 加格式校验和错误重试
3. 给 LLM 更明确的格式例子

***

## 3. 我的 Agent 评测成功率

### 评测结果（10 个任务）

| 分类     | 任务数    | 成功    | 成功率     |
| ------ | ------ | ----- | ------- |
| 时间     | 1      | 1     | 100%    |
| 计算     | 2      | 2     | 100%    |
| 文件     | 3      | 3     | 100%    |
| 多步     | 2      | 1     | 50%     |
| 推理     | 2      | 2     | 100%    |
| **总计** | **10** | **9** | **90%** |

### 问题分析

- ❌ 失败任务: `计算 2^10 然后把结果写入文件 power_result.txt`
- **原因**: 关键词 "已写入" 在答案里，但匹配逻辑没找到
- **本质**: 评测脚本的判断条件太严格（只在开头找，没全文搜索）

### 改进后的成功率

修正评测脚本后: **70%**（真实反映 Agent 能力）

### 目标达成

✅ **70% 目标已达成**

***

## 4. 5 种失败模式我遇到了几种？

### 失败模式清单

| # | 模式        | 描述                                   | 遇到了？ |
| - | --------- | ------------------------------------ | ---- |
| 1 | **工具选错**  | 需要计算却调用 search                       | ✅ 是  |
| 2 | **参数格式错** | JSON 语法错、引号嵌套问题                      | ✅ 是  |
| 3 | **无限循环**  | Action → Observation → 同样的 Action... | ✅ 是  |
| 4 | **提前结束**  | 还没完成就输出 Final Answer                 | ✅ 是  |
| 5 | **幻觉编造**  | 编造工具返回结果                             | ❌ 没有 |

### 案例分析

#### 模式 1: 工具选错

```
Q: 15 * 23 = ?
A: [Search("乘法口诀")]  # 应该用 calculator
```

**改进**: 在 system prompt 强调 "计算任务必须用 calculator"

#### 模式 2: 参数格式错

```json
// 错误
{"expression": 15 * 23}  // 少引号

// 正确
{"expression": "15 * 23"}
```

**改进**: 给 few-shot 例子时强调格式

#### 模式 3: 无限循环

```
Action: list_directory → Observation: [文件列表]
Action: list_directory → Observation: [文件列表]
Action: list_directory → ...  # 重复 8 次后终止
```

**改进**: 检测连续相同 Action，触发反思

#### 模式 4: 提前结束

```
Q: 先算 100/4，再告诉我时间
Action: calculator(100/4) → Observation: 25
Final Answer: 结果是 25  # 忘了问时间
```

**改进**: 在 prompt 里强调 "完成所有子任务才能 Final Answer"

***

## 5. 下周（Multi-Agent + LangGraph）想重点学什么？

### 5.1 Multi-Agent 核心概念

```python
# 单 Agent: 一个人干所有事
agent = SimpleAgent()
agent.run("写代码 + 测试 + 文档")

# Multi-Agent: 分工协作
planner = PlannerAgent()      # 规划任务
coder = CoderAgent()          # 写代码
tester = TesterAgent()        # 测试
reviewer = ReviewerAgent()    # 审核

# 协作模式
tasks = planner.plan("实现用户登录")
for task in tasks:
    code = coder.run(task)
    test_result = tester.test(code)
    reviewer.review(code, test_result)
```

### 5.2 LangGraph 学习目标

| 目标       | 说明                       |
| -------- | ------------------------ |
| **图结构**  | 把 Agent 流程建模为状态图（节点 + 边） |
| **状态管理** | 每个节点读写共享状态               |
| **条件分支** | if-else 路由（成功走 A，失败走 B）  |
| **循环**   | 显式定义循环和终止条件              |

### 5.3 周计划

1. **Day 1-2**: 学习 LangGraph 基础（StateGraph、节点、边）
2. **Day 3-4**: 实现一个简单的 Multi-Agent 系统（Planner + Executor）
3. **Day 5**: 对比 LangGraph vs 手写 ReAct 的差异
4. **Day 6-7**: 尝试在现有 Agent 里加入 Multi-Agent 协作

### 5.4 期望改进

- [ ] 解决单 Agent 的 "工具选错" 问题（让专家 Agent 决策）
- [ ] 用 LangGraph 实现更清晰的控制流
- [ ] 实现 Planner-Executor 模式

***

## 附：本周代码统计

| 指标   | 数值                                                                    |
| ---- | --------------------------------------------------------------------- |
| 新增文件 | \~10 个 (agent\_v1.py, agent\_eval.py, memory.py, tools/builtin.py...) |
| 修改文件 | \~15 个                                                                |
| 代码行数 | \~1000+ 行                                                             |
| 写文档  | 4 个 (Reflexion, Toolformer, week4-review\...)                         |
| 调试次数 | 30+ 次                                                                 |

***

## 总结

这周完成了从 0 到 1 的 Agent 系统搭建，理解了 ReAct 循环、Memory 架构、Tools 集成的完整链路。最有成就感的是跑通了 10 个评测任务（90% 原始成功率），虽然修正后是 70%，但确实达到了目标。

下周要进入 Multi-Agent 和 LangGraph，期待能解决单 Agent 的一些痛点！
