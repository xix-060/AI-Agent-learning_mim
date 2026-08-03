# Multi-Agent 三种协作范式

## 1. Supervisor（监督者模式）

```
架构：一个 Supervisor Agent 调度多个 Worker Agent
         ┌── Worker A（搜索）
Supervisor ──┼── Worker B（计算）
         └── Worker C（写作）

特点：中心化控制，Supervisor 决策"谁来做什么"
适用：任务可明确分工的场景
优点：流程清晰，可控
缺点：Supervisor 是瓶颈
```

## 2. Group Chat（群聊模式）

```
架构：多个 Agent 在一个"群聊"里，轮流发言
Agent A ←→ Agent B
   ↕           ↕
Agent C ←→ Agent D

特点：去中心化，Agent 之间直接交流
适用：头脑风暴、讨论、辩论
优点：灵活，涌现性强
缺点：容易跑偏，难控制
```

## 3. Workflow（工作流模式）

```
架构：固定流程，Agent 按顺序处理
Agent A → Agent B → Agent C → 输出
（每个 Agent 处理上一步的结果）

特点：流水线，每个 Agent 专注一个环节
适用：内容生产、数据处理流水线
优点：可预测，易调试
缺点：不灵活，不能跳步
```

## 范式选择指南

| 场景       | 推荐范式                  |
| :------- | :-------------------- |
| 任务可拆解分工  | Supervisor            |
| 需要讨论/创意  | Group Chat            |
| 固定流程的生产  | Workflow              |
| 复杂项目（混合） | Supervisor + Workflow |

## 对比表

| 范式         | 控制方式 | 灵活性 | 可控性 | 成本 |
| :--------- | :--- | :-- | :-- | :- |
| Supervisor | 中心化  | 中   | 高   | 中  |
| Group Chat | 去中心化 | 高   | 低   | 高  |
| Workflow   | 固定   | 低   | 高   | 低  |
