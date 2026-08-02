# Agent 全景图

## Agent = LLM + 4 大组件

1. 🧠 规划（Planning）
   - CoT / ToT / Plan-and-Execute / Reflexion
   - 决定"怎么做"
2. 💾 记忆（Memory）
   - 短期（对话历史）/ 长期（向量库）/ 情景（过往经历）
   - 决定"记得什么"
3. 🛠 工具使用（Tool Use）
   - Function Calling / MCP / 自定义工具
   - 决定"能做什么"
4. 🔄 行动循环（Action Loop）
   - ReAct: Thought → Action → Observation
   - 决定"怎么执行"

## Agent 分类

- 单 Agent：一个 LLM + 工具循环（本周学的）
- 多 Agent：多个 Agent 协作（下周学）
  - Supervisor 模式
  - Group Chat 模式
  - Workflow 模式

## Agent 框架对比

| 框架              | 特点         | 适用   |
| :-------------- | :--------- | :--- |
| 原生（本周）          | 手写循环，理解原理  | 学习   |
| LangChain Agent | 封装好，快速用    | 原型   |
| LangGraph       | 状态机，可控制流   | 生产   |
| AutoGen         | 多 Agent 对话 | 协作   |
| CrewAI          | 角色扮演       | 团队模拟 |
