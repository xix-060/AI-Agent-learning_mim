# 第 4-5 周 Agent 核心架构复盘

## 1. 最重要的 3 个收获

### 1.1 ReAct 循环（Agent 心脏）
ReAct（Reasoning + Acting）让 LLM 交替进行"思考 → 行动 → 观察"，直到得出答案。它的本质不是某个框架的 API，而是一种**让 LLM 学会"停下来调用工具、再继续推理"的范式**。在 knowledge_agent 里，Agent 遇到"现在几点"会主动调 `get_time`，遇到"计算 25*17+100"会调 `calculator`——这就是 ReAct 在起作用。理解了它，就理解了为什么 Agent = LLM + 工具 + 循环。

### 1.2 LangGraph 状态机（生产级 Agent）
LangGraph 把 Agent 抽象成一张状态图：节点是处理单元（agent / tools / human review），边是流转逻辑（条件分支、循环）。它的价值在于**把"自由对话的 LLM"约束成"可控、可中断、可恢复"的流程**。在 human_in_loop.py 里，用 `interrupt()` + `Command(resume)` 实现了人工审批暂停/恢复——这是普通 while 循环做不到的。`START`/`END` 常量、`ToolNode`、`add_conditional_edges` 这些 API 背后是"状态机思维"。

### 1.3 Multi-Agent 三种范式
多智能体不是"多个 Agent 凑一起"，而是有明确协作模式：
- **Sequential（顺序）**：A 的输出喂给 B，流水线式
- **Hierarchical（层级）**：Supervisor 调度子 Agent，分工明确
- **Collaborative（协作）**：Agent 之间平等对话、互启工具

CrewAI 的实践让我体会到：选错范式比不用框架更糟——顺序任务用层级调度会产生大量无效调用。

## 2. 项目 1 状态

- **成功率：100%（5/5）**
  - 评测集覆盖 PDF / TXT / Markdown 三种文档源
  - 5 个问题分别命中关键词：感知、推理、余弦、向量数据库、示例
  - 知识库 7 个分块，RAG 检索 + reranker 全链路打通

- **最难的部分：**
  1. **HITL 机制的理解**：最初用 `awaiting_human` 布尔标志实现暂停，根本行不通。理解 `interrupt()` 是"把状态序列化后挂起"、`Command(resume)` 是"注入人类输入后从断点继续"花了不少功夫——这是状态机思维和过程式思维的冲突。
  2. **环境一致性陷阱**：conda `ai-agent` 环境和系统 Python 反复打架，多次 `ModuleNotFoundError`。教训：跑之前先确认 `which python`，依赖装对环境。
  3. **Windows 编码与 emoji 冲突**：GBK 控制台遇到 emoji 直接 `UnicodeEncodeError`，排查过好几轮。最终统一用 UTF-8 + ASCII 替代 emoji。
  4. **CrewAI LLM 配置**：1.15+ 必须用 `crewai.LLM` 而非 `langchain_openai.ChatOpenAI`，否则 Pydantic 校验报错。框架版本变更带来的隐性 breaking change。

- **最有成就感的部分：**
  1. **评测 100% 通过**——不是"能跑"，而是"可验证地跑对"。这正是"评估比开发更重要"的闭环验证。
  2. **完整打通链路**：多源文档导入（PDF/TXT/MD/网页）→ 文本分割 → 向量检索 → reranker 重排序 → ReAct Agent → 工具调用 → 评测，每一环都亲手实现。
  3. **MCP Server 重构**：从 `mcp.server.Server` 迁移到 `FastMCP`，用 `@mcp.tool()` 装饰器 + 类型注解自动生成 schema，代码精简约 30%，还跑通了 Inspector 调试。
  4. **录制了 3-5 分钟演示视频**：项目结构 / 导入 / 对话 / 工具调用一镜到底，把抽象的 Agent 变成可展示的成果。

## 3. Agent 开发的关键认知

- **Agent = LLM + 工具 + 循环 + 记忆**：四要素缺一不可。没有工具，LLM 只会说话不会做事；没有循环，无法多步推理；没有记忆，每轮都是失忆的。
- **框架是工具，理解原理才是核心**：LangGraph / CrewAI / MCP 都会迭代升级（这周已经踩过 CrewAI 版本坑），但 ReAct、状态机、tool calling 协议这些原理不变。先懂原理，再学框架，框架变了能快速迁移。
- **评估比开发更重要（不评估不知道行不行）**：这次 5 题评测集 100% 通过，给了"真的做对了"的证据。没有 eval.py，"能跑"只是主观感觉。后续每个 Agent 都该先想"怎么评测"。

## 4. 下周（框架深入）想重点学什么

1. **LangGraph 进阶机制**：子图（subgraph）组合、动态路由、checkpoint 持久化、流式输出（streaming）。目标是能搭出"可中断、可回放、可观测"的复杂 Agent 工作流。
2. **Multi-Agent 协作模式深入**：Supervisor / Hierarchical / Swarm 三种模式的工程取舍，什么场景该用哪种，避免"为了多 Agent 而多 Agent"。
3. **Agent 可观测性与调试**：引入 LangSmith 或类似 trace 工具，看清每一步的 prompt、tool call、token 消耗——这是从"能跑"到"可优化"的关键。
4. **规划与反思能力**：学习 Plan-and-Execute、Self-Reflection / Reflexion，让 Agent 在复杂多步任务里先规划再执行、执行后自我修正，提升长程任务成功率。
5. **MCP 生态深入**：把 MCP Server 接入真实 Agent，体验"工具即服务"的标准化协作，对比手写 tool schema 的优劣。
