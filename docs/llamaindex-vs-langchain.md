# LlamaIndex vs LangChain

## 定位差异

| 维度   | LangChain     | LlamaIndex           |
| :--- | :------------ | :------------------- |
| 核心定位 | LLM 应用框架      | 数据连接框架               |
| 擅长   | Agent / 链式编排  | RAG / 索引             |
| 抽象   | Chain / Agent | Index / Query Engine |
| 适用   | 复杂 Agent 流程   | 文档密集型 RAG            |

## 核心概念

- Index：把数据组织成可检索的结构
- Query Engine：基于 Index 回答查询
- Node：最小数据单元（一个文本块）
- Document：原始文档（PDF/网页等）

## 何时选 LlamaIndex？

- 主要是 RAG 场景
- 文档来源多样（Notion/Slack/数据库）
- 需要高级索引（知识图谱/树状索引）
