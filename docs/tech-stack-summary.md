# 技术栈总结

## 已掌握的工具栈

### LLM 调用

| 方式     | 工具                      | 场景    |
| :----- | :---------------------- | :---- |
| 云端 API | DeepSeek/Qwen DashScope | 生产    |
| 本地推理   | Ollama                  | 隐私/离线 |
| 框架封装   | LangChain ChatOpenAI    | 应用    |

### RAG

| 组件        | 选型                             |
| :-------- | :----------------------------- |
| Embedding | text-embedding-v3              |
| 向量库       | Chroma（开发）/ Milvus（生产）         |
| 切块        | RecursiveCharacterTextSplitter |
| 评估        | RAGAS / 手动评估                   |

### Agent

| 组件      | 选型             |
| :------ | :------------- |
| 编排      | LangGraph      |
| 框架      | LangChain LCEL |
| 多 Agent | CrewAI         |
| 协议      | MCP            |

### 模型

| 任务    | 工具                       |
| :---- | :----------------------- |
| 加载/推理 | HuggingFace Transformers |
| 微调    | PEFT + TRL（第 7 周）        |
| 训练    | PyTorch                  |

### 部署

| 组件   | 工具                |
| :--- | :---------------- |
| API  | FastAPI + uvicorn |
| 本地模型 | Ollama            |
| 监控   | LangSmith         |
