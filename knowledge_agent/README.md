# 🧠 个人知识库 Agent

> 基于 LangGraph + RAG 的智能知识库助手，支持多源文档导入、跨源检索、工具调用。

## ✨ 功能

- 📄 多源文档导入：PDF / TXT / Markdown / 网页
- 🔍 跨源 RAG 检索：向量检索 + Reranker
- 🛠 工具调用：计算器 / 时间 / 文件操作
- 💬 多轮对话：短期记忆 + 长期记忆
- 🔄 LangGraph 编排：可暂停 / 可恢复 / 人机协同

## 🏗 架构

```
用户输入
    ↓
[LangGraph Agent]
    ↓
┌─────────────────┐
│   Agent Node    │  ← LLM 决策
└────────┬────────┘
         ↓
  [should_continue?]
     ↓        ↓
  [tools]    [END]
     ↓
[RAG Retrieve]  ← Chroma 向量库
     ↓
[Agent Node]    ← 工具结果
```

## 🚀 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 API Key
cp .env.example .env
# 编辑 .env

# 3. 导入文档
python main.py import --path ./data/sample.pdf

# 4. 对话
python main.py chat
```

## 📖 使用说明

### 导入文档

支持多种来源：

```bash
# PDF 文件
python main.py import --path ./data/uploads/report.pdf

# TXT / Markdown
python main.py import --path ./data/uploads/notes.md

# 网页 URL
python main.py import --path https://example.com/article
```

### 交互对话

```bash
python main.py chat
```

对话中可用命令：

| 命令      | 说明      |
| ------- | ------- |
| `help`  | 显示帮助    |
| `stats` | 查看知识库状态 |
| `clear` | 清空对话记忆  |
| `quit`  | 退出      |

### 查看知识库统计

```bash
python main.py stats
```

## 🌐 API 服务

通过 FastAPI 暴露 HTTP 接口，方便前端或其它客户端调用。

### 启动服务

```bash
# 在 knowledge_agent/ 目录下执行
python api.py
# 或指定 host/port
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

启动后访问：
- 根接口：<http://localhost:8000/>
- 交互式文档：<http://localhost:8000/docs>
- 健康检查：<http://localhost:8000/health>

### 接口列表

| 方法     | 路径             | 说明              |
| ------ | -------------- | --------------- |
| GET    | `/`            | 服务信息            |
| GET    | `/health`      | 健康检查            |
| POST   | `/chat`        | 对话（一次性返回）       |
| POST   | `/chat/stream` | 对话（SSE 流式返回）    |
| POST   | `/import`      | 导入文档（传文件路径或 URL） |
| POST   | `/upload`      | 上传文档并导入         |

### 调用示例

**对话**

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "你好", "session_id": "user1"}'
```

**前端 fetch 调用**

```javascript
const res = await fetch("http://localhost:8000/chat", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ message: "介绍一下知识库", session_id: "browser" }),
});
const data = await res.json();
console.log(data.response, data.elapsed_seconds);
```

**流式对话**（Server-Sent Events）

```javascript
const evt = new EventSource("http://localhost:8000/chat/stream");
// 注意：EventSource 默认只支持 GET，POST 流式请用 fetch + ReadableStream
```

**上传文档**

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@./data/uploads/notes.md"
```

> 响应字段：`response`（回答）、`session_id`、`elapsed_seconds`（耗时）、`tools_used`（使用的工具）。

## 📁 项目结构

```
knowledge_agent/
├── src/
│   ├── __init__.py
│   ├── agent.py          # LangGraph Agent 主逻辑
│   ├── rag.py            # RAG 检索模块（多源加载 + Reranker）
│   ├── tools.py          # 工具集（计算器/时间/文件操作）
│   ├── memory.py         # 记忆管理（短期 + 长期）
│   └── config.py         # 配置管理
├── tests/
├── docs/
│   └── architecture.md   # 架构文档
├── data/
│   ├── uploads/          # 上传的文档
│   └── chroma_db/        # 向量库
├── .env.example          # 环境变量模板
├── requirements.txt
├── api.py                # API 服务入口（FastAPI）
└── main.py               # 入口
```

## ⚙️ 配置说明

复制 `.env.example` 为 `.env` 并填写：

```bash
# LLM
LLM_API_KEY=your-api-key
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL=qwen-turbo

# Embedding（与 LLM 同服务商可省略）
EMBEDDING_MODEL=text-embedding-v3

# Reranker（与 LLM 同服务商可省略）
RERANKER_MODEL=gte-rerank
```

## 🛠 技术栈

| 组件     | 技术                       |
| ------ | ------------------------ |
| LLM 框架 | LangChain + LangGraph    |
| LLM 模型 | Qwen-Turbo               |
| 向量数据库  | ChromaDB                 |
| 嵌入模型   | text-embedding-v3        |
| 重排序模型  | gte-rerank               |
| PDF 解析 | pypdf                    |
| 网页抓取   | requests + BeautifulSoup |

## 📝 许可证

MIT License

## 📝 B站演示链接

https\://www\.bilibili.com/video/BV1Y2um6UEB8/?vd\_source=2b0d152e167a850719670c04905ef01e
