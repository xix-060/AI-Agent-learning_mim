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
