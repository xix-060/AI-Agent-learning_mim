# 第 6 周复盘

📅 **周期**: LCEL 管道 → LangGraph 进阶 + LangSmith → 本地部署/Ollama → PyTorch 训练 → FastAPI 服务化
📊 **代码统计**: 8 个 Commit（08-10 \~ 08-14）+ 任务 7.1 | 约 1,760 行代码 | 9 个新源文件

***

## 1. 本周学到的最重要的 3 个概念

### 1️⃣ LCEL 管道编排（LangChain Expression Language）

**核心思想**: 用 `|` 管道符把组件串成链，类比 Unix 管道 `cat file | grep "error" | wc -l`。所有组件都实现 `Runnable` 接口（invoke / batch / stream / ainvoke），所以任意组件可拼装。

**四大原语**:

| 原语                    | 作用         | 示例                                                          |
| --------------------- | ---------- | ----------------------------------------------------------- |
| `\|`                  | 管道，前输出→后输入 | `prompt \| llm \| parser`                                   |
| `RunnablePassthrough` | 透传输入       | `{"context": retriever, "question": RunnablePassthrough()}` |
| `RunnableParallel`    | 并行执行       | `RunnableParallel(a=..., b=...)`                            |
| `RunnableLambda`      | 包装普通函数     | `RunnableLambda(lambda x: x.upper())`                       |

**我的实现**（[lc\_pipeline.py](file:///e:/git/AI-Agent-learning_mim/src/lc_pipeline.py)，6 种链）:

```python
# LCEL 推荐写法
chain = prompt | llm | parser

# 旧版（已废弃）
# from langchain.chains import LLMChain
# chain = LLMChain(llm=llm, prompt=prompt)
```

**关键洞察**: LCEL 的本质是 **统一接口 + 声明式组合**。旧版 `LLMChain` 是"固定模板"，LCEL 是"乐高积木"——`RunnablePassthrough` 让 RAG 的"检索 context + 透传 question"一行搞定，这是旧 Chain 写起来很别扭的。

***

### 2️⃣ LangGraph 进阶 + LangSmith 可观测性

**LangGraph 进阶**（[advanced\_features.py](file:///e:/git/AI-Agent-learning_mim/src/langgraph_advanced/advanced_features.py)，302 行）:

- **Checkpointer 持久化**: `MemorySaver`（内存，重启丢）/ `SqliteSaver`（本地持久）/ `PostgresSaver`（生产级）。这是"可恢复"的物理基础。
- **可暂停/可恢复**: `interrupt()` 主动暂停 → 把状态序列化挂起 → `invoke(None, config)` 从断点继续。
- **Human-in-the-Loop 三种模式**: 审批（危险操作前问人）/ 编辑（人改 Agent 输出）/ 引导（人补信息）。

**LangSmith 追踪**（踩了不少坑）:

- 必须在 `.env` 配 `LANGCHAIN_ENDPOINT=https://apac.api.smith.langchain.com`（APAC 区域），否则 trace 路由错 → 403。
- 配好后追踪**自动启用**（`tracing_enabled` 已废弃，别再手动 import）。
- 在 [apac.smith.langchain.com](https://apac.smith.langchain.com) 的 `ai-agent-learning` 项目看 trace：执行树展开有节点（classifier/chat/code/math）、输入输出、循环图的多轮迭代。

**关键洞察**: LangGraph = **把自由对话的 LLM 约束成可控、可中断、可恢复的流程**；LangSmith = **从"能跑"到"可观测、可优化"**。没 trace 之前调 Agent 全靠猜，有了 trace 每一步的 prompt/tool call/token 都看得见。

***

### 3️⃣ PyTorch 训练循环（MLP 跑通 MNIST）

**第一次完整跑通训练循环**（[mnist\_mlp.py](file:///e:/git/AI-Agent-learning_mim/src/mnist_mlp.py)，249 行）:

```python
# 经典 5 步训练循环
for epoch in range(5):
    for x, y in train_loader:
        # 1. 前向
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        # 2. 反向
        optimizer.zero_grad()
        loss.backward()
        # 3. 更新
        optimizer.step()
```

**模型结构**: `784 → 256 → 10`（输入 28×28 展平，隐藏 256，输出 10 类）
**结果**: 5 个 epoch 后测试准确率 **>97%**，训练曲线见 [mnist-training-curve.png](file:///e:/git/AI-Agent-learning_mim/docs/mnist-training-curve.png)。

**关键洞察**: 训练循环的本质 = **前向算损失 → 反向算梯度 → 优化器更新参数**，循环往复。搞懂这个循环，再看 Transformer 训练、微调（下周）都是同一套骨架，只是模型和损失函数变了。

***

## 2. 最难的部分

### 数据加载连环坑（PyTorch/HF 那天最折磨）

本以为跑个 MNIST 很简单，结果栽在数据加载上一整天：

1. **HuggingFace 连不上**: `datasets.load_dataset("ylecun/mnist")` → `ConnectTimeout (WinError 10060)`。根因：直连 huggingface.co 被墙。
   → 解：设 `HF_ENDPOINT=https://hf-mirror.com`，且必须 **import 前设**。
2. **datasets 不吃环境变量**: 设了 `HF_ENDPOINT` 还是 `LocalEntryNotFoundError`——datasets 库在 import 时就读了 endpoint，设晚了无效。
   → 解：干脆绕过，用 `urllib.request` 直接从 Google Cloud Storage 下 MNIST 原始 IDX 文件，不依赖 torchvision/datasets。
3. **EOFError 文件不完整**: 下到一半的文件残留 → 反序列化失败。
   → 解：加文件大小校验，不完整就重下。
4. **中文字体缺失**: matplotlib 画训练曲线，中文标签全是方框（`missing glyph in DejaVu Sans`）。
   → 解：加 SimHei / Microsoft YaHei 字体。

**教训**: **数据加载是 ML 工程里最被低估的时间黑洞**。模型代码 30 行，数据管道折腾一天。以后新数据源先验证"能下下来、能读出来"再写模型。

### LlamaIndex + DashScope 集成

[llamaindex\_demo.py](file:///e:/git/AI-Agent-learning_mim/src/llamaindex_demo.py) 跑 DashScope 时：

- `text-embedding-v3` 不在 LlamaIndex 的 `OpenAIEmbeddingModelType` 枚举里 → 报错
- `qwen-turbo` 不被识别为 OpenAI 模型 → 报错

→ 解：LLM 用 `OpenAILike` 类、Embedding 用 `LangchainEmbedding` 包 `OpenAIEmbeddings`，绕开 LlamaIndex 的模型名枚举检查。**不同框架对"OpenAI 兼容"的兼容程度不一样**，LlamaIndex 比 LangChain 挑剔。

### 本周尾的网络/LLM 异常（任务 7.1）

给项目 1 加 API 时，`/chat` 端到端死活跑不通：

- DashScope 的 qwen-turbo 一句 "hi" 要 **52.5s**（正常 1-3s）
- LangSmith 连不上 APAC 端点，追踪器把 LLM 调用卡死（>5min 无输出）
- 诊断脚本证实：Embedding 0.5s 正常，LLM 能通但极慢，带工具的 Agent 调用直接挂起

→ 结论：**不是 API 代码的锅**（HTTP 层全正常），是环境网络 + qwen-turbo 即将下线导致的性能异常。模型更换推迟到十月。

***

## 3. 项目 1 现状：有 API 了，部分部署

### ✅ API 服务（任务 7.1，本周收尾）

把 Agent 包成 REST API（[knowledge\_agent/api.py](file:///e:/git/AI-Agent-learning_mim/knowledge_agent/api.py)）:

| 端点                         | 功能       | 状态                       |
| -------------------------- | -------- | ------------------------ |
| `GET /health`              | 健康检查     | ✅ 正常                     |
| `GET /`                    | 服务信息     | ✅ 正常                     |
| `POST /chat`               | 对话       | ⚠️ HTTP 层 OK，端到端待 LLM 恢复 |
| `POST /chat/stream`        | SSE 流式对话 | ✅ 路由 OK                  |
| `POST /import` / `/upload` | 导入/上传文档  | ✅ 路由 OK                  |

关键改动：从 `src/agent_api/main.py` 复制到 `knowledge_agent/api.py`，**只改了** **`PROJECT_ROOT`** **路径层级**（`parents[2]` → `parents[1]`），导入路径 `from knowledge_agent.src.agent import KnowledgeAgent` 不变。交互文档 `/docs` 可浏览器访问。

### ✅ 部署形态

- **Streamlit demo**（[hf\_space/app.py](file:///e:/git/AI-Agent-learning_mim/hf_space/app.py)）: 准备 HF Space 部署
- **FastAPI REST API**: `uvicorn api:app --port 8000`，前后端分离可用
- **Ollama 本地部署**（[ollama\_demo.py](file:///e:/git/AI-Agent-learning_mim/src/ollama_demo.py) + [ollama\_deploy.sh](file:///e:/git/AI-Agent-learning_mim/scripts/ollama_deploy.sh)）: 拉了 `qwen2.5:0.5b`，本地免费推理可用，但 0.5B 幻觉严重（把 RAG 解释成 "Retained Agreement"），只适合 hello world/闲聊，复杂任务要云端大模型或更大本地模型。

***

## 4. 本周代码统计

### 📊 提交记录（8 个 Commit）

| Date  | Commit  | 核心产出                                          |
| ----- | ------- | --------------------------------------------- |
| 08-10 | 7bdf1af | 接入长期记忆，跨会话记住用户信息                              |
| 08-10 | 745d635 | LangChain LCEL 6 种链实战                         |
| 08-10 | bae0239 | LangGraph 高级特性 + LangSmith 追踪                 |
| 08-13 | 6602c08 | Streamlit demo + LlamaIndex 兼容                |
| 08-13 | eed3df7 | hf\_demo.py HF 镜像 + datasets namespace + 编码修复 |
| 08-13 | 652f35c | Ollama 本地部署 Qwen2.5 + API 调用                  |
| 08-13 | c1a666f | PyTorch MLP 训练 MNIST（准确率 >97%）                |
| 08-14 | 032d3e6 | FastAPI 把 Agent 包成 REST API                   |

> 另：08-26 任务 7.1 收尾（knowledge\_agent/api.py + README + requirements，未提交）

### 📁 新增文件

```
src/
├── lc_pipeline.py                  # LCEL 6 种链
├── langgraph_advanced/
│   └── advanced_features.py        # LangGraph 进阶 + LangSmith
├── llamaindex_demo.py              # LlamaIndex + DashScope
├── hf_demo.py                      # HuggingFace 镜像版
├── ollama_demo.py                  # Ollama 本地推理
├── mnist_mlp.py                    # PyTorch MLP MNIST
└── agent_api/
    ├── basic.py                    # FastAPI 基础骨架
    └── main.py                     # Agent REST API
knowledge_agent/
└── api.py                          # 项目 1 API 服务（任务 7.1）
hf_space/
└── app.py                          # Streamlit demo
docs/
├── lcel-notes.md
├── langgraph-advanced.md
├── llamaindex-vs-langchain.md
├── pytorch-basics.md
└── mnist-training-curve.png
```

### 📈 代码量

```
本周新增约 1,760 行（8 个 commit ~1,560 + 任务 7.1 ~200）
```

***

## 5. 本周踩坑沉淀（已进项目记忆）

| 坑                       | 教训                                                           |
| ----------------------- | ------------------------------------------------------------ |
| HF 连不上                  | `HF_ENDPOINT=https://hf-mirror.com`，**import 前设**            |
| datasets 报错             | 新版强制 `namespace/name`（如 `stanfordnlp/imdb`）                  |
| LlamaIndex+DashScope    | LLM 用 `OpenAILike`，Embedding 用 `LangchainEmbedding`          |
| LangSmith 403           | APAC 区域必须设 `LANGCHAIN_ENDPOINT=apac.api.smith.langchain.com` |
| DashScope Embedding 400 | `OpenAIEmbeddings(check_embedding_ctx_length=False)`         |
| LangSmith 卡死 LLM        | 追踪连不上时会拖慢/阻塞 LLM 调用，必要时临时关 `LANGCHAIN_TRACING_V2`            |
| Trae 提交静默失败             | 用 `.git/logs/HEAD` 验证提交状态                                    |

***

## 6. 下周（微调）想重点学什么

### 🎯 核心目标

| 优先级 | 任务               | 预期产出                              |
| --- | ---------------- | --------------------------------- |
| ⭐⭐⭐ | **LoRA/PEFT 微调** | 在 Qwen2.5-0.5B 上微调一个垂直任务，跑通训练     |
| ⭐⭐⭐ | **训练循环进阶**       | 把本周 MLP 的 5 步循环迁移到 Transformer 微调 |
| ⭐⭐  | **数据集构建**        | 自建一个小规模指令微调数据集                    |
| ⭐⭐  | **微调 vs RAG 取舍** | 什么场景该微调，什么场景该 RAG                 |
| ⭐   | **量化部署**         | AWQ/GPTQ 量化 + 本地推理                |

### 💡 想搞清楚的问题

1. **微调到底改了什么？** 只改了一小部分参数（LoRA 的低秩矩阵），为什么就能改变行为？
2. **微调和 RAG 怎么选？** RAG 加新知识、微调改风格/格式——但边界在哪？
3. **0.5B 微调后能变强吗？** 本周发现 0.5B 幻觉严重，微调能救多少？
4. **微调数据要多干净？** 数据质量对效果的影响有多大？

### 🔗 前置知识（本周已掌握）

| 下周需要           | 本周学的                 | 关联            |
| -------------- | -------------------- | ------------- |
| Transformer 训练 | PyTorch 训练循环（MLP）    | 同一套前向/反向/优化骨架 |
| LoRA 低秩        | Attention 的 Q/K/V 投影 | LoRA 就插在投影矩阵上 |
| 数据加载           | MNIST 数据管道踩坑         | 知道数据加载是黑洞     |
| 模型评估           | Agent 评测思维           | "能验证地跑对"      |

***

## 📝 本周反思

### ✅ 做得好的地方

1. **框架横向对比**: LCEL vs 旧 Chain、LlamaIndex vs LangChain、本地 vs 云端，做了实打实的对比，不是只学一家。
2. **可观测性意识**: 主动接入 LangSmith trace，从"黑盒调 Agent"转向"看着 trace 调"。
3. **工程闭环**: 项目 1 从"能用"→"有 API"→"可部署（Streamlit/FastAPI）"，完整度上了一个台阶。
4. **踩坑即沉淀**: 每个坑都进了项目记忆，下次不重蹈。

### ⚠️ 需要改进

1. **网络环境太脆**: LangSmith/DashScope 连不上/慢的问题反复出现，影响验证效率。十月换模型时要一并解决网络/代理。
2. **本地模型太小**: 0.5B 幻觉严重，很多 demo 用不了。下周微调 0.5B 顺带验证"微调能救多少"。

### 🎯 下周期待

> 本周把 Agent 从"能对话"推进到"有 API、可部署、可观测"。
>
> 下周进入微调——终于要自己训模型了。从"用别人的模型"到"改别人的模型"，理解会再深一层。

***

**文档生成**: 2026-08-26
**下周目标**: LoRA 微调 + 自建微调数据集
