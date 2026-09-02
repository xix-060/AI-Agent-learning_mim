# 个人知识库 Agent 评测报告

> 项目 1（knowledge_agent）的量化评测汇总。覆盖三类评测：RAG 检索质量、Agent 任务成功率、部署性能基线。
> 所有数据均来自实测，未跑过的指标明确标注「待回填」，不编造。

---

## 1. 评测总览

| 维度 | 指标 | 值 | 数据来源 |
|---|---|---|---|
| RAG 检索质量 | context_precision | 1.00（5 题） | [rag-eval-manual.json](file:///e:/git/AI-Agent-learning_mim/docs/rag-eval-manual.json) |
| RAG 检索质量 | answer_relevancy | 1.00（5 题） | 同上 |
| RAG 生成质量 | faithfulness | 1.00（5 题手动） | 同上 |
| Agent 任务 | 任务成功率 | **80%（8/10）** | [agent-eval-report.json](file:///e:/git/AI-Agent-learning_mim/docs/agent-eval-report.json) |
| 部署性能 | /chat 平均延迟 | 56.7s | [DEPLOY.md](file:///e:/git/AI-Agent-learning_mim/knowledge_agent/DEPLOY.md) 性能基线 |
| 部署性能 | Token 吞吐 | ≈2.0 tokens/s | 同上 |
| 部署性能 | QPS @ P99<1s | 0（CPU 后端） | 同上；GPU+vLLM 待回填 |
| 部署性能 | /upload 向量化延迟 | 13.2s（1KB） | 同上 |

---

## 2. RAG 检索与生成质量（早期 RAG 模块，5 题手动评测）

口径：在通用 RAG 模块上对 5 道知识题做人工评测，逐题判定 `context_hit`（Top-K 是否命中相关片段）、`answer_relevant`（回答是否切题）、`faithfulness`（是否忠于检索上下文、无幻觉）。

| 问题 | context_hit | answer_relevant | faithfulness |
|---|---|---|---|
| AI 经历了几次浪潮？ | ✓ | ✓ | 1.0 |
| 谁提出了图灵测试？ | ✓ | ✓ | 1.0 |
| Transformer 是哪一年提出的？ | ✓ | ✓ | 1.0 |
| Agent 的四个核心组件是什么？ | ✓ | ✓ | 1.0 |
| MCP 是哪个公司提出的？ | ✓ | ✓ | 1.0 |

**结论**：小样本上检索命中与生成忠实度均满分。局限是题量小（5 题）、人工判定，不足以覆盖失败模式。后续应接入 RAGAS 自动化评测扩到 50+ 题。

---

## 3. Agent 任务成功率（10 题，关键词命中口径）

口径：见 [eval.py](file:///e:/git/AI-Agent-learning_mim/knowledge_agent/eval.py) 与 [src/agent_eval.py](file:///e:/git/AI-Agent-learning_mim/src/agent_eval.py)。覆盖时间/计算/文件/多步/推理五类，用「期望关键词是否出现在回答中」判定成功。

**总成功率：80%（8/10）**

| 类别 | 通过 / 总数 | 通过率 |
|---|---|---|
| 时间 | 1 / 1 | 100% |
| 计算 | 2 / 2 | 100% |
| 文件 | 2 / 3 | 66.7% |
| 多步 | 1 / 2 | 50% |
| 推理 | 2 / 2 | 100% |

### 3.1 失分分析（2 题失败，均属「文件写入」类）

| 问题 | 期望关键词 | 失败原因 |
|---|---|---|
| 创建文件 eval_test.txt，内容为'评估测试' | `已写入` | Agent 回答了结果但**未真正调用写文件工具**（steps=0），靠 LLM「口头声称」完成 |
| 计算 2^10 然后把结果写入文件 power_result.txt | `已写入` | 同上：算出了 1024，但**未实际写文件**就声称已写入（steps=0） |

### 3.2 根因与改进方向

- **根因**：LLM 在工具调用决策时，对「写文件」这类有副作用操作倾向于直接生成自然语言结论而非触发 `ToolNode`。这与项目记忆中「LLM 可能跳过必要工具调用」一致。
- **改进**：
  1. 对「写入/创建」类动词在 Router 强制走工具分支（硬规则兜底）；
  2. 工具执行结果回写后二次校验（如 `ls` 验证文件存在）；
  3. 评测脚本增加副作用断言（检查文件是否真的生成），而非仅判关键词。

### 3.3 复现命令

```bash
# 10 题任务评测
conda run -n ai-agent python src/agent_eval.py
# 报告输出：docs/agent-eval-report.json
```

---

## 4. 部署性能基线（CPU 后端，链路验证口径）

测试环境：本机 CPU + Ollama qwen2.5:0.5b（无 GPU），数据来自 Prometheus 实测（3 次 /chat + 1 次 /upload）。

| 指标 | 实测值 | 说明 |
|---|---|---|
| /chat 平均延迟 | 56.7s（34.7s / ~60s / ~75s） | 单次均在 120s 内 |
| Token 吞吐 | ≈2.0 tokens/s | 337 tokens / 170s 总时长 |
| /upload 向量化延迟 | 13.2s | 1KB 文档，DashScope embedding 网络往返为主 |
| QPS @ P99<1s | 0 | P99 落在 60~120s 档，远超 1s |

**定位说明**：CPU 0.5B 后端的目的不是达标性能，而是**验证全链路正确性**（导入 → 向量化 → 检索 → 生成 → 监控埋点）。P99<1s 的容量目标需 GPU + vLLM 7B 环境，上线前用同款 Locust 脚本（[src/loadtest/locustfile.py](file:///e:/git/AI-Agent-learning_mim/src/loadtest/locustfile.py)，改 host 与 QPS 目标）回填生产基线。

### 4.1 监控指标

| 指标名 | 含义 |
|---|---|
| `ka_requests_total` | HTTP 请求总数（method/path/status 标签） |
| `ka_request_latency_seconds` | 请求延迟直方图（bucket 到 300s） |
| `ka_llm_tokens_total` | LLM 输出 token 估算值（中文 2 字符/token） |

Grafana 面板截图见 [docs/img/grafana-dashboard.png](file:///e:/git/AI-Agent-learning_mim/docs/img/grafana-dashboard.png)。

### 4.2 复现命令

```bash
# 1. 启动容器栈
cd knowledge_agent/deploy && docker compose up -d --build

# 2. 冒烟测试
curl -X POST http://localhost:8081/upload -F "file=@data/sample.txt"
curl -X POST http://localhost:8081/chat -H "Content-Type: application/json" \
  -d '{"message":"文档的主要内容是什么?","session_id":"smoke"}'

# 3. 压测（Locust）
conda run -n ai-agent python -m locust -f src/loadtest/locustfile.py
```

---

## 5. 待办（评测维度）

- [ ] RAGAS 自动化评测接入，扩到 50+ 题，覆盖失败模式
- [ ] GPU + vLLM 回填 Locust 阶梯压测（并发 5/10/20/40），定位容量拐点
- [ ] Agent 评测增加副作用断言（文件实际生成校验），修复 2 题假成功
- [ ] 长文档检索召回率评测（Top-3 / Top-5 命中曲线）

---

**数据口径**：所有数值来自仓库内实测 JSON / Prometheus，未达标项均标注待回填。
