# 部署指南 — knowledge_agent 生产化

项目 1（RAG 知识库 Agent）的容器化部署与监控方案。

## 架构

```
用户 → Locust/前端 → FastAPI(ka-api 容器, 8081) → vLLM(7B, 宿主机 GPU)
                                        ↓
                                  Chroma 持久卷（向量库/上传文档）
                                        ↓
                            Prometheus (9091) → Grafana (3001) 监控
```

> 本机（无 GPU）验证时 LLM 后端为宿主机 Ollama（qwen2.5:0.5b, CPU）；
> GPU 服务器上按「三环境切换」改为 vLLM，架构不变。

## 三环境配置

| 文件 | LLM | Embedding | 场景 |
|------|-----|-----------|------|
| `.env.dev` | DeepSeek `deepseek-chat` | DashScope | 日常开发，便宜快速 |
| `.env.prod` | 本地 vLLM `Qwen2.5-7B-Instruct` | DashScope | GPU 生产，数据不出域 |
| `.env.docker` | `host.docker.internal:11434/v1`（Ollama）或宿主机 vLLM | DashScope | 容器网络 |

- 三个文件均为**占位 key 模板**（会提交 git）；真实 key 放 `knowledge_agent/.env`（已被 gitignore）
- DeepSeek 不提供 embedding/reranker，这两项统一走 DashScope
- Reranker 无 key 时自动降级（跳过重排序，不报错）

## 一键启动

```bash
# 1. 宿主机启动 vLLM（生产/GPU；本机验证跳过，用 Ollama）
bash scripts/start_vllm.sh

# 2. 容器启动 API + 监控
cd knowledge_agent/deploy && docker compose up -d --build

# 3. 验证
curl http://localhost:8081/health
```

## 全链路冒烟测试

```bash
# 1. 上传文档（向量化入库）
curl -X POST http://localhost:8081/upload -F "file=@data/sample.txt"

# 2. 提问（RAG 检索 + 生成）
curl -X POST http://localhost:8081/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "文档的主要内容是什么?", "session_id": "smoke"}'

# 3. Grafana 看曲线
#    http://localhost:3001 （admin/admin）
#    面板查询: rate(ka_requests_total[1m]) / ka_request_latency_seconds / ka_llm_tokens_total
```

## 监控指标

| 指标名 | 含义 |
|--------|------|
| `ka_requests_total` | HTTP 请求总数（method/path/status 标签）|
| `ka_request_latency_seconds` | 请求延迟直方图（bucket 到 300s）|
| `ka_llm_tokens_total` | LLM 输出 token 估算值（中文 2 字符/token）|

## 性能基线（实测）

测试环境：本机 CPU + Ollama qwen2.5:0.5b（无 GPU），数据来自 Prometheus 实测（3 次 /chat + 1 次 /upload）。

| 指标 | 值 |
|------|-----|
| QPS @ P99<1s | 0（CPU 后端 P99 落在 60~120s 档，远超 1s）|
| 平均延迟（/chat）| 56.7s（3 次实测：34.7s / ~60s / ~75s，单次 120s 内）|
| Token 吞吐 | ≈2.0 tokens/s（337 tokens / 170s 总时长）|
| /upload 向量化延迟 | 13.2s（1KB 文档，DashScope embedding 网络往返为主）|

> CPU 后端的定位是**验证链路正确性**；P99<1s 需 GPU + vLLM，
> 上线前用同款 Locust 脚本（改 host 与 QPS 目标）回填生产基线。

## 端口规划

| 服务 | 端口 | 说明 |
|------|------|------|
| ka-api | 8081 | 与根 deploy/ 全栈的 8080 区分 |
| prometheus | 9091 | |
| grafana | 3001 | |

## 常见问题

- **`host.docker.internal` 连不上**：Linux 需 compose 的 `extra_hosts`（已配置）；Windows/macOS Docker Desktop 自带
- **上传 500**：检查 `.env.docker` 的 EMBEDDING key（向量化必需）
- **问答不带文档内容**：先 `/upload` 或确认 Chroma 卷里有数据（`docker compose exec ka-api ls /app/knowledge_agent/data/chroma_db`）
