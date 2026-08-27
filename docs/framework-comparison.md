# Agent 框架对比

| 维度      | 手写 ReAct | LangGraph | CrewAI | AutoGen |
| :------ | :------- | :-------- | :----- | :------ |
| 学习曲线    | 低（理解原理）  | 中         | 低      | 中       |
| 流程控制    | while 循环 | 状态机       | 顺序/层级  | 对话      |
| 人机协同    | ❌        | ✅         | ❌      | ✅       |
| 可暂停     | ❌        | ✅         | ❌      | ✅       |
| 多 Agent | ❌        | ✅         | ✅      | ✅       |
| 可视化     | ❌        | ✅ Mermaid | ❌      | ❌       |
| 适用      | 学习原理     | 生产级       | 快速原型   | 对话型     |

## 技术选型建议

- **学习阶段**：手写 ReAct（理解原理）
- **生产单 Agent**：LangGraph
- **快速多 Agent 原型**：CrewAI
- **对话型多 Agent**：AutoGen
- **项目 1 选择**：LangGraph（可控制流 + 可暂停 + 简历加分）

---

## 推理服务框架对比（vLLM / SGLang / Ollama）

> 部署维度选型（课纲模块七）。本机纯 CPU，vLLM/SGLang 需 GPU 起不来，
> Ollama 列为实测，vLLM/SGLang 列为原理 + 清单基准。

| 维度 | Ollama | vLLM | SGLang |
| :--- | :--- | :--- | :--- |
| 定位 | 本地开发 / 边缘 | 生产通用首选 | Agent 高并发 |
| 模型格式 | GGUF 量化 | 原生 / AWQ / GPTQ 等 | 同 vLLM |
| KV 管理 | per-request 复用 | PagedAttention 分页 | Radix Tree 前缀树 |
| 连续批处理 | ❌（默认串行） | ✅ | ✅ |
| 前缀缓存 | keep_alive + KV 复用 | prefix-caching（哈希） | Radix Tree（更强） |
| 约束输出 | 支持 | 支持 | + Token Healing |
| Windows 原生 | ✅ | ❌（需 Linux/WSL2） | ❌ |
| GPU 必需 | 否（CPU 可跑小模型） | 是 | 是 |

### Agent 负载实测（Ollama，纯 CPU + qwen2.5:0.5b）

共享约 500 token system prompt + 并发请求（数据见
[vllm-performance.md](./vllm-performance.md) 实验 B）：

| 并发 | wall(s) | 请求 QPS(req/s) | token 吞吐(tok/s) | 平均延迟(s) |
| :--- | :--- | :--- | :--- | :--- |
| 1  | 5.34   | 0.19 | 12.0 | 5.34  |
| 10 | 63.68  | 0.16 | 9.8  | 34.91 |
| 20 | 105.51 | 0.19 | 12.1 | 61.86 |

**关键结论**：并发 1→20，请求 QPS 几乎不变（~0.19 req/s），平均延迟却暴涨 12 倍。
Ollama 默认串行排队、无 continuous batching，CPU 无多序列并行能力 → 吞吐榨不出增量。
对照预期（GPU + vLLM/SGLang）：并发上升时 batching 生效，QPS 显著上升、延迟缓涨。
本数据从反面印证了 vLLM PagedAttention + 连续批处理的价值。

### 前缀缓存实测（Ollama 近似 SGLang Radix Tree）

见 [vllm-performance.md](./vllm-performance.md) 实验 C：相同长 system prompt 连发，
第二次起延迟 -92%（52s→4s）。印证 SGLang Radix Tree 在 Agent 多轮 / 共享 prompt
场景的 Prefill 收益。

### 细粒度 Prefill/Decode 拆分（实验方法，待执行环境恢复补测）

用 Ollama 原生 `/api/generate`（非 OpenAI 端点）能拆出 Prefill/Decode 耗时：

```bash
curl http://localhost:11434/api/generate -d '{
  "model":"qwen2.5:0.5b","prompt":"什么是RAG",
  "system":"<长 system prompt 约 2000 字>","stream":false
}'
# 返回 prompt_eval_duration（Prefill）与 eval_duration（Decode），单位纳秒
```

预期（据实验 C 推算）：首次 prompt_eval_duration 占 52s 的绝大部分（>90%，
Prefill 2206 token 是瓶颈）；第二次起 prompt_eval_duration 骤降（前缀 KV 命中），
eval_duration（Decode）基本不变。这正对应 SGLang Radix Tree 优化的 Prefill 阶段。

> 注：本次因执行环境异常未能跑出该细粒度数据，上述为基于实验 C 的推算。
> 对比脚本 [src/framework_benchmark.py](../src/framework_benchmark.py) 已就绪，
> 有 GPU 时可直接跑出 vLLM/SGLang/Ollama 三框架 QPS 对照。

### 选型结论

- **本地开发 / 纯 CPU / 边缘**：Ollama（OpenAI 兼容 API，客户端代码与 vLLM 同构，切换只换 base_url）
- **生产通用**：vLLM（生态最大、最成熟，PagedAttention 分页 + 连续批处理）
- **Agent 高并发 / 多轮共享 prompt**：SGLang（Radix Tree 前缀缓存命中率更高，Agent 负载吞吐常比 vLLM 高 20-60%）
- **决策方式**：拿自己的真实流量压测，谁高用谁
