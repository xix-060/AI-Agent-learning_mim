# vLLM 部署性能报告（CPU + Ollama 替代方案）

> 本机为纯 CPU，vLLM 不支持 Windows 原生且需 NVIDIA GPU。按清单"无 GPU 替代路径 B"
> 用 Ollama 的 OpenAI 兼容端点（`http://localhost:11434/v1`）作为后端，
> 客户端代码（[src/vllm_client.py](../src/vllm_client.py)）与调用真 vLLM 完全同构，
> 只换 `base_url`/`model`。绝对性能与 GPU+vLLM 不可比，但相对趋势与对照结论有效。

## 环境

| 项 | 值 |
|---|---|
| 后端 | Ollama（替代 vLLM，纯 CPU） |
| 模型 | qwen2.5:0.5b（Q4_K_M，397MB） |
| OpenAI 端点 | http://localhost:11434/v1 |
| 处理器 | 100% CPU（无 GPU batching 能力） |
| 验证 | `curl http://localhost:11434/v1/models` → 返回 `qwen2.5:0.5b` |

## 实验 A：max_tokens 对生成吞吐的影响

固定单请求，改 `max_tokens`，每组 2 次取平均。

| max_tokens | 实际生成 token | 平均耗时(s) | 平均吞吐(tok/s) |
|---|---|---|---|
| 64  | 64       | 7.15  | 13.9 |
| 128 | 128      | 11.76 | 11.1 |
| 256 | 177~245* | 18.0  | 11.7 |

\* 256 组未到上限即遇 EOS 自然停止。

**结论**：单请求 decode 吞吐稳定在 11~14 tok/s，与生成长度基本无关。
decode 是访存密集型，单请求速度受 CPU 内存带宽限制，生成长一点只是"线性累加"，
单位时间产出不变。首次 64-token 那 11.44s 含模型冷加载开销，第二次降到 2.86s。

## 实验 B：并发数对吞吐的影响（关键对照实验）

固定 `max_tokens=64`，改并发数，测整体吞吐与单请求延迟。

| 并发 | wall_time(s) | 总生成 token | 整体吞吐(tok/s) | 平均延迟(s) | 失败 |
|---|---|---|---|---|---|
| 1  | 5.34   | 64   | 12.0 | 5.34  | 0 |
| 10 | 63.68  | 626  | 9.8  | 34.91 | 0 |
| 20 | 105.51 | 1280 | 12.1 | 61.86 | 0 |

**核心发现（反证 continuous batching 的价值）**：

并发 1→20，整体吞吐**几乎不变**（12.0→12.1 tok/s），但平均延迟**暴涨 12 倍**
（5.34s→61.86s）。

原因：Ollama 默认 `OLLAMA_NUM_PARALLEL=1`，请求**串行排队**；CPU 无法把多个
序列拼成大矩阵并行计算（这正是 GPU batching 的能力）。所以：

- 吞吐 = 单请求吞吐（并发再多也榨不出更多算力）
- 延迟 = 排队等待时间随并发线性增长

**与 vLLM/GPU 的对照意义**：清单目标"并发提升吞吐→PagedAttention 在起作用"。
本机测出的是反面——**并发不提升吞吐，因为没有 batching 能力**。这恰好从对照侧
印证了 vLLM 的 PagedAttention + continuous batching 解决的就是这个问题：
GPU 上把并发请求的 KV Cache 分页复用、拼批并行计算，吞吐随并发上升而延迟不暴涨。

## 实验 C：前缀缓存价值验证（Agent 场景命脉）

模拟 Agent 流量：固定约 2206 token 的长 system prompt（工具描述），连发 4 次
相同请求，看第二次起的延迟变化。

| 轮次 | prompt_tokens | 耗时(s) | 较首次降幅 |
|---|---|---|---|
| 1 | 2206 | 52.05 | -    |
| 2 | 2206 | 4.23  | -92% |
| 3 | 2206 | 3.88  | -93% |
| 4 | 2206 | 3.60  | -93% |

**结论**：第二次起延迟从 52s 骤降到 ~4s（降 92%），之后稳定在 3.6~3.9s。

原因拆解：

1. 首次 52s = 模型冷加载到内存 + Prefill 2206 token（计算密集，CPU 上极慢）+ decode
2. 第二次起：模型已在内存（`keep_alive` 保活）+ Ollama 对相同前缀的 KV cache 复用，
   Prefill 大幅缩减，剩余耗时以 decode 为主

这正是清单强调的"Agent 场景命脉"：Agent 每轮都带同样的 system prompt + 工具描述，
前缀缓存命中后能省大量 Prefill 时间。本实验用 Ollama 的 KV 复用近似验证了这一收益
（真 vLLM 的 `--enable-prefix-caching` 机制不同但收益方向一致）。

## 总结：CPU 后端的真实特性与对照价值

| 维度 | 本机实测 | GPU + vLLM（对照预期） |
|---|---|---|
| 单请求 decode | ~11 tok/s | 50~150+ tok/s |
| 并发→吞吐 | 不变（串行排队） | 显著上升（batching） |
| 并发→延迟 | 线性暴涨 | 缓慢上升 |
| 前缀缓存收益 | 第二次起 -92% | 同方向，TTFT 降幅更大 |

**面试可讲的三点**：

1. **为什么并发上不去**：CPU 无 batching 能力，多请求只能串行；vLLM 的
   PagedAttention 把 KV Cache 分页按需分配，配合 continuous batching 让
   GPU 把多序列拼批并行算，吞吐随并发上升——本实验从反面印证其价值。
2. **前缀缓存是 Agent 提速命脉**：相同 system prompt 第二次起 Prefill 骤降 92%，
   Agent 多轮对话/Few-shot/多 Agent 共享 prompt 都吃这个红利。
3. **绝对值不可比但相对趋势有效**：纯 CPU 数据不能和 4090 比，但
   "并发不提升吞吐""前缀缓存收益"两条结论方向与 GPU 一致，可作为对照基线。

## 复现方法

```bash
# 1. 确认后端就绪（Ollama 已运行 + 模型已拉取）
bash scripts/start_vllm.sh

# 2. 单次调用验证
python src/vllm_client.py    # 含单次 + 并发 10/20

# 3. 端点验证
curl http://localhost:11434/v1/models
```

切到真 vLLM（有 GPU 时）只需：`base_url` 改 `http://localhost:8000/v1`、
`model` 改 `qwen2.5-7b`、`api_key` 改 `EMPTY`，客户端代码无需改动。
