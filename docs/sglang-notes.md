# SGLang 核心技术（Radix Tree 缓存、Token Healing）

## 1. Radix Tree（基数树）前缀缓存

vLLM 的 prefix caching：按前缀哈希匹配。
SGLang：把所有请求的 KV Cache 组织成一棵 Radix Tree，**自动复用任何公共前缀**。
Agent 场景的杀手锏：

- 多轮对话：历史前缀全命中
- Few-shot：示例部分命中
- 多 Agent 共享同一 system prompt：全员命中
  实测 Agent 负载下 SGLang 吞吐常比 vLLM 高 20-60%（取决于命中率）

## 2. Token Healing（词元修复）

问题：用户 prompt 以 "Hi!" 结尾，tokenizer 可能切成 \["Hi", "!"]，
但约束解码时若强制以 "!" 开头生成，可能拼出非法 token。
SGLang 的解法：回退 prompt 最后一个 token，让生成阶段重新组合，消除边界伪影。
适用：结构化输出（JSON/正则约束）时保证语法正确。

## 3. 其他特性

- Frontend DSL：一套 DSL 描述复杂调用流（分支/并行/多轮）
- zero-overhead scheduler

## 4. vLLM vs SGLang 怎么选（2025）

| 维度    | vLLM    | SGLang          |
| :---- | :------ | :-------------- |
| 社区/生态 | 最大，事实标准 | 增长快，Agent 场景口碑好 |
| 前缀缓存  | 支持      | Radix Tree 更强   |
| 约束输出  | 支持      | + Token Healing |
| 稳定性   | 更成熟     | 迭代快偶有坑          |
| 建议    | 通用首选    | Agent 高并发可测     |

#### → 生产决策方式：拿自己的真实流量压测，谁高用谁

## 5. CPU 实测印证（Ollama 近似 Radix Tree 收益）

本机纯 CPU 跑不了 SGLang，用 Ollama 的 KV 复用近似验证"前缀缓存"在
Agent 场景的收益（数据见 [vllm-performance.md](./vllm-performance.md) 实验 C）。

固定约 2206 token 的长 system prompt（工具描述），连发 4 次相同请求：

| 轮次 | 耗时(s) | 较首次降幅 |
| :--- | :--- | :--- |
| 1 | 52.05 | -    |
| 2 | 4.23  | -92% |
| 3 | 3.88  | -93% |
| 4 | 3.60  | -93% |

- 首次 52s = 模型冷加载 + Prefill 2206 token（计算密集，CPU 上极慢）+ decode
- 第二次起骤降到 ~4s：模型已在内存（keep_alive）+ 前缀 KV 复用，Prefill 大幅缩减

这正是 Radix Tree 卖点在 Agent 场景的体现：多轮对话 / Few-shot /
多 Agent 共享 system prompt，前缀命中后 Prefill 几乎归零。SGLang 把所有请求
的 KV 组织成基数树自动复用任意公共前缀，命中率比 vLLM 的哈希匹配更高，
Agent 负载下吞吐常高 20-60%。

> Ollama 的前缀复用是 per-request 级别；SGLang 的 Radix Tree 是跨请求的
> 全局前缀树，命中率上限更高。但收益方向一致，本实验印证的是"前缀缓存
> 对 Agent 流量的价值"这一共性。
