# KV Cache 与 PagedAttention

## 1. 为什么需要 KV Cache？（核心必背）

自回归生成：每生成 1 个 token，Attention 要对所有历史 token 算 Q·K。
不做缓存的话，生成第 N 个 token 要重算前 N-1 个的 K/V —— O(N²) 浪费。

KV Cache：把每层的 K/V 存下来，新 token 只算自己的 Q/K/V。
代价：显存。KV Cache 大小 = 2 × 层数 × KV头数 × 头维度 × 序列长度 × 精度字节 × batch

## 2. KV Cache 有多大？（面试必算）

以 Qwen2.5-7B、FP16、batch=1 为例：

- 层数 L=28，KV 头数=4（GQA），头维度=128
- 每 token 每层：2 × 4 × 128 × 2字节 = 2KB
- 全部层：2KB × 28 = 56KB/token
- 10K token 上下文：56KB × 10000 ≈ 560MB（单请求！）
- 100 并发 × 10K：56GB —— 爆显存了！
  → 这就是 PagedAttention 要解决的问题

## 3. PagedAttention（vLLM 论文核心）

操作系统"虚拟内存分页"思想搬到 KV Cache：

| <br />  | 传统（连续分配）         | PagedAttention                         |
| :------ | :--------------- | :------------------------------------- |
| 分配方式    | 按最大长度预留整块        | 按需分页（每页16/32 token）                    |
| 浪费      | 内部+外部碎片 \~60-80% | <4%                                    |
| 并发容量    | 小                | 大 2-4 倍                                |
| 更长的输出共享 | 难                | copy-on-write 轻松实现（如 beam search 共享前缀） |

## 4. Prefill vs Decode 两阶段（SGLang 对比要用）

| 阶段           | 干什么            | 特点                   | 瓶颈   |
| :----------- | :------------- | :------------------- | :--- |
| Prefill（预填充） | 一次性吃掉整个 prompt | 计算密集（大矩阵乘）           | 算力   |
| Decode（解码）   | 逐 token 生成     | 访存密集（每步读全量 KV Cache） | 显存带宽 |

## 5. 论文阅读指引

[Efficient Memory Management for LLM Serving with PagedAttention (SOSP'23)](https://arxiv.org/abs/2309.06180)

只看：摘要 → 图1（碎片浪费）→ 图3（分页结构）→ 图8（吞吐对比）
