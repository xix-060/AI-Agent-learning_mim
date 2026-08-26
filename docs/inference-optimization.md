# 推理优化技术

## 1. KV Cache

Transformer 自回归生成时，每生成一个 token 都要重新计算所有历史 token 的 K/V。

KV Cache 把历史 K/V 缓存起来，避免重复计算。


代价：显存增加（与序列长度成正比）

## 2. PagedAttention（vLLM）

KV Cache 的内存管理优化：

- 把 KV Cache 分成固定大小的"页"
- 按需分配，减少碎片
- 支持更高的 batch size
- 吞吐量提升 2-4 倍

## 3. 常用推理框架

| 框架           | 特点                 | 适用   |
| :----------- | :----------------- | :--- |
| Transformers | 基础，功能全             | 开发调试 |
| vLLM         | PagedAttention，高吞吐 | 生产部署 |
| SGLang       | Radix Tree 缓存，极速   | 高并发  |
| Ollama       | 本地一键部署             | 个人使用 |
| TensorRT-LLM | NVIDIA 优化          | 极致性能 |

## 4. 量化推理

- GPTQ：训练后量化，4bit/8bit
- AWQ：激活感知量化，精度更好
- GGUF：llama.cpp 格式，CPU/GPU 混合

## 5. 推理加速技巧

- 批量推理（batch）
- 流式输出（用户体验好）
- 投机解码（Speculative Decoding）
- 前缀缓存（相同 system prompt 复用）
