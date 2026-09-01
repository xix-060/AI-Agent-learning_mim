# 第 9 周复盘

## 1. 三大核心收获
1. KV Cache/PagedAttention：为什么并发能上去（原理+模拟+实测三连）
   - 会算了：Qwen2.5-7B FP16(GQA) = 2×28层×4头×128维×2B = **56KB/token**，10K 上下文单请求 ≈560MB，100 并发×10K = 56GB 直接爆显存 → 这就是 PagedAttention 存在的理由
   - 实测从反面印证：本机 CPU 并发 1→20，吞吐纹丝不动（12.0→12.1 tok/s）、延迟暴涨 12 倍（5.3s→61.9s）——没有 batching 能力就是这样；vLLM 分页+拼批解决的正是这个
2. 异步编程：AsyncOpenAI + gather，10 请求提速 **1.5** 倍
   - 实测（agent-api, 10 请求 max_tokens=32）：同步逐个 44.8s → 异步并发 30.2s = **1.5x**
   - 诚实的结论：1.5x 而不是 5-10x，因为后端是 Ollama CPU 串行排队（OLLAMA_NUM_PARALLEL=1），API 层异步只能榨干排队空闲，天花板在 batching
   - 坑实测过：async 里用同步客户端 = 事件循环卡死；`localhost` 在 Windows 先解析 IPv6 命中 wslrelay 黑洞，压测必须写 `127.0.0.1`
3. 可观测：压测找拐点 + Prometheus/Grafana 曲线
   - Locust 阶梯压测 → 埋点（计数/延迟直方图/token）→ Prometheus 抓取 → Grafana 四面板（QPS/P99/P95/Token）
   - 项目 1 全链路冒烟 5 项 PASS，曲线能对应到每一次真实请求

## 2. 我的性能数据
- 拐点并发 / 峰值 QPS / P99（本机 CPU + Ollama qwen2.5:0.5b，无 GPU）：
  - 拐点：**并发=1 即饱和**——并发 1→20 QPS 恒定 ~0.19 req/s（无 batching 榨不出增量），延迟 5.34s→61.86s 线性暴涨 12 倍
  - Locust 对 agent-api 压测汇总：27 请求，QPS 0.12，**P50 26s / P95 67s / P99 73s**，失败率 0%
  - Grafana 面板实测：QPS ~0.49 req/s，P99 峰值 1.83min，P95 峰值 1.33min，Token 吞吐 6.5 ops/s
  - 对照结论：P99<1s 的最大并发 = **0**（CPU 后端分钟级延迟），需 GPU+vLLM 回填
- 前缀缓存对 Agent 流量的收益：**-92%**
  - 2206 token 长 system prompt（Agent 工具描述）连发 4 次：52.05s → 4.23s（-92%）→ 3.88s → 3.60s
  - 机制：KV 复用省掉重复 Prefill；Agent 每轮共享 prompt 正好吃这个红利（真 vLLM `--enable-prefix-caching` / SGLang Radix Tree 方向一致）

## 3. 面试必答题自测
- [x] 7B 模型 KV Cache 怎么算？
  → 公式 `2 × 层数 × KV头数 × 头维度 × 精度字节 × seq_len × batch`；Qwen2.5-7B FP16 GQA：2×28×4×128×2B = 56KB/token；10K 上下文 ≈560MB/请求，100 并发 = 56GB → 爆显存
- [x] Prefill 和 Decode 瓶颈有何不同？
  → Prefill 一次吃整个 prompt，大矩阵乘 = **计算密集，瓶颈算力**；Decode 逐 token 生成，每步读全量 KV Cache = **访存密集，瓶颈显存带宽**
- [x] vLLM 和 SGLang 选型逻辑？
  → 通用生产首选 vLLM（PagedAttention 分页，生态最全）；Agent 高并发/多轮共享 prompt 选 SGLang（Radix Tree 前缀树，前缀命中更强）；小模型本地开发 Ollama
- [x] async 里写同步库会怎样？
  → 整个事件循环被卡死，并发归零（所有协程排队等这一个阻塞调用）；必须用 httpx.AsyncClient / AsyncOpenAI / asyncio.sleep，CPU 密集丢 run_in_executor
- [x] 为什么只看平均延迟会被骗？
  → 平均值被长尾稀释：平均 1s 可能 P99=8s；P99 才是告警和用户投诉的来源（慢的那 1% 往往是超时重试/排队雪崩的前兆）

## 4. 下两周（项目冲刺）规划
项目 B 选什么领域？（法律/医疗/金融/学术）

**建议：学术（论文问答助手）**——理由：
1. 数据零成本：arXiv PDF 公开可批量获取，不用爬虫对抗和脱敏
2. 与自己场景契合：本身就是学习笔记/论文阅读流，做出来自己天天用，demo 有说服力
3. 与项目 1 差异化：项目 1 是通用 RAG，项目 B 可叠加 Agent 能力（多轮检索规划、引用溯源、长文档分块策略对比）
4. 备选：法律（数据公开但清洗重）、医疗（数据敏感获取难）、金融（公告类可行，但与项目 1 重合度偏高）

> 待确认：领域最终拍板后，两周拆解为「周 1：数据管道 + Chunking 实验；周 2：Agent 化 + 压测部署」
