# Python 异步编程（LLM 服务必修）

## 1. 为什么 LLM 服务必须异步

LLM 调用 = 网络 IO + 长等待（秒级）。

同步模式：1 个请求等 LLM 回复时，整个线程卡死。

异步模式：等待时切去处理其他请求 —— 单进程扛住高并发。

## 2. 核心概念

- 协程：async def 定义的函数，可暂停/恢复
- await：挂起点，"等这个 IO，期间让出控制权"
- 事件循环：调度器，uvicorn 内置

## 3. 常见坑（面试题）

❌ 在 async def 里用同步阻塞调用（requests / time.sleep / 同步 openai 客户端）
→ 整个事件循环卡住，并发归零

✅ 用 httpx.AsyncClient / asyncio.sleep / AsyncOpenAI

## 4. 正确姿势

| 场景        | 错误         | 正确                  |
| :-------- | :--------- | :------------------ |
| HTTP 请求   | requests   | httpx.AsyncClient   |
| OpenAI 调用 | OpenAI()   | AsyncOpenAI()       |
| 睡眠        | time.sleep | await asyncio.sleep |
| CPU 密集任务  | 直接算        | run\_in\_executor   |
