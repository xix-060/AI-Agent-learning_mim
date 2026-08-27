"""vLLM/Ollama OpenAI 兼容客户端：吞吐与并发压测。

本机为纯 CPU，后端用 Ollama 替代 vLLM（API 形态一致，仅性能不同）。
base_url 指向 Ollama 的 OpenAI 兼容端点 http://localhost:11434/v1，
调用 DeepSeek 云端时无需改代码，只换 base_url/model/api_key。
"""

import time
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

# 指向本地 Ollama（替代 vLLM）。Ollama 不校验 api_key，但 openai 库要求非空
BASE_URL = "http://localhost:11434/v1"
MODEL = "qwen2.5:0.5b"
API_KEY = "ollama"
# 0.5b 在 CPU 上较慢，超时放宽
TIMEOUT = 180

client = OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=TIMEOUT)


def chat(prompt: str, max_tokens: int = 256) -> tuple[str, dict]:
    """单次对话调用，返回回答与耗时/吞吐指标。"""
    start = time.time()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.7,
    )
    elapsed = time.time() - start
    meta = {
        "elapsed": round(elapsed, 2),
        "prompt_tokens": resp.usage.prompt_tokens,
        "completion_tokens": resp.usage.completion_tokens,
        # 单请求生成吞吐（tokens/s），衡量 decode 速度
        "tokens_per_sec": round(resp.usage.completion_tokens / elapsed, 1),
    }
    return resp.choices[0].message.content, meta


def batch_test(num_requests: int = 10) -> float:
    """简单并发压测：ThreadPoolExecutor 并发打多个请求，测整体吞吐。

    Returns:
        整体吞吐（tokens/s）。并发能把后端打满时该值反映 batching 能力。
    """
    prompts = [f"写一段关于人工智能的介绍，第 {i} 种角度" for i in range(num_requests)]
    with ThreadPoolExecutor(max_workers=num_requests) as pool:
        start = time.time()
        results = list(pool.map(lambda p: chat(p, max_tokens=128), prompts))
        wall_time = time.time() - start
    total_tokens = sum(r[1]["completion_tokens"] for r in results)
    print(f"{'=' * 50}")
    print(f"并发 {num_requests} 请求")
    print(f"总耗时: {wall_time:.1f}s")
    print(f"总生成 token: {total_tokens}")
    print(f"整体吞吐: {total_tokens / wall_time:.1f} tokens/s")
    print(f"{'=' * 50}")
    return total_tokens / wall_time


if __name__ == "__main__":
    # 单次测试
    answer, meta = chat("一句话解释什么是 PagedAttention")
    print(f"回答: {answer}\n指标: {meta}\n")
    # 并发测试
    batch_test(10)
    batch_test(20)
