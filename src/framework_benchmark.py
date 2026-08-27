"""vLLM vs SGLang vs Ollama 对比测试（Agent 负载模拟）。

本机纯 CPU，vLLM/SGLang 需 GPU 起不来，脚本用 try/except 容错跳过，
只跑能起的后端（Ollama）。有 GPU 时同一脚本可同时对比三框架。
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

# Agent 场景：相同 system prompt + 变化的用户问题
SYSTEM_PROMPT = (
    "你是一个知识库助手。可用工具：search/calculator。"
    "请思考后决定是否调用工具。以下是工具说明（约 500 token）：\n"
    "search(query)：在知识库检索相关文档，返回 top-3 段落。\n"
    "calculator(expression)：计算数学表达式，返回数值结果。\n"
    "调用工具前必须说明理由，工具返回后需总结。"
)
QUESTIONS = [
    f"问题 {i}：解释概念 {c}"
    for i, c in enumerate(
        [
            "RAG",
            "Agent",
            "LoRA",
            "MCP",
            "KV Cache",
            "DPO",
            "Embedding",
            "Function Calling",
        ]
    )
]


def bench(
    base_url: str, model: str, name: str, rounds: int = 3, api_key: str = "EMPTY"
) -> float | None:
    """对指定后端做 Agent 负载压测，返回 QPS（请求/秒）。

    Args:
        base_url: OpenAI 兼容端点
        model: 模型名
        name: 后端显示名
        rounds: 压测轮数（每轮并发 8 个不同问题、相同 system prompt）
        api_key: API key（Ollama 不校验，传任意非空）

    Returns:
        QPS，后端不可用返回 None
    """
    client = OpenAI(base_url=base_url, api_key=api_key, timeout=300)

    def one_call(q: str) -> float:
        start = time.time()
        client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": q},
            ],
            max_tokens=128,
            temperature=0.7,
        )
        return time.time() - start

    # 预热一轮（触发前缀缓存写入 + 模型加载）
    try:
        one_call("warmup")
    except Exception as e:
        print(f"{name} 预热失败: {e}")
        return None

    # 并发压测：每轮 8 个不同问题、相同 system prompt
    with ThreadPoolExecutor(max_workers=8) as pool:
        start = time.time()
        for _ in range(rounds):
            list(pool.map(one_call, QUESTIONS))
        wall = time.time() - start
    qps = len(QUESTIONS) * rounds / wall
    avg = wall / (len(QUESTIONS) * rounds)
    print(f"{name:8} 总耗时 {wall:.1f}s | QPS {qps:.1f} | 平均延迟 {avg:.2f}s")
    return qps


if __name__ == "__main__":
    print("Agent 负载模拟（共享 system prompt + 并发 8）：\n")
    results: dict[str, dict] = {}
    backends = [
        ("vLLM", "http://localhost:8000/v1", "qwen2.5-7b", "EMPTY"),
        ("SGLang", "http://localhost:8001/v1", "Qwen/Qwen2.5-7B-Instruct", "EMPTY"),
        ("Ollama", "http://localhost:11434/v1", "qwen2.5:0.5b", "ollama"),
    ]
    for name, url, model, key in backends:
        try:
            qps = bench(url, model, name, api_key=key)
            results[name] = {"started": qps is not None, "qps": qps}
        except Exception as e:
            print(f"{name} 未启动: {e}")
            results[name] = {"started": False, "error": str(e)}
    # 结果落盘（不依赖 stdout，便于 CI/脚本读取）
    Path("_tmp_fwbench.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
