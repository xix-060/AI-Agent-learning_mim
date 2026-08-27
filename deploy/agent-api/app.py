"""异步 Agent API（容器版）：由 src/async_api/main.py 改造

改造点：
1. 默认后端指向 compose 网络内的 ollama 服务（主机名 ollama:11434）
2. 新增 /metrics 端点 + 请求计数/延迟直方图，供 Prometheus 抓取
3. 去掉 __main__ 启动块，由 Dockerfile CMD 运行 uvicorn
"""

import os
import time
import asyncio
from contextlib import asynccontextmanager
from typing import Callable, Awaitable

from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
from prometheus_client import Counter, Histogram, make_asgi_app

# 后端预设：与 src/async_api/main.py 保持一致；容器默认 ollama（走服务名）
BACKEND_PRESETS: dict[str, dict[str, str]] = {
    "ollama": {
        "base_url": "http://ollama:11434/v1",
        "model": "qwen2.5:0.5b",
        "api_key": "ollama",
    },
    "vllm": {
        "base_url": "http://vllm:8000/v1",
        "model": "qwen2.5-7b",
        "api_key": "EMPTY",
    },
    "sglang": {
        "base_url": "http://sglang:30000/v1",
        "model": "qwen2.5-7b",
        "api_key": "EMPTY",
    },
}

BACKEND = os.getenv("ASYNC_BACKEND", "ollama").lower()
if BACKEND not in BACKEND_PRESETS:
    raise ValueError(f"未知后端 {BACKEND!r}，可选: {', '.join(BACKEND_PRESETS)}")

_preset = BACKEND_PRESETS[BACKEND]
LLM_BASE_URL = os.getenv("ASYNC_LLM_BASE_URL", _preset["base_url"])
LLM_MODEL = os.getenv("ASYNC_LLM_MODEL", _preset["model"])
LLM_API_KEY = os.getenv("ASYNC_LLM_API_KEY", _preset["api_key"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    """生命周期管理：全局共享一个 AsyncClient"""
    app.state.llm = AsyncOpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
    yield
    await app.state.llm.close()


app = FastAPI(title="异步 Agent API（容器版）", lifespan=lifespan)

# ---- Prometheus 指标 ----
REQUEST_COUNT = Counter(
    "agent_api_requests_total", "HTTP 请求总数", ["method", "path", "status"]
)
LATENCY = Histogram(
    "agent_api_request_latency_seconds",
    "请求延迟（秒）",
    ["path"],
    buckets=(0.5, 1, 2, 5, 10, 30, 60, 120, 300),
)
LLM_TOKENS = Counter("llm_tokens_total", "LLM 输出 token 总数", ["model"])


@app.middleware("http")
async def metrics_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """记录每个请求的计数与延迟"""
    start = time.perf_counter()
    response = await call_next(request)
    path = request.url.path
    REQUEST_COUNT.labels(request.method, path, response.status_code).inc()
    LATENCY.labels(path).observe(time.perf_counter() - start)
    return response


app.mount("/metrics", make_asgi_app())  # Prometheus 抓取端点


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    max_tokens: int = Field(256, ge=1, le=2048)


class ChatResponse(BaseModel):
    response: str
    elapsed: float
    tokens_per_sec: float


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """异步对话：等待 LLM 期间事件循环可服务其他请求"""
    start = time.time()
    resp = await app.state.llm.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": req.message}],
        max_tokens=req.max_tokens,
        temperature=0.7,
    )
    elapsed = time.time() - start
    tokens = resp.usage.completion_tokens
    LLM_TOKENS.labels(LLM_MODEL).inc(tokens)  # Grafana: rate(llm_tokens_total[1m])
    return ChatResponse(
        response=resp.choices[0].message.content,
        elapsed=round(elapsed, 2),
        tokens_per_sec=round(tokens / elapsed, 1),
    )


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """真流式输出：token 一到就转发"""

    async def generate():
        stream = await app.state.llm.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": req.message}],
            max_tokens=req.max_tokens,
            stream=True,
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield f"data: {chunk.choices[0].delta.content}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/batch")
async def batch_chat(messages: list[str]):
    """并发批量：asyncio.gather 同时打多个 LLM 请求"""

    async def one(m: str):
        resp = await app.state.llm.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": m}],
            max_tokens=64,
        )
        return resp.choices[0].message.content

    start = time.time()
    results = await asyncio.gather(*[one(m) for m in messages])
    return {
        "count": len(results),
        "elapsed": round(time.time() - start, 2),
        "results": results,
    }
