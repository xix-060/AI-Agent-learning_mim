"""异步高并发 Agent API：async/await + AsyncOpenAI + 流式"""

import time
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

# 后端预设：本机纯 CPU 用 Ollama；vLLM / SGLang 预设保留（服务器 GPU 场景）
BACKEND_PRESETS: dict[str, dict[str, str]] = {
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "model": "qwen2.5:0.5b",
        "api_key": "ollama",
    },
    "vllm": {
        "base_url": "http://localhost:8000/v1",
        "model": "qwen2.5-7b",
        "api_key": "EMPTY",
    },
    "sglang": {
        "base_url": "http://localhost:30000/v1",
        "model": "qwen2.5-7b",
        "api_key": "EMPTY",
    },
}

# 选后端：ASYNC_BACKEND=ollama|vllm|sglang，单个字段仍可用 ASYNC_LLM_* 覆盖
BACKEND = os.getenv("ASYNC_BACKEND", "ollama").lower()
if BACKEND not in BACKEND_PRESETS:
    raise ValueError(f"未知后端 {BACKEND!r}，可选: {', '.join(BACKEND_PRESETS)}")

_preset = BACKEND_PRESETS[BACKEND]
LLM_BASE_URL = os.getenv("ASYNC_LLM_BASE_URL", _preset["base_url"])
LLM_MODEL = os.getenv("ASYNC_LLM_MODEL", _preset["model"])
LLM_API_KEY = os.getenv("ASYNC_LLM_API_KEY", _preset["api_key"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    """生命周期管理：全局共享一个 AsyncClient（关键！）"""
    app.state.llm = AsyncOpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
    yield
    await app.state.llm.close()


app = FastAPI(title="异步 Agent API", lifespan=lifespan)


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
            stream=True,  # 流式
        )
        async for chunk in stream:  # 逐 token 消费
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


if __name__ == "__main__":
    # 直接传 app 对象：`python src/async_api/main.py` 和
    # `python -m src.async_api.main` 两种方式都能启动（字符串形式会因找不到 src 包而报错）
    uvicorn.run(app, host="0.0.0.0", port=8080)
