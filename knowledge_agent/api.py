"""知识库 Agent API 服务"""

import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中，解决 "No module named 'knowledge_agent'" 问题
# api.py 位于 knowledge_agent/ 下，上一层即项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException, UploadFile, File  # noqa: E402
from fastapi.responses import StreamingResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402
from typing import Optional  # noqa: E402
import uvicorn  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
import json  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from prometheus_client import Counter, Histogram, make_asgi_app  # noqa: E402

load_dotenv()

app = FastAPI(title="知识库 Agent API", version="1.0.0")

# ========== Prometheus 指标 ==========
REQUEST_COUNT = Counter(
    "ka_requests_total", "HTTP 请求总数", ["method", "path", "status"]
)
LATENCY = Histogram(
    "ka_request_latency_seconds",
    "请求延迟（秒）",
    ["path"],
    buckets=(0.5, 1, 2, 5, 10, 30, 60, 120, 300),
)
LLM_TOKENS = Counter(
    "ka_llm_tokens_total", "LLM 输出 token 总数（中文按 2 字符/token 估算）", ["path"]
)


@app.middleware("http")
async def metrics_middleware(request, call_next):
    """记录每个请求的计数与延迟"""
    start = time.perf_counter()
    response = await call_next(request)
    path = request.url.path
    REQUEST_COUNT.labels(request.method, path, response.status_code).inc()
    LATENCY.labels(path).observe(time.perf_counter() - start)
    return response


app.mount("/metrics", make_asgi_app())  # Prometheus 抓取端点


# ========== 初始化 Agent ==========

# 延迟导入避免启动慢
_agent = None
_rag = None


def get_agent():
    global _agent, _rag
    if _agent is None:
        from knowledge_agent.src.agent import KnowledgeAgent

        _agent = KnowledgeAgent()
        _rag = _agent.rag
    return _agent, _rag


# ========== 数据模型 ==========


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000)
    session_id: Optional[str] = "default"


class ChatResponse(BaseModel):
    response: str
    session_id: str
    elapsed_seconds: float
    tools_used: list[str] = []


class ImportRequest(BaseModel):
    file_path: str


# ========== 路由 ==========


@app.get("/")
async def root():
    return {
        "service": "Knowledge Agent API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": ["/chat", "/import", "/health"],
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """对话"""
    try:
        agent, _ = get_agent()
        start = time.time()
        response = agent.chat(req.message, req.session_id)
        elapsed = time.time() - start
        LLM_TOKENS.labels("/chat").inc(max(1, len(response) // 2))  # 中文近似估算

        return ChatResponse(
            response=response,
            session_id=req.session_id,
            elapsed_seconds=round(elapsed, 2),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/import")
async def import_document(req: ImportRequest):
    """导入文档"""
    try:
        _, rag = get_agent()
        result = rag.import_document(req.file_path)
        return {
            **result,
            "file_path": req.file_path,
            "status": "success" if result.get("success") else "failed",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """上传文档"""
    try:
        # 用 config 的绝对路径（容器内 CWD 不同于本地，相对路径会写错位置）
        from knowledge_agent.src.config import UPLOAD_DIR

        os.makedirs(UPLOAD_DIR, exist_ok=True)
        file_path = str(UPLOAD_DIR / file.filename)

        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        _, rag = get_agent()
        result = rag.import_document(file_path)

        return {
            "filename": file.filename,
            "file_path": file_path,
            "success": result.get("success", False),
            "chunks": result.get("chunks", 0),
            "source": result.get("source", file.filename),
            "status": "success" if result.get("success") else "failed",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ========== 流式输出 ==========


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """流式对话"""

    async def generate():
        agent, _ = get_agent()
        # 这里简化：实际可以用 LangGraph 的 stream
        response = agent.chat(req.message, req.session_id)
        # 模拟流式
        chunk_size = 10
        for i in range(0, len(response), chunk_size):
            chunk = response[i : i + chunk_size]
            yield f"data: {json.dumps({'chunk': chunk})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
