"""FastAPI 基础"""

import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中，解决 "No module named 'src'" 问题
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402
from typing import Optional  # noqa: E402
import uvicorn  # noqa: E402

app = FastAPI(title="AI Agent API", version="1.0.0")


# ========== 数据模型 ==========


class ChatRequest(BaseModel):
    """对话请求"""

    message: str = Field(..., min_length=1, max_length=10000)
    session_id: Optional[str] = "default"


class ChatResponse(BaseModel):
    """对话响应"""

    response: str
    session_id: str
    elapsed_seconds: float


# ========== 模拟 Agent ==========


class MockAgent:
    def chat(self, message: str, session_id: str) -> tuple[str, float]:
        import time

        start = time.time()
        # 这里替换成你的真实 Agent
        response = f"收到：{message}"
        elapsed = time.time() - start
        return response, elapsed


agent = MockAgent()


# ========== API 路由 ==========


@app.get("/")
async def root():
    return {"message": "AI Agent API", "docs": "/docs"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """对话接口"""
    try:
        response, elapsed = agent.chat(req.message, req.session_id)
        return ChatResponse(
            response=response, session_id=req.session_id, elapsed_seconds=elapsed
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # 使用 app 对象直接启动，避免 "No module named 'src'" 导入路径问题
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
