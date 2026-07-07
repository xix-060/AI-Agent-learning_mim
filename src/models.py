"""Pydantic 数据模型 - 第 1 周练习"""

from pydantic import BaseModel, Field, field_validator
from enum import Enum
from datetime import datetime


class RoleEnum(str, Enum):
    """对话⻆⾊"""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Message(BaseModel):
    """单条对话消息"""

    role: RoleEnum
    content: str = Field(..., min_length=1, max_length=10000)
    timestamp: datetime = Field(default_factory=datetime.now)

    @field_validator("content")
    @classmethod
    def content_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("content 不能为空⽩")
        return v


class ChatRequest(BaseModel):
    """LLM 对话请求"""

    messages: list[Message] = Field(..., min_length=1)
    model: str = "deepseek-chat"
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(2048, ge=1, le=8192)


class ChatResponse(BaseModel):
    """LLM 对话响应"""

    content: str
    model: str
    usage: dict
    elapsed_seconds: float
