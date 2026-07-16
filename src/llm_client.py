"""LLM 客户端封装 - 第 1 周核心产出"""

import os
import time
from dotenv import load_dotenv
from openai import OpenAI
from src.models import Message, ChatResponse, RoleEnum

load_dotenv()


class LLMClient:
    """LLM 客户端：封装 OpenAI 兼容接口"""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ):
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.base_url = base_url or os.getenv("LLM_BASE_URL")
        self.model = model or os.getenv("LLM_MODEL", "deepseek-chat")

        if not self.api_key:
            raise ValueError("请设置 LLM_API_KEY 环境变量")

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def chat(
        self, messages: list[Message], temperature: float = 0.7, max_tokens: int = 2048
    ) -> ChatResponse:
        """发送对话请求"""
        start = time.time()

        # 转换为 API 格式
        api_messages = [{"role": m.role.value, "content": m.content} for m in messages]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=api_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        elapsed = time.time() - start
        return ChatResponse(
            content=response.choices[0].message.content,
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            elapsed_seconds=round(elapsed, 2),
        )

    def simple_chat(
        self, user_input: str, system_prompt: str = "你是一个有用的助手"
    ) -> str:
        """简化的单轮对话"""
        messages = [
            Message(role=RoleEnum.SYSTEM, content=system_prompt),
            Message(role=RoleEnum.USER, content=user_input),
        ]
        return self.chat(messages).content
