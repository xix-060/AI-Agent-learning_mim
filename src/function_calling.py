"""Function Calling 实战 - AI 工具调用"""

import json
import logging
import os
import re
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa: E402

from src.llm_client import LLMClient  # noqa: E402
from src.tools.builtin import ToolRegistry, create_default_registry  # noqa: E402

logger = logging.getLogger(__name__)


def _parse_tool_args(raw: str | None) -> dict[str, Any]:
    """容错解析 LLM 返回的工具参数 JSON。

    LLM 常返回非法 JSON（单引号、尾逗号、截断），这里多级兜底：
    空/None → {}；正则修复单引号与尾逗号；非 dict → {}。

    Args:
        raw: LLM 返回的 arguments 字符串，可能为 None/空/非法 JSON。

    Returns:
        解析后的参数 dict；解析失败返回 {}。
    """
    if not raw:
        return {}
    text = raw.strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # 修复常见错误：单引号 → 双引号、去掉尾逗号
        fixed = text.replace("'", '"')
        fixed = re.sub(r",\s*([}\]])", r"\1", fixed)
        try:
            parsed = json.loads(fixed)
        except json.JSONDecodeError:
            logger.warning("无法解析工具参数 JSON: %r", raw)
            return {}
    if not isinstance(parsed, dict):
        return {}
    return parsed


class FunctionCallingAgent:
    """支持 Function Calling 的对话 Agent。"""

    def __init__(self) -> None:
        self.client: LLMClient = LLMClient()
        self.registry: ToolRegistry = create_default_registry()
        self.system_prompt: str = (
            "你是一个有用的助手。当用户的请求需要使用工具时，必须调用工具，不要自己编造答案。\n"
            "        可用工具：\n"
            "        1. calculator - 数学计算\n"
            "        2. get_current_time - 获取当前时间"
            '（注意：涉及"现在几点""当前日期"等问题必须调用此工具，不要自己编造时间）\n'
            "        3. unit_converter - 单位换算\n"
            "        如果用户的请求不需要工具（如闲聊、自我介绍），直接用自然语言回答即可。"
        )

    def _llm_call(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
    ):  # noqa: ANN202 - 返回 OpenAI SDK 对象，类型省略
        """调用 LLM（收敛两次重复调用，统一参数）。"""
        kwargs: dict[str, Any] = {
            "model": self.client.model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools is not None:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        return self.client.client.chat.completions.create(**kwargs)

    def run(self, user_input: str) -> str:
        """完整 Function Calling 循环。"""
        print(f"\n🙋 用户：{user_input}")

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_input},
        ]

        # 步骤 1：第一次调用 LLM，看它要不要调用工具
        print("🤔 思考中...")
        response = self._llm_call(
            messages,
            tools=self.registry.get_openai_tools_schema(),
            temperature=0.0,
        )
        if not response.choices:
            return "（API 返回空，请重试）"
        msg = response.choices[0].message
        messages.append(msg.model_dump())

        # 步骤 2：如果 LLM 决定调用工具
        if msg.tool_calls:
            for tool_call in msg.tool_calls:
                func_name = tool_call.function.name
                func_args = _parse_tool_args(tool_call.function.arguments)

                print(f"🔧 调用工具: {func_name}({func_args})")

                # 本地执行工具（registry.execute 内部处理未知工具错误）
                result = self.registry.execute(func_name, **func_args)

                print(f"📤 工具结果: {result}")

                # 把结果发回给 LLM
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(result),
                    }
                )

            # 步骤 3：第二次调用 LLM，让它基于工具结果生成自然语言回复
            print("🤔 生成回复中...")
            final_response = self._llm_call(messages, temperature=0.7)
            if not final_response.choices:
                return "（API 返回空，请重试）"
            answer = final_response.choices[0].message.content
        else:
            # 没有调用工具，直接用第一次的回复
            answer = msg.content

        print(f"🤖 AI：{answer}")
        return answer


def main() -> None:
    """测试 Function Calling。"""
    agent = FunctionCallingAgent()

    # 测试用例
    test_cases = [
        "帮我算一下 (123 + 456) * 2",
        "现在几点了？",
        "把 100 公斤换算成磅",
        "sin(pi/2) + cos(0) 等于多少？",
        "100 华氏度是多少摄氏度？",
        "你好，介绍一下你自己",  # 不需要工具
    ]

    for case in test_cases:
        agent.run(case)
        print("-" * 60)


if __name__ == "__main__":
    main()
