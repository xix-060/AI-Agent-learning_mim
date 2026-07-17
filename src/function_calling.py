"""Function Calling 实战 - AI ⼯具调⽤"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm_client import LLMClient
from src.tools import TOOLS_SCHEMA, TOOL_MAP


class FunctionCallingAgent:
    """⽀持 Function Calling 的对话 Agent"""

    def __init__(self):
        self.client = LLMClient()
        self.system_prompt = """你是一个有用的助手。当用户的请求需要使用工具时，必须调用工具，不要自己编造答案。
        可用工具：
        1. calculate - 数学计算
        2. get_current_time - 获取当前时间（注意：涉及"现在几点""当前日期"等问题必须调用此工具，不要自己编造时间）
        3. unit_converter - 单位换算
        如果用户的请求不需要工具（如闲聊、自我介绍），直接用自然语言回答即可。"""

    def run(self, user_input: str) -> str:
        """完整 Function Calling 循环"""
        print(f"\n🙋 ⽤⼾：{user_input}")

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_input},
        ]

        # 步骤 1：第⼀次调⽤ LLM，看它要不要调⽤⼯具
        print("🤔 思考中...")
        response = self.client.client.chat.completions.create(
            model=self.client.model,
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
            temperature=0.0,
        )

        msg = response.choices[0].message
        messages.append(msg.model_dump())

        # 步骤 2：如果 LLM 决定调⽤⼯具
        if msg.tool_calls:
            for tool_call in msg.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)

                print(f"🔧 调⽤⼯具: {func_name}({func_args})")

                # 本地执⾏函数
                func = TOOL_MAP.get(func_name)
                if func:
                    result = func(**func_args)
                else:
                    result = f"错误：未知⼯具 {func_name}"

                print(f"📤 ⼯具结果: {result}")

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
            final_response = self.client.client.chat.completions.create(
                model=self.client.model,
                messages=messages,
                temperature=0.7,
            )
            answer = final_response.choices[0].message.content
        else:
            # 没有调用工具，直接用第一次的回复
            answer = msg.content

        print(f"🤖 AI：{answer}")
        return answer


def main():
    """测试 Function Calling"""
    agent = FunctionCallingAgent()

    # 测试⽤例
    test_cases = [
        "帮我算⼀下 (123 + 456) * 2",
        "现在几点了？",
        "把 100 公斤换算成磅",
        "sin(pi/2) + cos(0) 等于多少？",
        "100 华氏度是多少摄氏度？",
        "你好，介绍⼀下你自己",  # 不需要⼯具
    ]

    for case in test_cases:
        agent.run(case)
        print("-" * 60)


if __name__ == "__main__":
    main()
