"""上下文工程实验：观察上下文长度对成本和响应的影响"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm_client import LLMClient
from src.models import Message, RoleEnum


def test_context_length():
    """测试不同上下文长度的影响"""
    client = LLMClient()

    lengths = [1, 5, 20, 50]

    results = []
    for num_turns in lengths:
        history = []
        for i in range(num_turns):
            history.append(
                {"role": "user", "content": f"这是第{i+1}轮对话，请记住数字 {i+1}。"}
            )
            history.append(
                {"role": "assistant", "content": f"好的，我记住了数字{i+1}。"}
            )

        messages = [Message(role=RoleEnum.SYSTEM, content="你是助手")]
        for h in history:
            messages.append(Message(role=h["role"], content=h["content"]))
        messages.append(
            Message(role=RoleEnum.USER, content="我问的是第几轮？请只回答数字")
        )

        response = client.chat(messages)
        print(
            f"轮数={num_turns}: 输入token={response.usage['prompt_tokens']}, "
            f"输出token={response.usage['completion_tokens']}, "
            f"总token={response.usage['total_tokens']}, "
            f"耗时={response.elapsed_seconds}s, "
            f"成本估计=${response.usage['total_tokens'] * 0.001 / 1000:.4f}"
        )
        results.append(response.usage)

    return results


def test_context_overflow():
    """测试上下文溢出（构造超长上下文）"""
    client = LLMClient()

    long_text = "今天天气真好。" * 5000
    messages = [
        Message(role=RoleEnum.SYSTEM, content="你是助手"),
        Message(role=RoleEnum.USER, content=f"{long_text}\n\n请用一句话总结上面的内容"),
    ]

    response = client.chat(messages)
    print("\n超长上下文测试:")
    print(f"  输入token={response.usage['prompt_tokens']}")
    print(f"  输出token={response.usage['completion_tokens']}")
    print(f"  总token={response.usage['total_tokens']}")
    print(f"  成本估计=${response.usage['total_tokens'] * 0.001 / 1000:.4f}")
    print(f"  耗时={response.elapsed_seconds}s")


if __name__ == "__main__":
    print("实验 1：上下文长度对成本的影响")
    test_context_length()
    print("\n实验 2：超长上下文测试")
    test_context_overflow()
