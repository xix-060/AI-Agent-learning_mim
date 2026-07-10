"""第⼀个 LLM 程序 - 第 1 周⾥程碑"""

from src.llm_client import LLMClient


def main():
    print("🤖 AI Agent 学习 - 第 1 个 LLM 程序")
    print("=" * 50)

    client = LLMClient()

    # 单轮对话
    question = "⽤⼀句话解释什么是 AI Agent"
    print(f"\n🙋 我：{question}")

    answer = client.simple_chat(question)
    print(f"\n🤖 AI：{answer}")

    # 多轮对话
    print("\n" + "=" * 50)
    print("进⼊多轮对话（输⼊ quit 退出）")
    print("=" * 50)

    history = []
    while True:
        user_input = input("\n🙋 我：")
        if user_input.strip().lower() in ("quit", "exit", "q"):
            break

        from src.models import Message, RoleEnum

        history.append(Message(role=RoleEnum.USER, content=user_input))

        response = client.chat(history)
        print(f"\n🤖 AI：{response.content}")
        print(
            f" ⏱ {response.elapsed_seconds}s | 💰 {response.usage['total_tokens']} tokens"
        )

        history.append(Message(role=RoleEnum.ASSISTANT, content=response.content))


if __name__ == "__main__":
    main()
