"""Ollama 本地模型 API 调用"""

import requests


OLLAMA_BASE = "http://localhost:11434"


def list_models():
    """列出本地模型"""
    resp = requests.get(f"{OLLAMA_BASE}/api/tags")
    models = resp.json()["models"]
    for m in models:
        size_mb = m["size"] / 1024 / 1024
        print(f"  {m['name']} ({size_mb:.0f}MB)")
    return models


def chat(model: str, message: str, system: str = "你是助手") -> str:
    """对话"""
    resp = requests.post(
        f"{OLLAMA_BASE}/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": message},
            ],
            "stream": False,
        },
    )
    return resp.json()["message"]["content"]


def chat_with_langchain(model: str = "qwen2.5:0.5b"):
    """用 LangChain 调 Ollama"""
    from langchain_community.chat_models import ChatOllama

    llm = ChatOllama(model=model, base_url=OLLAMA_BASE)

    from langchain_core.messages import HumanMessage, SystemMessage

    response = llm.invoke(
        [
            SystemMessage(content="你是助手"),
            HumanMessage(content="用一句话解释 RAG"),
        ]
    )
    print(f"[LangChain] {response.content}")
    return response.content


def compare_local_vs_cloud():
    """对比本地 vs 云端"""
    import os
    from dotenv import load_dotenv

    load_dotenv()

    question = "什么是 Attention 机制？"

    # 本地
    print("🏠 本地模型（Qwen2.5-0.5B）：")
    local_answer = chat("qwen2.5:0.5b", question)
    print(f"  {local_answer[:150]}...")

    # 云端
    print("\n☁️ 云端模型（DeepSeek）：")
    from openai import OpenAI

    client = OpenAI(
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
    )
    cloud_resp = client.chat.completions.create(
        model=os.getenv("LLM_MODEL", "deepseek-chat"),
        messages=[{"role": "user", "content": question}],
        max_tokens=200,
    )
    print(f"  {cloud_resp.choices[0].message.content[:150]}...")


def main():
    print("📋 本地模型列表：")
    list_models()

    print("\n" + "=" * 60)
    print("💬 对话测试")
    print("=" * 60)

    questions = [
        "你好",
        "用 Python 写 hello world",
        "解释什么是 RAG",
    ]

    for q in questions:
        print(f"\n❓ {q}")
        answer = chat("qwen2.5:0.5b", q)
        print(f"🤖 {answer}")

    print("\n" + "=" * 60)
    print("📊 本地 vs 云端对比")
    print("=" * 60)
    compare_local_vs_cloud()


if __name__ == "__main__":
    main()
