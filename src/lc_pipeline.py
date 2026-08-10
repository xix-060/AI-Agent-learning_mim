"""LangChain LCEL 实战"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma
from pydantic import BaseModel, Field

load_dotenv()


# ========== LLM 初始化 ==========

llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL", "deepseek-chat"),
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL"),
    temperature=0.7,
)


# ========== 1. 最简链：Prompt | LLM | Parser ==========


def basic_chain():
    """基础链"""
    prompt = ChatPromptTemplate.from_template("用一句话解释：{topic}")

    # LCEL 链
    chain = prompt | llm | StrOutputParser()

    result = chain.invoke({"topic": "什么是 Transformer"})
    print(f"[basic_chain] {result}")
    return chain


# ========== 2. RAG 链（用 LCEL 编排）==========


def rag_chain():
    """用 LCEL 编排 RAG"""
    # 准备知识库
    documents = [
        "LangChain 是 LLM 应用开发框架，提供 LCEL、Agent、Memory 等组件。",
        "LangGraph 是 LangChain 团队的 Agent 编排框架，基于状态机。",
        "LCEL 是 LangChain 的表达式语言，用管道符 | 编排组件。",
        "Chroma 是轻量级向量数据库，适合开发原型。",
        "RAG 通过检索外部知识增强 LLM，减少幻觉。",
    ]

    embeddings = OpenAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL", "text-embedding-v3"),
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
        check_embedding_ctx_length=False,
    )

    vectorstore = Chroma.from_texts(
        texts=documents,
        embedding=embeddings,
        collection_name="lc_demo",
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # RAG Prompt
    prompt = ChatPromptTemplate.from_template("""根据以下资料回答问题：

资料：{context}

问题：{question}

回答：""")

    # LCEL RAG 链
    rag_chain = (
        {
            "context": retriever
            | (lambda docs: "\n\n".join(d.page_content for d in docs)),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    # 测试
    questions = [
        "LangChain 有哪些组件？",
        "LCEL 是什么？",
        "RAG 有什么用？",
    ]

    for q in questions:
        result = rag_chain.invoke(q)
        print(f"\n❓ {q}")
        print(f"🤖 {result}")

    return rag_chain


# ========== 3. 结构化输出链 ==========


class SentimentResult(BaseModel):
    """情感分析结果"""

    sentiment: str = Field(description="情感：正面/负面/中性")
    confidence: float = Field(description="置信度 0-1")
    keywords: list[str] = Field(description="关键词列表")


def structured_chain():
    """结构化输出"""
    parser = JsonOutputParser(pydantic_object=SentimentResult)

    prompt = ChatPromptTemplate.from_template(
        """分析以下文本的情感，返回 JSON。

文本：{text}

{format_instructions}""",
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | llm | parser

    texts = [
        "这家餐厅味道真好，服务也很棒！",
        "产品质量太差了，退货了。",
        "今天天气一般，不好不坏。",
    ]

    for text in texts:
        result = chain.invoke({"text": text})
        print(f"\n📝 {text}")
        print(f"📊 {result}")

    return chain


# ========== 4. 并行链 ==========


def parallel_chain():
    """并行执行多个链"""
    summary_prompt = ChatPromptTemplate.from_template("用一句话总结：{text}")
    translation_prompt = ChatPromptTemplate.from_template("翻译成英文：{text}")
    keywords_prompt = ChatPromptTemplate.from_template(
        "提取3个关键词，用逗号分隔：{text}"
    )

    chain = RunnableParallel(
        summary=summary_prompt | llm | StrOutputParser(),
        translation=translation_prompt | llm | StrOutputParser(),
        keywords=keywords_prompt | llm | StrOutputParser(),
    )

    text = "LangChain 是一个强大的 LLM 应用开发框架，支持链式调用、Agent、记忆等功能。"
    result = chain.invoke({"text": text})

    print(f"\n📝 原文：{text}")
    print(f"📋 总结：{result['summary']}")
    print(f"🌐 翻译：{result['translation']}")
    print(f"🔑 关键词：{result['keywords']}")

    return chain


# ========== 5. 带 Fallback 的链 ==========


def fallback_chain():
    """主链失败时自动切换备选"""
    primary_prompt = ChatPromptTemplate.from_template("用学术论文风格回答：{question}")
    fallback_prompt = ChatPromptTemplate.from_template(
        "用通俗易懂的方式回答：{question}"
    )

    primary_chain = primary_prompt | llm | StrOutputParser()
    fallback_chain = fallback_prompt | llm | StrOutputParser()

    # 主链失败时用 fallback
    chain_with_fallback = primary_chain.with_fallbacks([fallback_chain])

    result = chain_with_fallback.invoke({"question": "什么是量子计算"})
    print(f"\n[fallback_chain] {result}")

    return chain_with_fallback


# ========== 6. 流式输出 ==========


def streaming_chain():
    """流式输出"""
    prompt = ChatPromptTemplate.from_template("写一首关于{topic}的短诗")
    chain = prompt | llm | StrOutputParser()

    print("\n[streaming] 流式输出：")
    for chunk in chain.stream({"topic": "秋天"}):
        print(chunk, end="", flush=True)
    print()


# ========== 演示 ==========
def main():
    print("=" * 60)
    print("1. 基础链")
    print("=" * 60)
    basic_chain()

    print("\n" + "=" * 60)
    print("2. RAG 链")
    print("=" * 60)
    rag_chain()

    print("\n" + "=" * 60)
    print("3. 结构化输出链")
    print("=" * 60)
    structured_chain()

    print("\n" + "=" * 60)
    print("4. 并行链")
    print("=" * 60)
    parallel_chain()

    print("\n" + "=" * 60)
    print("5. Fallback 链")
    print("=" * 60)
    fallback_chain()

    print("\n" + "=" * 60)
    print("6. 流式输出")
    print("=" * 60)
    streaming_chain()


if __name__ == "__main__":
    main()
