"""LlamaIndex 快速入门"""

import os
import pathlib
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# 配置 LLM（用 OpenAILike，专为 OpenAI 兼容服务设计，不做 model 名枚举校验）
Settings.llm = OpenAILike(
    model=os.getenv("LLM_MODEL", "qwen-turbo"),
    api_key=os.getenv("LLM_API_KEY"),
    api_base=os.getenv("LLM_BASE_URL"),
    is_chat_model=True,
)
# 配置 Embedding（DashScope 的 text-embedding-v3 不在 LlamaIndex 枚举里，用 LangChain 适配器）
Settings.embed_model = LangchainEmbedding(
    OpenAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL", "text-embedding-v3"),
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
        check_embedding_ctx_length=False,
    )
)

# 创建示例文档
pathlib.Path("data/llamaindex_docs").mkdir(parents=True, exist_ok=True)
pathlib.Path("data/llamaindex_docs/intro.txt").write_text(
    "LlamaIndex 是一个数据连接框架，专注于把私有数据接入 LLM。"
    "它的核心是 Index（索引）和 Query Engine（查询引擎）。"
    "与 LangChain 相比，LlamaIndex 更专注于 RAG 场景。",
    encoding="utf-8",
)

# 加载文档
documents = SimpleDirectoryReader("data/llamaindex_docs").load_data()
print(f"加载了 {len(documents)} 个文档")

# 创建索引
index = VectorStoreIndex.from_documents(documents)

# 查询
query_engine = index.as_query_engine()

questions = [
    "LlamaIndex 是什么？",
    "LlamaIndex 和 LangChain 有什么区别？",
]

for q in questions:
    response = query_engine.query(q)
    print(f"\n❓ {q}")
    print(f"🤖 {response}")
