"""配置管理模块"""

import os
from pathlib import Path
from dotenv import load_dotenv

# 加载 .env（优先项目根目录的 .env，其次 knowledge_agent/.env）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env")
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# 项目根目录（knowledge_agent/）
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 数据目录
DATA_DIR = PROJECT_ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
CHROMA_DB_DIR = DATA_DIR / "chroma_db"

# 确保目录存在
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)


class LLMConfig:
    """LLM 配置"""

    API_KEY = os.getenv("LLM_API_KEY")
    BASE_URL = os.getenv(
        "LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    MODEL = os.getenv("LLM_MODEL", "qwen-turbo")
    TEMPERATURE = 0.7
    MAX_TOKENS = 2048


class EmbeddingConfig:
    """向量嵌入模型配置"""

    API_KEY = os.getenv("EMBEDDING_API_KEY", LLMConfig.API_KEY)
    BASE_URL = os.getenv("EMBEDDING_BASE_URL", LLMConfig.BASE_URL)
    MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v3")


class RerankerConfig:
    """Reranker 重排序配置"""

    API_KEY = os.getenv("RERANKER_API_KEY", LLMConfig.API_KEY)
    BASE_URL = os.getenv("RERANKER_BASE_URL", "https://dashscope.aliyuncs.com/api/v1")
    MODEL = os.getenv("RERANKER_MODEL", "gte-rerank")


class VectorDBConfig:
    """向量数据库配置"""

    COLLECTION_NAME = "knowledge_base"
    PERSIST_DIR = str(CHROMA_DB_DIR)


class RAGConfig:
    """RAG 配置"""

    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    TOP_K = 4  # 向量检索返回数量
    RERANK_TOP_K = 3  # 重排序后保留数量
    SIMILARITY_THRESHOLD = 0.5


class WebConfig:
    """网页抓取配置"""

    REQUEST_TIMEOUT = 15
    USER_AGENT = "Mozilla/5.0 (KnowledgeAgent/1.0)"
    MAX_CONTENT_LENGTH = 50000  # 最大抓取字符数
