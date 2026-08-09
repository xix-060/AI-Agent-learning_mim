"""RAG 检索增强生成模块 - 多源文档导入 + 向量检索 + Reranker"""

import os
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

from knowledge_agent.src.config import (
    EmbeddingConfig,
    RerankerConfig,
    VectorDBConfig,
    RAGConfig,
    WebConfig,
    UPLOAD_DIR,
)


class DocumentLoader:
    """多源文档加载器 - 支持 PDF / TXT / Markdown / 网页"""

    @staticmethod
    def load_pdf(file_path: str) -> List[Document]:
        """加载 PDF 文档"""
        from pypdf import PdfReader

        reader = PdfReader(file_path)
        documents = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text.strip():
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"source": Path(file_path).name, "page": i + 1},
                    )
                )
        return documents

    @staticmethod
    def load_text(file_path: str) -> List[Document]:
        """加载 TXT 文档"""
        content = Path(file_path).read_text(encoding="utf-8", errors="ignore")
        return [
            Document(page_content=content, metadata={"source": Path(file_path).name})
        ]

    @staticmethod
    def load_markdown(file_path: str) -> List[Document]:
        """加载 Markdown 文档"""
        content = Path(file_path).read_text(encoding="utf-8", errors="ignore")
        return [
            Document(page_content=content, metadata={"source": Path(file_path).name})
        ]

    @staticmethod
    def load_csv(file_path: str) -> List[Document]:
        """加载 CSV 文档"""
        import pandas as pd

        df = pd.read_csv(file_path)
        content = df.to_string()
        return [
            Document(page_content=content, metadata={"source": Path(file_path).name})
        ]

    @staticmethod
    def load_webpage(url: str) -> List[Document]:
        """抓取网页内容"""
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError("需要安装 beautifulsoup4: pip install beautifulsoup4")

        resp = requests.get(
            url,
            timeout=WebConfig.REQUEST_TIMEOUT,
            headers={"User-Agent": WebConfig.USER_AGENT},
        )
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        # 移除脚本和样式
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)
        if len(text) > WebConfig.MAX_CONTENT_LENGTH:
            text = text[: WebConfig.MAX_CONTENT_LENGTH]

        title = soup.title.string.strip() if soup.title else url
        return [Document(page_content=text, metadata={"source": url, "title": title})]

    @classmethod
    def load(cls, source: str) -> List[Document]:
        """根据来源自动选择加载方式

        Args:
            source: 文件路径或网页 URL
        """
        # 网页 URL
        if source.startswith(("http://", "https://")):
            return cls.load_webpage(source)

        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {source}")

        suffix = path.suffix.lower()
        loaders = {
            ".pdf": cls.load_pdf,
            ".txt": cls.load_text,
            ".md": cls.load_markdown,
            ".csv": cls.load_csv,
        }

        loader = loaders.get(suffix)
        if loader is None:
            # 未知类型按文本处理
            return cls.load_text(source)

        return loader(str(path))

    @classmethod
    def load_directory(cls, dir_path: str) -> List[Document]:
        """从目录批量加载所有支持的文档"""
        dir_path = Path(dir_path)
        supported = {".pdf", ".txt", ".md", ".csv"}
        documents = []

        for file_path in sorted(dir_path.iterdir()):
            if file_path.is_file() and file_path.suffix.lower() in supported:
                try:
                    docs = cls.load(str(file_path))
                    documents.extend(docs)
                except Exception as e:
                    print(f"  [跳过] {file_path.name}: {e}")

        return documents


class Reranker:
    """重排序器 - 调用阿里云 gte-rerank 模型"""

    def __init__(self):
        self.api_key = RerankerConfig.API_KEY
        self.base_url = RerankerConfig.BASE_URL
        self.model = RerankerConfig.MODEL

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = RAGConfig.RERANK_TOP_K,
    ) -> List[Dict[str, Any]]:
        """对检索结果进行重排序

        Args:
            query: 用户查询
            documents: 检索结果列表
            top_k: 重排序后保留的数量
        """
        if not documents or not self.api_key:
            return documents[:top_k]

        try:
            texts = [doc["content"] for doc in documents]
            resp = requests.post(
                f"{self.base_url}/services/rerank/text-rerank/text-rerank",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "input": {"query": query, "documents": texts},
                    "parameters": {"top_n": top_k, "return_documents": False},
                },
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()

            # 解析重排序结果
            results = []
            for item in data.get("output", {}).get("results", []):
                idx = item["index"]
                score = item["relevance_score"]
                doc = documents[idx].copy()
                doc["score"] = score
                doc["reranked"] = True
                results.append(doc)

            return results if results else documents[:top_k]

        except Exception:
            # 静默降级，只在调试时输出
            return documents[:top_k]


class RAGEngine:
    """RAG 引擎 - 文档加载 + 向量检索 + 重排序"""

    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model=EmbeddingConfig.MODEL,
            api_key=EmbeddingConfig.API_KEY,
            base_url=EmbeddingConfig.BASE_URL,
            check_embedding_ctx_length=False,
        )
        self.vectorstore: Optional[Chroma] = None
        self.reranker = Reranker()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=RAGConfig.CHUNK_SIZE,
            chunk_overlap=RAGConfig.CHUNK_OVERLAP,
        )
        self._init_vectorstore()

    def _init_vectorstore(self) -> None:
        """初始化或加载已有向量存储"""
        if os.path.exists(VectorDBConfig.PERSIST_DIR):
            try:
                self.vectorstore = Chroma(
                    persist_directory=VectorDBConfig.PERSIST_DIR,
                    embedding_function=self.embeddings,
                    collection_name=VectorDBConfig.COLLECTION_NAME,
                )
            except Exception as e:
                print(f"  [RAG] 加载已有向量存储失败: {e}")

    def import_document(self, source: str) -> Dict[str, Any]:
        """导入单个文档（文件路径或网页 URL）

        Returns:
            导入结果统计
        """
        # 加载文档
        documents = DocumentLoader.load(source)
        if not documents:
            return {"success": False, "message": "文档内容为空"}

        # 分割文档
        splits = self.text_splitter.split_documents(documents)

        # 写入向量库
        if self.vectorstore is None:
            self.vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=self.embeddings,
                persist_directory=VectorDBConfig.PERSIST_DIR,
                collection_name=VectorDBConfig.COLLECTION_NAME,
            )
        else:
            self.vectorstore.add_documents(splits)

        source_name = documents[0].metadata.get("source", source)
        return {
            "success": True,
            "source": source_name,
            "documents": len(documents),
            "chunks": len(splits),
        }

    def import_uploaded_files(self) -> List[Dict[str, Any]]:
        """导入 uploads 目录中所有文件"""
        results = []
        documents = DocumentLoader.load_directory(str(UPLOAD_DIR))
        if not documents:
            return results

        splits = self.text_splitter.split_documents(documents)

        if self.vectorstore is None:
            self.vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=self.embeddings,
                persist_directory=VectorDBConfig.PERSIST_DIR,
                collection_name=VectorDBConfig.COLLECTION_NAME,
            )
        else:
            self.vectorstore.add_documents(splits)

        # 按来源统计
        source_set = {doc.metadata.get("source") for doc in documents}
        for src in source_set:
            results.append({"success": True, "source": src})

        return results

    def search(
        self,
        query: str,
        top_k: int = RAGConfig.TOP_K,
        use_reranker: bool = True,
    ) -> List[Dict[str, Any]]:
        """检索：向量检索 → Reranker 重排序

        Args:
            query: 查询文本
            top_k: 向量检索返回数量
            use_reranker: 是否使用重排序
        """
        if self.vectorstore is None:
            return []

        try:
            results = self.vectorstore.similarity_search_with_score(query, k=top_k)
        except Exception as e:
            print(f"  [RAG] 向量检索出错: {e}")
            return []

        formatted = []
        for doc, score in results:
            similarity = 1.0 / (1.0 + score) if score > 0 else 1.0
            formatted.append(
                {
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "unknown"),
                    "score": similarity,
                }
            )

        # Reranker 重排序
        if use_reranker and len(formatted) > 1:
            formatted = self.reranker.rerank(
                query, formatted, top_k=RAGConfig.RERANK_TOP_K
            )

        return formatted

    def get_stats(self) -> Dict[str, Any]:
        """获取知识库统计信息"""
        if self.vectorstore is None:
            return {"status": "not_initialized", "chunk_count": 0}

        try:
            count = self.vectorstore._collection.count()
            return {
                "status": "ready",
                "chunk_count": count,
                "collection": VectorDBConfig.COLLECTION_NAME,
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def clear(self) -> None:
        """清空向量库"""
        if self.vectorstore is not None:
            self.vectorstore.delete_collection()
            self.vectorstore = None
