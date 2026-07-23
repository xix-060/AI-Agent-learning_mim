"""文档加载器：支持 PDF、TXT、Markdown"""

import re
from pathlib import Path

import pdfplumber


class DocumentLoader:
    """文档加载器"""

    @staticmethod
    def load_pdf(file_path: str) -> list[str]:
        """加载 PDF，返回每页文本列表"""
        pages = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
        return pages

    @staticmethod
    def load_txt(file_path: str) -> str:
        """加载纯文本"""
        return Path(file_path).read_text(encoding="utf-8")

    @staticmethod
    def load_markdown(file_path: str) -> str:
        """加载 Markdown（简单去除标记）"""
        text = Path(file_path).read_text(encoding="utf-8")
        text = re.sub(r"#+\s*", "", text)
        text = re.sub(r"\*+([^*]+)\*+", r"\1", text)
        text = re.sub(r"`([^`]+)`", r"\1", text)
        return text

    @classmethod
    def load(cls, file_path: str) -> list[str]:
        """根据扩展名自动选择加载器"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        ext = path.suffix.lower()

        if ext == ".pdf":
            return cls.load_pdf(file_path)
        elif ext == ".txt":
            return [cls.load_txt(file_path)]
        elif ext == ".md":
            return [cls.load_markdown(file_path)]
        else:
            raise ValueError(f"不支持的文件类型: {ext}")


class TextChunker:
    """文本切块器"""

    @staticmethod
    def fixed_size(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
        """固定大小切块（带重叠）"""
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            if end >= len(text):
                break
            start = end - overlap

        return chunks

    @staticmethod
    def by_sentence(text: str, sentences_per_chunk: int = 5) -> list[str]:
        """按句子切块"""
        sentences = re.split(r"(?<=[。！？.!?])\s*", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks = []
        for i in range(0, len(sentences), sentences_per_chunk):
            chunk = "".join(sentences[i : i + sentences_per_chunk])
            chunks.append(chunk)

        return chunks

    @staticmethod
    def by_paragraph(text: str) -> list[str]:
        """按段落切块"""
        paragraphs = text.split("\n\n")
        return [p.strip() for p in paragraphs if p.strip()]
