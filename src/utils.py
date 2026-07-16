"""通用工具函数集合 - 第 1 周练习(python typing)"""

from typing import Any, Optional
from pathlib import Path
import json
import re


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """截断文本到指定⻓度，超出部分用 suffix 替代"""
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def count_tokens_approx(text: str) -> int:
    """粗略估算 token 数（中文 1 字 ≈ 1.5 token，英文 4 字符 ≈ 1 token）"""
    chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    other_chars = len(text) - chinese_chars
    return int(chinese_chars * 1.5 + other_chars / 4)


def safe_json_loads(text: str, default: Any = None) -> Any:
    """安全的 JSON 解析，失败返回默认值"""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return default if default is not None else {}


def read_text_file(path: str | Path, encoding: str = "utf-8") -> Optional[str]:
    """读取文本文件，不存在返回 None"""
    p = Path(path)
    if not p.exists():
        return None
    return p.read_text(encoding=encoding)


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """将文本切分成带重叠的块（RAG 预演）"""
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks
