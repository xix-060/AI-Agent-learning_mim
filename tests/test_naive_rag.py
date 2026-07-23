"""Naive RAG 测试"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from src.embedder import Embedder
from src.llm_client import LLMClient
from src.naive_rag.rag import NaiveRAG
from src.naive_rag.loader import TextChunker


@pytest.fixture(scope="module")
def rag():
    embedder = Embedder()
    llm = LLMClient()
    rag = NaiveRAG(embedder, llm, top_k=2)

    docs = [
        "Python 由 Guido van Rossum 创建于 1991 年。",
        "JavaScript 由 Brendan Eich 创建于 1995 年。",
        "Java 由 James Gosling 创建于 1995 年。",
    ]
    rag.add_documents(docs)
    return rag


def test_retrieve(rag):
    results = rag.retrieve("Python 是谁发明的？")
    assert len(results) == 2
    assert "Guido" in results[0][0].content
    assert results[0][1] > results[1][1]


def test_query(rag):
    result = rag.query("JavaScript 谁发明的？")
    assert "Brendan" in result["answer"]


def test_chunker_fixed():
    text = "a" * 1200
    chunks = TextChunker.fixed_size(text, chunk_size=500, overlap=50)
    assert len(chunks) == 3
    assert len(chunks[0]) == 500


def test_chunker_sentence():
    text = "第一句话。第二句话！第三句话？第四句话。"
    chunks = TextChunker.by_sentence(text, sentences_per_chunk=2)
    assert len(chunks) == 2
