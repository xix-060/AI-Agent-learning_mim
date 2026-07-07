"""utils ⼯具函数的单元测试"""

from src.utils import (
    truncate_text,
    count_tokens_approx,
    safe_json_loads,
    read_text_file,
    chunk_text,
)


def test_truncate_text_short():
    assert truncate_text("hello", 10) == "hello"


def test_truncate_text_long():
    assert truncate_text("hello world", 5) == "he..."


def test_count_tokens_chinese():
    assert count_tokens_approx("你好世界") == 6  # 4 * 1.5


def test_count_tokens_english():
    assert count_tokens_approx("hello") == 1  # 5 / 4 = 1.25 -> 1


def test_safe_json_loads_valid():
    assert safe_json_loads('{"a": 1}') == {"a": 1}


def test_safe_json_loads_invalid():
    assert safe_json_loads("not json", default={}) == {}


def test_read_text_file_not_exist(tmp_path):
    assert read_text_file(tmp_path / "no_exist.txt") is None


def test_read_text_file_exist(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("hello", encoding="utf-8")
    assert read_text_file(f) == "hello"


def test_chunk_text_short():
    assert chunk_text("short", 100) == ["short"]


def test_chunk_text_long():
    text = "a" * 600
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    assert len(chunks) == 2
    assert len(chunks[0]) == 500
