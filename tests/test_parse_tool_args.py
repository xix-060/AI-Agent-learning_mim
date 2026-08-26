"""_parse_tool_args 容错测试（项 3：JSON 解析容错 + 边界防御）。"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.function_calling import _parse_tool_args  # noqa: E402


def test_none_returns_empty():
    """None → {}。"""
    assert _parse_tool_args(None) == {}


def test_empty_string_returns_empty():
    """空串 → {}。"""
    assert _parse_tool_args("") == {}
    assert _parse_tool_args("   ") == {}


def test_single_quote_and_trailing_comma_fixed():
    """单引号 + 尾逗号 → 正则修复后解析成功。"""
    out = _parse_tool_args("{'a': 1,}")
    assert out == {"a": 1}


def test_invalid_json_returns_empty_and_warns(caplog):
    """非法 JSON → {} 且记 warning 日志。"""
    with caplog.at_level(logging.WARNING, logger="src.function_calling"):
        out = _parse_tool_args("not json at all")
    assert out == {}
    assert "无法解析" in caplog.text


def test_non_dict_returns_empty():
    """非 dict（如 list）→ {}。"""
    assert _parse_tool_args("[1, 2, 3]") == {}
    assert _parse_tool_args('"just a string"') == {}


def test_valid_dict_passes_through():
    """合法 dict 原样返回。"""
    out = _parse_tool_args('{"value": 100, "from_unit": "kg", "to_unit": "lb"}')
    assert out == {"value": 100, "from_unit": "kg", "to_unit": "lb"}
