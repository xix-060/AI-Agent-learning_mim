"""unit_converter 测试（项 2：常量抽取 + 比例换算合并）。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tools.builtin import (  # noqa: E402
    _LENGTH_UNITS,
    _WEIGHT_UNITS,
    unit_converter,
)


def test_length_km_to_m():
    """长度：km → m。"""
    out = unit_converter(2, "km", "m")
    assert "2000" in out


def test_weight_kg_to_lb():
    """重量：kg → lb。"""
    out = unit_converter(100, "kg", "lb")
    # 100 kg = 220.462 lb
    assert "220.4" in out


def test_temperature_c_to_f():
    """温度：C → F。"""
    out = unit_converter(100, "C", "F")
    assert "212.0" in out


def test_unsupported_unit_returns_msg():
    """不支持的换算返回提示字符串。"""
    out = unit_converter(1, "kg", "m")
    assert "不支持" in out


def test_constants_are_module_level():
    """常量已提到模块级（非函数内局部）。"""
    assert "m" in _LENGTH_UNITS
    assert "kg" in _WEIGHT_UNITS
    assert _LENGTH_UNITS["km"] == 1000.0
