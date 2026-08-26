"""calculator 安全求值测试（项 1：eval → ast 安全解析）。

验证：
1. 正常表达式能算对；
2. 危险结构（属性访问 / import / 名称逃逸）被拦，不抛到调用方。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tools.builtin import calculator, _safe_eval  # noqa: E402


# ========== 正常表达式 ==========
def test_safe_eval_basic_arith():
    """基础四则运算。"""
    assert _safe_eval("2 + 3 * 4") == "14"


def test_safe_eval_power():
    """幂运算。"""
    assert _safe_eval("2 ** 10") == "1024"


def test_safe_eval_func_with_const():
    """白名单函数 + 白名单常量。"""
    assert _safe_eval("sin(pi/2)") == "1.0"


def test_calculator_caret_to_pow():
    """calculator 把 ^ 翻译成 **。"""
    assert "1024" in calculator("2 ^ 10")


def test_calculator_returns_expression_prefix():
    """返回值带原表达式前缀。"""
    out = calculator("1 + 1")
    assert out.startswith("1 + 1 =")


# ========== 危险结构必须被拦 ==========
def test_safe_eval_attribute_access_blocked():
    """属性访问（沙箱逃逸经典手法）必须被拦。"""
    for evil in [
        "().__class__.__bases__",
        "''.__class__",
        "__import__('os')",
        "getattr(__builtins__, 'eval')",
    ]:
        # _safe_eval 自身会抛 ValueError；calculator 包装为返回错误字符串
        try:
            _safe_eval(evil)
            raised = False
        except (ValueError, SyntaxError, NameError):
            raised = True
        assert raised, f"危险表达式未被拦: {evil}"


def test_calculator_attribute_access_returns_error_msg():
    """calculator 对危险表达式返回错误字符串，不抛异常。"""
    out = calculator("().__class__.__bases__")
    assert out.startswith("计算错误")


def test_safe_eval_unknown_name_blocked():
    """不在白名单的名称（如 open）必须被拦。"""
    try:
        _safe_eval("open('x')")
        raised = False
    except (ValueError, NameError):
        raised = True
    assert raised


def test_safe_eval_no_builtins():
    """__builtins__ 不暴露：尝试访问 list 等内置应被拦（list 不在白名单）。"""
    # list 不在 _SAFE_FUNCS，但 [1,2] 字面量走 ast.List 节点
    # 这里测的是把 list 当函数调用应被拦
    try:
        _safe_eval("list('abc')")
        raised = False
    except (ValueError, NameError):
        raised = True
    assert raised


def test_calculator_syntax_error_returns_msg():
    """语法错误返回错误字符串，不抛异常。"""
    out = calculator("2 + +")
    assert out.startswith("计算错误")
