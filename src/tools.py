"""可被 LLM 调⽤的⼯具函数集合"""

import math
from datetime import datetime


# ========== ⼯具函数实现 ==========
def calculate(expression: str) -> str:
    """计算数学表达式 支持加减乘除、幂、三角函数等"""
    # 安全地替换⼀些函数名
    safe_expr = expression.replace("^", "**")
    try:
        # 注意：⽣产环境应该⽤ ast.literal_eval 或 sympy，这⾥简化
        allowed = {
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "sqrt": math.sqrt,
            "log": math.log,
            "log10": math.log10,
            "pi": math.pi,
            "e": math.e,
            "abs": abs,
        }
        result = eval(safe_expr, {"__builtins__": {}}, allowed)
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误: {e}"


def get_current_time() -> str:
    """获取当前时间"""
    now = datetime.now()
    weekdays = ["一", "二", "三", "四", "五", "六", "日"]
    weekday_str = weekdays[now.weekday()]
    return now.strftime(f"%Y年%m月%d日 %H:%M:%S 星期{weekday_str}")


def unit_converter(value: float, from_unit: str, to_unit: str) -> str:
    """单位换算（⻓度/重量/温度）"""
    # ⻓度换算（统⼀到⽶）
    length = {
        "m": 1,
        "km": 1000,
        "cm": 0.01,
        "mm": 0.001,
        "mile": 1609.34,
        "ft": 0.3048,
    }
    # 重量换算（统⼀到克）
    weight = {"g": 1, "kg": 1000, "lb": 453.592, "oz": 28.3495}

    if from_unit in length and to_unit in length:
        result = value * length[from_unit] / length[to_unit]
        return f"{value} {from_unit} = {result:.4f} {to_unit}"
    elif from_unit in weight and to_unit in weight:
        result = value * weight[from_unit] / weight[to_unit]
        return f"{value} {from_unit} = {result:.4f} {to_unit}"
    elif from_unit == "C" and to_unit == "F":
        result = value * 9 / 5 + 32
        return f"{value}°C = {result:.1f}°F"
    elif from_unit == "F" and to_unit == "C":
        result = (value - 32) * 5 / 9
        return f"{value}°F = {result:.1f}°C"
    else:
        return f"不支持的换算: {from_unit} → {to_unit}"


# ========== ⼯具 Schema（告诉 LLM 有哪些⼯具可⽤）==========
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "计算数学表达式。支持加减乘除、幂(^)、sin/cos/tan/sqrt/log等。",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式，如 '2+3*4', 'sin(pi/2)', 'sqrt(16)'",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前日期和时间",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "unit_converter",
            "description": "单位换算。支持长度(m,km,cm,mm,mile,ft)、重量(g,kg,lb,oz)、温度 (C,F)",
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {"type": "number", "description": "要换算的数值"},
                    "from_unit": {"type": "string", "description": "原始单位"},
                    "to_unit": {"type": "string", "description": "⽬标单位"},
                },
                "required": ["value", "from_unit", "to_unit"],
            },
        },
    },
]

# ⼯具名 → 函数的映射表
TOOL_MAP = {
    "calculate": calculate,
    "get_current_time": get_current_time,
    "unit_converter": unit_converter,
}
