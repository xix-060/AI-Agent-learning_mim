"""工具集模块 - 计算器 / 时间 / 文件操作"""

from datetime import datetime
from langchain_core.tools import tool
from knowledge_agent.src.config import UPLOAD_DIR


@tool
def calculator(expression: str) -> str:
    """数学计算。支持加减乘除、三角函数、对数等。

    Args:
        expression: 数学表达式，如 '2 + 3 * 4', 'sin(3.14)', 'log(100, 10)'
    """
    import math

    safe_globals = {
        "__builtins__": {},
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "pow": pow,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "log10": math.log10,
        "log2": math.log2,
        "sqrt": math.sqrt,
        "pi": math.pi,
        "e": math.e,
        "ceil": math.ceil,
        "floor": math.floor,
    }

    try:
        result = eval(expression, safe_globals, {})
        return f"计算结果: {expression} = {result}"
    except Exception as e:
        return f"计算错误: {e}"


@tool
def get_time() -> str:
    """获取当前日期和时间。"""
    now = datetime.now()
    return now.strftime("%Y年%m月%d日 %H:%M:%S")


@tool
def list_files() -> str:
    """列出 data/uploads 目录中的所有文件。"""
    if not UPLOAD_DIR.exists():
        return "上传目录不存在"

    files = [f for f in UPLOAD_DIR.iterdir() if f.is_file() and f.name != ".gitkeep"]
    if not files:
        return "暂无上传文件"

    lines = []
    for f in sorted(files):
        size = f.stat().st_size
        unit = "bytes"
        if size > 1024:
            size = size / 1024
            unit = "KB"
        if size > 1024:
            size = size / 1024
            unit = "MB"
        lines.append(f"  - {f.name} ({size:.1f} {unit})")

    return "上传文件列表:\n" + "\n".join(lines)


@tool
def read_file(filename: str, max_chars: int = 5000) -> str:
    """读取 data/uploads 目录中指定文件的内容。

    Args:
        filename: 文件名
        max_chars: 最大读取字符数，默认 5000
    """
    file_path = UPLOAD_DIR / filename

    if not file_path.exists():
        return f"错误: 文件 {filename} 不存在"

    try:
        content = file_path.read_text(encoding="utf-8", errors="ignore")
        if len(content) > max_chars:
            content = content[:max_chars] + "\n... (内容已截断)"
        return content
    except Exception as e:
        return f"读取文件出错: {e}"


@tool
def write_file(filename: str, content: str) -> str:
    """写入内容到 data/uploads 目录中的文件。

    Args:
        filename: 文件名
        content: 要写入的内容
    """
    file_path = UPLOAD_DIR / filename
    try:
        file_path.write_text(content, encoding="utf-8")
        return f"已写入 {len(content)} 字符到 {filename}"
    except Exception as e:
        return f"写入文件出错: {e}"


# 注册所有工具
AVAILABLE_TOOLS = [
    calculator,
    get_time,
    list_files,
    read_file,
    write_file,
]
