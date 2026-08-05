"""MCP Server 实战：暴露工具给 MCP 客户端
使用 FastMCP（MCP 官方推荐的现代 API）
"""

import math
from datetime import datetime
from pathlib import Path
from mcp.server.fastmcp import FastMCP

# ========== 创建 MCP Server ==========
mcp = FastMCP("my-agent-tools")


# ========== 定义工具（使用装饰器 + 类型注解）==========


@mcp.tool()
def calculator(expression: str) -> str:
    """数学计算。支持加减乘除、三角函数、对数等。

    Args:
        expression: 数学表达式，如 '2+3*4', 'sin(pi/2)', 'log(100, 10)'
    """
    safe_expr = expression.replace("^", "**")
    allowed = {
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "sqrt": math.sqrt,
        "log": math.log,
        "pi": math.pi,
        "e": math.e,
        "abs": abs,
    }
    try:
        result = eval(safe_expr, {"__builtins__": {}}, allowed)
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误: {e}"


@mcp.tool()
def get_time() -> str:
    """获取当前日期和时间"""
    now = datetime.now()
    weekdays = "一二三四五六日"
    weekday = weekdays[now.weekday()]
    return f"当前时间：{now.strftime('%Y年%m月%d日 %H:%M:%S')} 星期{weekday}"


@mcp.tool()
def read_file(file_path: str) -> str:
    """读取本地文件内容

    Args:
        file_path: 文件路径（支持绝对路径或相对路径）
    """
    path = Path(file_path)
    if not path.exists():
        return f"错误：文件不存在 {file_path}"
    try:
        content = path.read_text(encoding="utf-8")
        if len(content) > 10000:
            content = content[:10000] + "\n...(内容过长已截断)"
        return content
    except Exception as e:
        return f"读取错误: {e}"


@mcp.tool()
def write_file(file_path: str, content: str) -> str:
    """写入内容到本地文件

    Args:
        file_path: 文件路径（支持绝对路径或相对路径）
        content: 要写入的内容
    """
    try:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"[OK] Written {len(content)} chars to {file_path}"
    except Exception as e:
        return f"[ERROR] Write failed: {e}"


@mcp.tool()
def list_directory(dir_path: str = ".") -> str:
    """列出目录内容

    Args:
        dir_path: 目录路径，默认为当前目录
    """
    path = Path(dir_path)
    if not path.exists() or not path.is_dir():
        return f"Error: Directory not found {dir_path}"
    items = []
    for item in sorted(path.iterdir()):
        type_icon = "[DIR]" if item.is_dir() else "[FILE]"
        size = f" ({item.stat().st_size} bytes)" if item.is_file() else ""
        items.append(f"{type_icon} {item.name}{size}")
    return "\n".join(items)


# ========== 启动 Server ==========

if __name__ == "__main__":
    import sys
    import io

    # 设置 stdout 使用 UTF-8 编码
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    print("=" * 60)
    print("[START] MCP Server Starting...")
    print(
        f"[TOOLS] Available: {[tool.name for tool in mcp._tool_manager.list_tools()]}"
    )
    print("[MODE] stdio (Standard Input/Output)")
    print("=" * 60)

    try:
        # 默认使用 stdio 传输，也可以用 mcp.run(transport="streamable-http")
        mcp.run()
    except KeyboardInterrupt:
        print("\n[STOP] Server stopped")
        sys.exit(0)
