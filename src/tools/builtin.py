"""Agent 内置工具集"""

import math
import requests
from datetime import datetime
from pathlib import Path
from typing import Callable
from dataclasses import dataclass


@dataclass
class ToolDefinition:
    """工具定义"""

    name: str
    description: str
    parameters: dict  # JSON Schema
    func: Callable


# ========== 工具实现 ==========


def calculator(expression: str) -> str:
    """数学计算"""
    safe_expr = expression.replace("^", "**").replace("×", "*").replace("÷", "/")
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
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
    }
    try:
        result = eval(safe_expr, {"__builtins__": {}}, allowed)
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误: {e}"


def get_current_time(timezone: str = "Asia/Shanghai") -> str:
    """获取当前时间"""
    now = datetime.now()
    weekdays = "一二三四五六日"
    return f"当前时间：{now.strftime('%Y年%m月%d日 %H:%M:%S')} 星期{weekdays[now.weekday()]}"


def read_file(file_path: str) -> str:
    """读取文件内容"""
    path = Path(file_path)
    if not path.exists():
        return f"错误：文件不存在 {file_path}"
    if path.stat().st_size > 10000:
        content = path.read_text(encoding="utf-8")[:10000]
        return content + "\n...(文件过长，已截断)"
    return path.read_text(encoding="utf-8")


def write_file(file_path: str, content: str) -> str:
    """写入文件"""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return f"已写入 {len(content)} 字符到 {file_path}"


def list_directory(dir_path: str = ".") -> str:
    """列出目录内容"""
    path = Path(dir_path)
    if not path.exists():
        return f"错误：目录不存在 {dir_path}"
    if not path.is_dir():
        return f"错误：不是目录 {dir_path}"

    items = []
    for item in sorted(path.iterdir()):
        type_str = "📁" if item.is_dir() else "📄"
        size = item.stat().st_size if item.is_file() else ""
        items.append(f"{type_str} {item.name} {size}")

    return "\n".join(items) if items else "（空目录）"


def web_search(query: str) -> str:
    """网络搜索（免费无 API Key，多源 fallback）"""
    # 离线知识库（最终 fallback）
    offline_knowledge = {
        "python": "Python 是由 Guido van Rossum 于 1991 年创建的解释型高级编程语言，以简洁易读著称。",
        "java": "Java 是由 Sun Microsystems（现 Oracle）于 1995 年推出的面向对象编程语言，广泛用于企业级应用。",
        "javascript": "JavaScript 是由 Brendan Eich 于 1995 年创建的脚本语言，是 Web 开发的核心技术之一。",
        "react": "React 是 Facebook 开发的前端 JavaScript 库，用于构建用户界面，采用组件化和虚拟 DOM 技术。",
        "vue": "Vue.js 是尤雨溪开发的渐进式前端框架，以轻量、灵活、易上手著称。",
        "llm": "LLM (Large Language Model) 是大语言模型，如 GPT、BERT、LLaMA 等，是 AI 自然语言处理的基础。",
        "gpt": "GPT (Generative Pre-trained Transformer) 是 OpenAI 开发的大语言模型系列，包括 GPT-3、GPT-4 等。",
        "transformer": "Transformer 是 Google 于 2017 年在论文《Attention Is All You Need》中提出的深度学习架构。",
        "attention": "Attention（注意力机制）是 Transformer 的核心组件，让模型能够聚焦输入的关键部分。",
        "rag": "RAG (Retrieval-Augmented Generation) 是检索增强生成技术，结合向量检索和 LLM 生成。",
        "agent": "AI Agent 是能够自主规划、使用工具、与环境交互的智能体，是 AI 应用的高级形态。",
        "langchain": "LangChain 是开源的 LLM 应用开发框架，提供 Chains、Agents、Memory 等组件。",
    }

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    # 方案 1: Wikipedia API
    try:
        wiki_query = query.replace(" ", "_")
        resp = requests.get(
            f"https://zh.wikipedia.org/api/rest_v1/page/summary/{wiki_query}",
            headers=headers,
            timeout=5,
        )
        if resp.status_code == 200:
            data = resp.json()
            if data.get("extract"):
                return f"📖 [Wikipedia] {data['extract'][:500]}"
    except Exception:
        pass

    # 方案 2: DuckDuckGo 代理 (免费)
    try:
        resp = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": "1"},
            headers=headers,
            timeout=5,
        )
        data = resp.json()
        results = []
        if data.get("AbstractText"):
            results.append(data["AbstractText"])
        if data.get("Answer"):
            results.append(f"答案: {data['Answer']}")
        for topic in data.get("RelatedTopics", [])[:2]:
            if isinstance(topic, dict) and topic.get("Text"):
                results.append(topic["Text"])
        if results:
            return "🌐 [DuckDuckGo] " + "\n".join(results[:2])
    except Exception:
        pass

    # 方案 3: Hacker News API (技术类问题)
    try:
        resp = requests.get(
            "https://hn.algolia.com/api/v1/search",
            params={"query": query, "tags": "story", "hitsPerPage": 1},
            timeout=5,
        )
        data = resp.json()
        if data.get("hits"):
            hit = data["hits"][0]
            return f"🔶 [Hacker News] {hit.get('title', '')} - {hit.get('url', '')}"
    except Exception:
        pass

    # 离线 fallback
    query_lower = query.lower()
    for key, value in offline_knowledge.items():
        if key in query_lower:
            return f"📚 [离线知识] {value}"

    return f"未找到 '{query}' 的相关信息（所有在线源均不可用）"


def python_executor(code: str) -> str:
    """安全的 Python 代码执行（受限环境）"""
    import io
    import contextlib
    import re

    # 危险关键字检查（更精准的匹配）
    dangerous_patterns = [
        r"import\s+(os|sys|subprocess|shutil)",
        r"from\s+(os|sys|subprocess|shutil)\s+import",
        r"exec\s*\(",
        r"eval\s*\(",
        r"compile\s*\(",
        r"__import__",
        r"open\s*\(",
        r"\.write\s*\(",
        r"__builtins__",
    ]

    for pattern in dangerous_patterns:
        if re.search(pattern, code):
            return f"错误：检测到危险操作 '{pattern}'，禁止执行"

    stdout = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout):
            exec(
                code,
                {
                    "__builtins__": {
                        "print": print,
                        "len": len,
                        "range": range,
                        "str": str,
                        "int": int,
                        "float": float,
                        "list": list,
                        "dict": dict,
                        "tuple": tuple,
                        "sum": sum,
                        "min": min,
                        "max": max,
                        "abs": abs,
                        "round": round,
                        "sorted": sorted,
                        "enumerate": enumerate,
                        "zip": zip,
                        "map": map,
                        "filter": filter,
                    }
                },
            )
        return stdout.getvalue() or "（无输出）"
    except Exception as e:
        return f"执行错误: {e}"


# ========== 工具注册表 ==========


class ToolRegistry:
    """工具注册表：统一管理所有工具"""

    def __init__(self):
        self.tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition):
        """注册工具"""
        self.tools[tool.name] = tool

    def get(self, name: str) -> ToolDefinition | None:
        """获取工具"""
        return self.tools.get(name)

    def execute(self, name: str, **kwargs) -> str:
        """执行工具"""
        tool = self.get(name)
        if not tool:
            return f"错误：未知工具 '{name}'"
        try:
            return tool.func(**kwargs)
        except Exception as e:
            return f"工具执行错误: {e}"

    def list_tools(self) -> list[str]:
        """列出所有工具名"""
        return list(self.tools.keys())

    def get_descriptions(self) -> str:
        """获取所有工具的描述"""
        lines = []
        for name, tool in self.tools.items():
            params = ", ".join(tool.parameters.get("properties", {}).keys())
            lines.append(f"- {name}({params}): {tool.description}")
        return "\n".join(lines)

    def get_openai_tools_schema(self) -> list[dict]:
        """获取 OpenAI Function Calling 格式的工具 Schema"""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in self.tools.values()
        ]


def create_default_registry() -> ToolRegistry:
    """创建默认工具注册表"""
    registry = ToolRegistry()

    registry.register(
        ToolDefinition(
            name="calculator",
            description="数学计算。支持加减乘除、三角函数、对数等。",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式，如 '2+3*4', 'sin(pi/2)'",
                    }
                },
                "required": ["expression"],
            },
            func=lambda expression: calculator(expression),
        )
    )

    registry.register(
        ToolDefinition(
            name="get_current_time",
            description="获取当前日期和时间",
            parameters={
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "时区，默认 Asia/Shanghai",
                    }
                },
            },
            func=lambda timezone="Asia/Shanghai": get_current_time(timezone),
        )
    )

    registry.register(
        ToolDefinition(
            name="read_file",
            description="读取本地文件内容",
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "文件路径"}
                },
                "required": ["file_path"],
            },
            func=lambda file_path: read_file(file_path),
        )
    )

    registry.register(
        ToolDefinition(
            name="write_file",
            description="写入内容到本地文件",
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "文件路径"},
                    "content": {"type": "string", "description": "要写入的内容"},
                },
                "required": ["file_path", "content"],
            },
            func=lambda file_path, content: write_file(file_path, content),
        )
    )

    registry.register(
        ToolDefinition(
            name="list_directory",
            description="列出目录内容",
            parameters={
                "type": "object",
                "properties": {
                    "dir_path": {
                        "type": "string",
                        "description": "目录路径，默认当前目录",
                    }
                },
            },
            func=lambda dir_path=".": list_directory(dir_path),
        )
    )

    registry.register(
        ToolDefinition(
            name="web_search",
            description="网络搜索信息",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词"}
                },
                "required": ["query"],
            },
            func=lambda query: web_search(query),
        )
    )

    return registry


# ========== 演示 ==========
def demo():
    registry = create_default_registry()

    print("🛠 工具注册表")
    print("=" * 60)
    print(registry.get_descriptions())

    print(f"\n📋 已注册工具: {registry.list_tools()}")

    # 测试每个工具
    print("\n🧪 工具测试")
    print("=" * 60)

    print("\n[calculator]")
    print(registry.execute("calculator", expression="2 + 3 * 4"))

    print("\n[get_current_time]")
    print(registry.execute("get_current_time"))

    print("\n[list_directory]")
    print(registry.execute("list_directory", dir_path="."))

    print("\n[write_file]")
    print(
        registry.execute(
            "write_file", file_path="data/test_tool.txt", content="Hello from tool!"
        )
    )

    print("\n[read_file]")
    print(registry.execute("read_file", file_path="data/test_tool.txt"))

    print("\n[web_search]")
    print(registry.execute("web_search", query="Python programming language"))


if __name__ == "__main__":
    demo()
