"""MCP Server 概念演示（简化版）"""
# 注意：完整 MCP Server 需要 stdio 通信，这里只演示工具定义

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tools.builtin import create_default_registry

registry = create_default_registry()

# MCP Server 暴露的就是这些工具
print("MCP Server 暴露的工具：")
for schema in registry.get_openai_tools_schema():
    print(f"  - {schema['function']['name']}: {schema['function']['description']}")

print("\n💡 实际 MCP Server 会通过 stdio 与客户端通信")
print("💡 第 5 周会实战完整的 MCP Server")
