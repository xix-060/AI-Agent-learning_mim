# MCP（Model Context Protocol）协议

## 1. 什么是 MCP？

MCP 是 Anthropic 在 2024 年 11 月提出的开放协议，
用于标准化 LLM 应用与外部工具/资源的连接方式。

类比：MCP 之于 AI 工具 = USB 之于电脑外设
之前每个工具都要单独写适配，MCP 提供统一接口。

## 2. 为什么需要 MCP？

- 之前：每个 LLM 框架（LangChain/AutoGen/Cursor）有自己的工具格式
- 之后：工具只写一次 MCP Server，所有支持 MCP 的客户端都能用

## 3. MCP 架构

```
MCP Host（如 Claude Desktop）
    ↓ MCP Protocol
MCP Client
    ↓
MCP Server（你写的工具）
    ↓
本地资源（文件/数据库/API）
```

<br />

## 4. MCP 三大能力

- Tools：可执行的函数（如发邮件、查数据库）
- Resources：可读取的数据（如文件内容、配置）
- Prompts：预设的提示模板

## 5. 和 Function Calling 的关系

- Function Calling：LLM 输出 JSON 调用意图
- MCP：标准化工具的"服务端"定义和通信
- 关系：MCP Server 暴露工具 → 客户端获取工具列表 → LLM Function Calling 选择 → 客户端通过 MCP 执行

## 6. 简单 MCP Server 示例（伪代码）

```Python
from mcp import Server, Tool

server = Server("my-tools")

@server.tool()
def get_weather(city: str) -> str:
    """获取天气"""
    return f"{city} 今天晴，25°C"

server.run()  # 启动 MCP 服务
```

## MCP Server 实战总结

### 架构

```
MCP Host (Claude/Cursor)
    ↓ stdio/SSE
MCP Client (内置)
    ↓ MCP Protocol
MCP Server (你的 mcp_server.py)
    ↓
工具执行
```

### 关键点

1. Server 通过 stdio 通信（标准输入输出）
2. 两个核心回调：list\_tools() 和 call\_tool()
3. 工具用 JSON Schema 描述参数
4. 返回 TextContent（也可以返回图片等）

### 部署方式

1. 本地 stdio（今天实现）
2. 远程 SSE（生产用）
3. Docker 容器化
