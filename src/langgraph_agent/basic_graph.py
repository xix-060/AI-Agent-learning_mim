"""LangGraph 基础：状态机 Agent"""

import os
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

load_dotenv()


# ========== 1. 定义工具 ==========


@tool
def calculator(expression: str) -> str:
    """数学计算。输入数学表达式，如 '2+3*4'。"""
    import math

    safe_expr = expression.replace("^", "**")
    allowed = {
        "sin": math.sin,
        "cos": math.cos,
        "sqrt": math.sqrt,
        "pi": math.pi,
        "abs": abs,
    }
    try:
        result = eval(safe_expr, {"__builtins__": {}}, allowed)
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误: {e}"


@tool
def get_time() -> str:
    """获取当前时间。"""
    from datetime import datetime

    return datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")


@tool
def search_info(query: str) -> str:
    """搜索信息。输入搜索关键词。"""
    knowledge = {
        "python": "Python 由 Guido van Rossum 于 1991 年创建。",
        "langgraph": "LangGraph 是 LangChain 团队的 Agent 编排框架。",
        "mcp": "MCP 是 Anthropic 2024 年提出的工具连接协议。",
    }
    for key, val in knowledge.items():
        if key in query.lower():
            return val
    return f"未找到 '{query}' 的信息"


tools = [calculator, get_time, search_info]


# ========== 2. 初始化 LLM ==========

llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL", "deepseek-chat"),
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL"),
    temperature=0.1,
)
llm_with_tools = llm.bind_tools(tools)


# ========== 3. 定义状态 ==========


class AgentState(TypedDict):
    """Agent 状态"""

    messages: Annotated[list[BaseMessage], add_messages]


# ========== 4. 定义节点 ==========


def agent_node(state: AgentState) -> dict:
    """Agent 节点：调用 LLM 决策"""
    system_prompt = SystemMessage(
        content="你是一个智能助手，能使用工具完成任务。请先思考再行动。"
    )
    messages = [system_prompt] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def should_continue(state: AgentState) -> Literal["tools", END]:
    """条件边：决定下一步"""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END


# ========== 5. 构建图 ==========


def build_graph():
    """构建 LangGraph"""
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("agent", agent_node)
    tool_node = ToolNode(tools)
    workflow.add_node("tools", tool_node)

    # 设置入口
    workflow.add_edge(START, "agent")

    # 添加条件边
    workflow.add_conditional_edges("agent", should_continue)
    workflow.add_edge("tools", "agent")  # 工具执行后回到 agent

    # 编译
    return workflow.compile()


# ========== 6. 运行 ==========


def main():
    """运行 LangGraph Agent"""
    app = build_graph()

    # 可视化（输出 Mermaid 图）
    print("📊 Agent 状态图（Mermaid）：")
    print(app.get_graph().draw_mermaid())
    print()

    # 测试
    questions = [
        "现在几点了？",
        "计算 123 * 456 + 789",
        "LangGraph 是什么？",
        "Python 是谁发明的？",
    ]

    for q in questions:
        print(f"\n{'=' * 60}")
        print(f"❓ 问题：{q}")
        print(f"{'=' * 60}")

        result = app.invoke({"messages": [HumanMessage(content=q)]})

        # 打印最后一条 AI 消息
        last_msg = result["messages"][-1]
        print(f"🤖 回答：{last_msg.content}")

        # 打印工具调用过程
        for msg in result["messages"]:
            if (
                isinstance(msg, AIMessage)
                and hasattr(msg, "tool_calls")
                and msg.tool_calls
            ):
                for tc in msg.tool_calls:
                    print(f"   🔧 调用工具：{tc['name']}({tc['args']})")


if __name__ == "__main__":
    main()
