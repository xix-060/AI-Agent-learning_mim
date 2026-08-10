"""LangGraph 高级特性"""

from typing import Annotated, TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL", "qwen-turbo"),
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL"),
    temperature=0.1,
)


# ========== 1. 分支路由 ==========


class BranchState(TypedDict):
    """分支状态"""

    messages: Annotated[list[BaseMessage], add_messages]
    route: str  # 路由标记


def classify_intent(state: BranchState) -> dict:
    """意图分类节点"""
    last_msg = state["messages"][-1]

    classification_llm = llm.with_structured_output(
        {
            "title": "classify",
            "type": "object",
            "properties": {
                "intent": {"type": "string", "enum": ["chat", "code", "math"]}
            },
            "required": ["intent"],
        }
    )

    result = classification_llm.invoke(
        f"分类以下问题的意图（chat/code/math）：{last_msg.content}"
    )
    intent = (
        result.intent if hasattr(result, "intent") else result.get("intent", "chat")
    )

    return {"route": intent}


def route_function(
    state: BranchState,
) -> Literal["chat_agent", "code_agent", "math_agent"]:
    """路由函数"""
    route = state.get("route", "chat")
    if route == "code":
        return "code_agent"
    elif route == "math":
        return "math_agent"
    return "chat_agent"


def chat_agent(state: BranchState) -> dict:
    """闲聊 Agent"""
    response = llm.invoke(
        [
            SystemMessage(content="你是一个友好的聊天助手。"),
            *state["messages"],
        ]
    )
    return {"messages": [response]}


def code_agent(state: BranchState) -> dict:
    """代码 Agent"""
    response = llm.invoke(
        [
            SystemMessage(content="你是一个代码专家。提供简洁的代码和解释。"),
            *state["messages"],
        ]
    )
    return {"messages": [response]}


def math_agent(state: BranchState) -> dict:
    """数学 Agent"""
    response = llm.invoke(
        [
            SystemMessage(content="你是一个数学专家。一步一步解答，展示过程。"),
            *state["messages"],
        ]
    )
    return {"messages": [response]}


def build_branch_graph():
    """构建分支路由图"""
    workflow = StateGraph(BranchState)

    workflow.add_node("classifier", classify_intent)
    workflow.add_node("chat", chat_agent)
    workflow.add_node("code", code_agent)
    workflow.add_node("math", math_agent)

    workflow.add_edge(START, "classifier")
    workflow.add_conditional_edges(
        "classifier",
        route_function,
        {
            "chat_agent": "chat",
            "code_agent": "code",
            "math_agent": "math",
        },
    )
    workflow.add_edge("chat", END)
    workflow.add_edge("code", END)
    workflow.add_edge("math", END)

    return workflow.compile()


# ========== 2. 循环 + 终止条件 ==========


@tool
def search(query: str) -> str:
    """搜索信息"""
    knowledge = {
        "python": "Python 1991 年由 Guido 创建",
        "langchain": "LangChain 是 LLM 框架",
    }
    for k, v in knowledge.items():
        if k in query.lower():
            return v
    return "未找到"


@tool
def calculate(expression: str) -> str:
    """数学计算"""
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"错误: {e}"


class LoopState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    step_count: int


def loop_agent(state: LoopState) -> dict:
    """Agent 节点"""
    system = SystemMessage(content="你是助手，可用工具解决问题。")
    llm_with_tools = llm.bind_tools([search, calculate])
    response = llm_with_tools.invoke([system] + state["messages"])
    return {"messages": [response], "step_count": state.get("step_count", 0) + 1}


def should_continue_loop(state: LoopState) -> Literal["tools", "__end__"]:
    """条件边：先检查步数，再检查工具"""
    # 步数限制
    if state.get("step_count", 0) >= 5:
        return END

    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"
    return END


def build_loop_graph():
    """构建带循环和步数限制的图"""
    workflow = StateGraph(LoopState)

    workflow.add_node("agent", loop_agent)
    workflow.add_node("tools", ToolNode([search, calculate]))

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue_loop)
    workflow.add_edge("tools", "agent")  # 循环回 agent

    return workflow.compile(checkpointer=MemorySaver())


# ========== 3. 子图（Subgraph）==========


class ResearchState(TypedDict):
    """研究流程状态"""

    topic: str
    research: str
    analysis: str
    report: str


def research_node(state: ResearchState) -> dict:
    """研究节点"""
    result = llm.invoke(f"调研'{state['topic']}'，列出3个要点。")
    return {"research": result.content}


def analysis_node(state: ResearchState) -> dict:
    """分析节点"""
    result = llm.invoke(f"分析以下调研，提炼关键洞察：\n{state['research']}")
    return {"analysis": result.content}


def report_node(state: ResearchState) -> dict:
    """报告节点"""
    result = llm.invoke(
        f"基于以下信息写报告：\n调研：{state['research']}\n分析：{state['analysis']}"
    )
    return {"report": result.content}


def build_subgraph():
    """构建研究子图"""
    workflow = StateGraph(ResearchState)
    workflow.add_node("research", research_node)
    workflow.add_node("analysis", analysis_node)
    workflow.add_node("report", report_node)

    workflow.add_edge(START, "research")
    workflow.add_edge("research", "analysis")
    workflow.add_edge("analysis", "report")
    workflow.add_edge("report", END)

    return workflow.compile()


# ========== 演示 ==========
def demo_branch():
    """演示分支路由"""
    print("=" * 60)
    print("1. 分支路由图")
    print("=" * 60)

    app = build_branch_graph()

    # 打印 Mermaid 图
    print("\n📊 状态图：")
    print(app.get_graph().draw_mermaid())

    questions = [
        "你好，今天天气怎么样？",  # chat
        "用 Python 写一个快速排序",  # code
        "计算 15 * 23 + 7",  # math
    ]

    for q in questions:
        result = app.invoke({"messages": [HumanMessage(content=q)], "route": ""})
        print(f"\n❓ {q}")
        print(f"🤖 {result['messages'][-1].content[:100]}")
        print(f"   路由：{result.get('route', 'unknown')}")


def demo_loop():
    """演示循环"""
    print("\n" + "=" * 60)
    print("2. 循环 + 步数限制")
    print("=" * 60)

    app = build_loop_graph()

    result = app.invoke(
        {
            "messages": [HumanMessage(content="Python 是什么时候创建的？")],
            "step_count": 0,
        },
        config={"configurable": {"thread_id": "test-1"}},
    )

    print(f"步数：{result.get('step_count', 0)}")
    print(f"回答：{result['messages'][-1].content[:100]}")


def demo_subgraph():
    """演示子图"""
    print("\n" + "=" * 60)
    print("3. 研究子图")
    print("=" * 60)

    app = build_subgraph()

    result = app.invoke({"topic": "AI Agent 的发展趋势"})
    print(f"\n📋 报告：\n{result['report'][:200]}...")


if __name__ == "__main__":
    demo_branch()
    demo_loop()
    demo_subgraph()
