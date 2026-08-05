"""LangGraph 进阶：真正的 Human-in-the-Loop

核心要点：
1. 使用 interrupt() 暂停执行
2. 使用 Command(resume=...) 恢复执行
3. 配合 MemorySaver 保存状态
"""

import os
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt

load_dotenv()


# ========== 工具定义 ==========


@tool
def write_file(file_path: str, content: str) -> str:
    """写入文件到指定路径。⚠️ 需要人工确认！"""
    from pathlib import Path

    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return f"已写入 {len(content)} 字符到 {file_path}"


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """发送邮件（模拟）。⚠️ 需要人工确认！"""
    return f"邮件已发送给 {to}，主题：{subject}"


@tool
def delete_file(file_path: str) -> str:
    """删除文件。⚠️ 需要人工确认！"""
    from pathlib import Path

    path = Path(file_path)
    if path.exists():
        path.unlink()
        return f"已删除 {file_path}"
    return f"文件不存在：{file_path}"


# 需要人工审核的危险工具
DANGEROUS_TOOLS = {"write_file", "send_email", "delete_file"}


# 安全工具示例（不需要审核）
@tool
def get_current_time() -> str:
    """获取当前时间。"""
    from datetime import datetime

    return datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")


tools = [write_file, send_email, delete_file, get_current_time]
tool_node = ToolNode(tools)


# ========== LLM ==========

llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL", "deepseek-chat"),
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL"),
    temperature=0.1,
)
llm_with_tools = llm.bind_tools(tools)


# ========== 状态 ==========


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ========== 节点 ==========


def agent_node(state: AgentState) -> dict:
    """Agent 决策节点"""
    system_prompt = SystemMessage(content="你是一个智能助手。可以使用工具完成任务。")
    messages = [system_prompt] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def human_review_node(state: AgentState) -> Command[Literal["agent", "tools"]]:
    """人工审核节点 - 使用 interrupt() 暂停

    这是真正的 HITL：
    1. 调用 interrupt() 暂停执行，显示待审核的工具调用
    2. 人类通过 Command(resume=...) 恢复，传入审核结果
    """
    last_message = state["messages"][-1]
    tool_calls = last_message.tool_calls

    # 显示待审核的工具调用
    print("\n" + "=" * 60)
    print("⚠️ 检测到危险操作，需要人工确认！")
    print("=" * 60)
    for tc in tool_calls:
        print(f"  🔧 工具：{tc['name']}")
        print(f"  📝 参数：{tc['args']}")
    print("=" * 60)

    # 暂停执行，等待人类审核
    # 人类通过 Command(resume={"action": "approve"|"reject", ...}) 恢复
    human_review = interrupt(
        {
            "question": "是否批准这些工具调用？",
            "tool_calls": tool_calls,
            "options": ["approve", "reject"],
        }
    )

    # 处理人类审核结果
    if human_review.get("action") == "approve":
        print("✅ 已批准，继续执行...")
        return Command(goto="tools")
    else:
        print("❌ 已拒绝，返回 Agent...")
        # 添加拒绝消息给 Agent
        return Command(
            goto="agent",
            update={
                "messages": [HumanMessage(content="用户拒绝了这个操作，请重新考虑。")]
            },
        )


# ========== 条件边 ==========


def should_review(state: AgentState) -> str:
    """判断是否需要人工审核"""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        # 检查是否有危险工具
        for tc in last_message.tool_calls:
            if tc["name"] in DANGEROUS_TOOLS:
                return "need_review"
        return "safe"  # 安全工具直接执行
    return "end"  # 无工具调用，结束


# ========== 构建图 ==========


def build_graph():
    """构建带 HITL 的 LangGraph"""
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("agent", agent_node)
    workflow.add_node("human_review", human_review_node)
    workflow.add_node("tools", tool_node)

    # 入口
    workflow.add_edge(START, "agent")

    # 条件分支：agent -> ?
    workflow.add_conditional_edges(
        "agent",
        should_review,
        {
            "need_review": "human_review",  # 危险工具 -> 人工审核
            "safe": "tools",  # 安全工具 -> 直接执行
            "end": END,  # 无工具 -> 结束
        },
    )

    # 工具执行完回到 agent
    workflow.add_edge("tools", "agent")

    # 人类审核后（通过 Command goto 决定走向）

    # 编译（带 MemorySaver 实现暂停/恢复）
    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)


# ========== 运行 ==========


def run_interactive():
    """交互式运行"""
    app = build_graph()
    config = {"configurable": {"thread_id": "session-1"}}

    print("🤖 Human-in-the-Loop Agent（输入 quit 退出）")
    print("💡 危险操作会暂停等待你确认")
    print("   approve = 批准执行")
    print("   reject  = 拒绝操作")
    print("=" * 60)

    while True:
        user_input = input("\n🙋 你：").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break

        # 第一次运行
        result = app.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
        )

        # 检查是否需要人工审核（有 __interrupt__）
        while hasattr(result, "__interrupt__") or (
            isinstance(result, dict) and result.get("__interrupt__")
        ):
            # 让用户审核
            action = input("\n✅ 审核 (approve/reject): ").strip().lower()

            if action in ("approve", "a", "批准"):
                # 批准 - 用 Command(resume=...) 恢复
                result = app.invoke(
                    Command(resume={"action": "approve"}),
                    config=config,
                )
            else:
                # 拒绝
                result = app.invoke(
                    Command(resume={"action": "reject"}),
                    config=config,
                )

        # 打印最终回复
        last_msg = result["messages"][-1]
        if hasattr(last_msg, "content") and last_msg.content:
            print(f"\n🤖 Agent：{last_msg.content}")


def run_demo():
    """演示模式（自动批准）"""
    app = build_graph()
    config = {"configurable": {"thread_id": "demo-1"}}

    # 测试场景
    test_cases = [
        ("现在几点了？", "approve"),  # 安全工具
        ("写一个文件 test.txt，内容是 hello", "approve"),  # 危险工具
        ("删除 test.txt 文件", "reject"),  # 危险工具（拒绝）
    ]

    for question, auto_action in test_cases:
        print(f"\n{'=' * 60}")
        print(f"❓ 问题：{question}")
        print(f"💡 自动审核：{auto_action}")
        print(f"{'=' * 60}")

        # 第一次运行
        result = app.invoke(
            {"messages": [HumanMessage(content=question)]},
            config=config,
        )

        # 处理 interrupt
        interrupt_count = 0
        while hasattr(result, "__interrupt__") or (
            isinstance(result, dict) and result.get("__interrupt__")
        ):
            interrupt_count += 1
            print(f"  ⏸️ 暂停 #{interrupt_count}")

            # 自动执行审核动作
            action = auto_action if interrupt_count == 1 else "approve"
            result = app.invoke(
                Command(resume={"action": action}),
                config=config,
            )
            print(f"  ▶️ 自动 {action}")

        # 打印结果
        last_msg = result["messages"][-1]
        if hasattr(last_msg, "content") and last_msg.content:
            print(f"🤖 回答：{last_msg.content}")

        # 打印工具调用记录
        for msg in result["messages"]:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    print(f"  🔧 {tc['name']}({tc['args']})")


if __name__ == "__main__":
    import sys

    if "--demo" in sys.argv:
        run_demo()
    else:
        run_interactive()
