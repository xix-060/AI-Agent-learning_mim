"""LangGraph Agent 主逻辑

架构流程（对应 README）:
  用户输入 → Agent Node → should_continue?
                             ├─ tools → RAG Retrieve → Agent Node
                             └─ END
"""

from typing import TypedDict, Annotated, Sequence, Literal
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

from knowledge_agent.src.config import LLMConfig
from knowledge_agent.src.memory import ConversationMemory
from knowledge_agent.src.tools import AVAILABLE_TOOLS
from knowledge_agent.src.rag import RAGEngine


SYSTEM_PROMPT = """你是一个智能知识库助手。你可以回答问题、检索知识库、调用工具。

## 重要规则
- 用户问时间/日期时，必须调用 get_time 工具，不要自己猜测
- 用户要计算时，必须调用 calculator 工具
- 用户要查看/读写文件时，必须调用对应文件工具
- 你没有实时信息，所有实时数据必须通过工具获取

## 工作方式
1. 理解用户意图
2. 如果需要计算、时间、文件操作，调用对应工具
3. 如果用户询问知识库相关内容，使用知识库检索结果
4. 综合信息给出准确回答，引用来源

## 回答风格
- 简洁明了，使用中文
- 不确定时诚实告知
- 适当使用列表格式"""


class AgentState(TypedDict):
    """Agent 状态"""

    messages: Annotated[Sequence[BaseMessage], add_messages]


class KnowledgeAgent:
    """知识库 Agent - LangGraph 编排"""

    def __init__(self):
        # LLM
        self.llm = ChatOpenAI(
            model=LLMConfig.MODEL,
            api_key=LLMConfig.API_KEY,
            base_url=LLMConfig.BASE_URL,
            temperature=LLMConfig.TEMPERATURE,
        )
        self.llm_with_tools = self.llm.bind_tools(AVAILABLE_TOOLS)

        # 组件
        self.memory = ConversationMemory()
        self.rag = RAGEngine()
        self.checkpointer = MemorySaver()

        # 工具节点
        self.tool_node = ToolNode(AVAILABLE_TOOLS)

        # 构建图
        self.graph = self._build_graph()

    def _build_graph(self):
        """构建 LangGraph 状态图

        流程:
          START → agent → should_continue?
                           ├─ "tools" → tools → agent
                           └─ "end"   → END
        """
        workflow = StateGraph(AgentState)

        # 节点
        workflow.add_node("agent", self._agent_node)
        workflow.add_node("tools", self._tools_node)

        # 入口
        workflow.set_entry_point("agent")

        # 条件边：判断是否需要调用工具
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {"tools": "tools", "end": END},
        )

        # tools 执行后回到 agent
        workflow.add_edge("tools", "agent")

        return workflow.compile(checkpointer=self.checkpointer)

    def _agent_node(self, state: AgentState) -> dict:
        """Agent 节点 - LLM 决策"""
        messages = list(state["messages"])

        # 注入系统提示
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

        # 注入 RAG 上下文（如果用户问的是知识性问题）
        rag_context = self._maybe_retrieve_rag(messages)
        if rag_context:
            messages.insert(1, HumanMessage(content=rag_context))

        response = self.llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def _tools_node(self, state: AgentState) -> dict:
        """工具执行节点"""
        result = self.tool_node.invoke(state)
        return {"messages": result["messages"]}

    def _should_continue(self, state: AgentState) -> Literal["tools", "end"]:
        """判断是否继续调用工具"""
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return "end"

    def _maybe_retrieve_rag(self, messages: list) -> str:
        """判断是否需要 RAG 检索，并返回上下文"""
        # 获取最后一条用户消息
        user_query = ""
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                user_query = msg.content
                break

        if not user_query:
            return ""

        # 知识库为空则跳过
        stats = self.rag.get_stats()
        if stats.get("chunk_count", 0) == 0:
            return ""

        # 检索
        results = self.rag.search(user_query, use_reranker=True)
        if not results:
            return ""

        # 构建上下文
        parts = ["[知识库检索结果]"]
        for i, r in enumerate(results, 1):
            parts.append(f"\n[{i}] 来源: {r['source']} (相关度: {r['score']:.2f})")
            parts.append(r["content"][:500])

        return "\n".join(parts)

    def chat(self, user_input: str, thread_id: str = "default") -> str:
        """对话"""
        self.memory.add_user_message(user_input)

        config = {"configurable": {"thread_id": thread_id}}
        initial_state = {"messages": [HumanMessage(content=user_input)]}

        result = self.graph.invoke(initial_state, config)

        # 提取最后一条 AI 回复
        ai_response = ""
        for msg in reversed(result["messages"]):
            if isinstance(msg, type(msg)) and msg.__class__.__name__ == "AIMessage":
                if not getattr(msg, "tool_calls", None):
                    ai_response = msg.content
                    break

        self.memory.add_ai_message(ai_response)
        return ai_response

    def import_document(self, source: str) -> dict:
        """导入文档到知识库"""
        return self.rag.import_document(source)

    def get_stats(self) -> dict:
        """获取知识库统计"""
        return self.rag.get_stats()

    def clear_memory(self) -> None:
        """清空对话记忆"""
        self.memory.clear_short_term()
