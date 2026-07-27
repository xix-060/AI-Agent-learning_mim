"""从零实现 ReAct Agent（不用任何框架）"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
from dataclasses import dataclass
from typing import Callable, Optional
from src.llm_client import LLMClient


@dataclass
class Tool:
    """工具定义"""

    name: str
    description: str
    func: Callable[[str], str]


@dataclass
class AgentStep:
    """Agent 的一步"""

    thought: str
    action: str
    action_input: str
    observation: str = ""


class ReActAgent:
    """ReAct Agent：Thought-Action-Observation 循环"""

    # ReAct Prompt 模板（带 Few-shot 示例）
    PROMPT_TEMPLATE = """尽可能回答以下问题。

【重要规则】
1. 你必须使用工具来获取准确信息，不要依赖你的内部知识
2. 你每次只能输出一步：Thought + Action + Action Input
3. DO NOT generate Observation yourself! Observation 由系统在工具执行后提供
4. 当你认为已经有足够信息时，输出：Thought + Final Answer

可用工具：
{tools_description}

输出格式（严格遵守）：

当需要使用工具时：
Thought: 你的思考过程
Action: 工具名称（必须是 [{tool_names}] 之一）
Action Input: 工具参数

当有足够信息时：
Thought: 我已经知道最终答案了
Final Answer: 你的最终回答

示例1 - 询问时间：
Question: 现在几点了？
Thought: 用户询问当前时间，我需要使用 Time 工具获取
Action: Time
Action Input:



示例2 - 搜索信息：
Question: Python 是谁发明的？
Thought: 用户询问 Python 的发明者，我需要使用 Search 工具
Action: Search
Action Input: Python 发明者



示例3 - 数学计算：
Question: 计算 15 + 27
Thought: 这是一个数学计算问题，我需要使用 Calculator 工具
Action: Calculator
Action Input: 15 + 27



现在开始回答以下问题：

Question: {question}
{agent_scratchpad}"""

    def __init__(
        self,
        llm: LLMClient,
        tools: list[Tool],
        max_steps: int = 5,
        verbose: bool = True,
    ):
        self.llm = llm
        self.tools = {t.name: t for t in tools}
        self.max_steps = max_steps
        self.verbose = verbose
        self.history: list[AgentStep] = []

    def _format_tools(self) -> str:
        """格式化工具描述"""
        lines = []
        for name, tool in self.tools.items():
            lines.append(f"- {name}: {tool.description}")
        return "\n".join(lines)

    def _format_scratchpad(self) -> str:
        """格式化历史步骤（scratchpad）"""
        lines = []
        for step in self.history:
            lines.append(f"Thought: {step.thought}")
            lines.append(f"Action: {step.action}")
            lines.append(f"Action Input: {step.action_input}")
            lines.append(f"Observation: {step.observation}")
            lines.append("")
        return "\n".join(lines)

    def _call_llm(self, question: str) -> str:
        """调用 LLM 获取下一步"""
        prompt = self.PROMPT_TEMPLATE.format(
            tools_description=self._format_tools(),
            tool_names=", ".join(self.tools.keys()),
            question=question,
            agent_scratchpad=self._format_scratchpad(),
        )

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"📡 调用 LLM（第 {len(self.history)+1} 步）...")
            print(f"{'='*60}")

        response = self.llm.simple_chat(prompt)
        return response

    def _parse_response(self, response: str) -> Optional[dict]:
        """解析 LLM 输出，提取 Thought/Action/Action Input 或 Final Answer"""
        # 检查是否是最终答案
        final_match = re.search(r"Final Answer:\s*(.+?)(?:\n|$)", response, re.DOTALL)
        if final_match:
            return {"type": "final", "answer": final_match.group(1).strip()}

        # 解析 Thought
        thought_match = re.search(
            r"Thought:\s*(.+?)(?:\nAction:|$)", response, re.DOTALL
        )
        thought = thought_match.group(1).strip() if thought_match else ""

        # 解析 Action
        action_match = re.search(
            r"Action:\s*(.+?)(?:\nAction Input:|$)", response, re.DOTALL
        )
        if not action_match:
            return {"type": "error", "message": "无法解析 Action", "raw": response}

        action = action_match.group(1).strip()

        # 解析 Action Input
        input_match = re.search(
            r"Action Input:\s*(.+?)(?:\nObservation:|\nThought:|$)", response, re.DOTALL
        )
        action_input = input_match.group(1).strip() if input_match else ""

        return {
            "type": "step",
            "thought": thought,
            "action": action,
            "action_input": action_input,
        }

    def _execute_tool(self, action: str, action_input: str) -> str:
        """执行工具"""
        tool = self.tools.get(action)
        if not tool:
            return f"错误：未知工具 '{action}'，可用工具：{list(self.tools.keys())}"
        try:
            result = tool.func(action_input)
            # 截断过长的观察结果
            if len(result) > 2000:
                result = result[:2000] + "...(截断)"
            return result
        except Exception as e:
            return f"工具执行错误: {e}"

    def run(self, question: str) -> str:
        """运行 Agent"""
        if self.verbose:
            print("\n🤖 ReAct Agent 启动")
            print(f"❓ 问题：{question}")
            print(f"🛠 可用工具：{list(self.tools.keys())}")
            print(f"📋 最大步数：{self.max_steps}")

        self.history = []

        for step_num in range(self.max_steps):
            # 1. 调用 LLM
            response = self._call_llm(question)

            if self.verbose:
                print(f"\n🤖 LLM 输出：\n{response}")

            # 2. 解析
            parsed = self._parse_response(response)

            if parsed is None:
                if self.verbose:
                    print("⚠️ 无法解析，重试")
                continue

            # 3. 如果是最终答案
            if parsed["type"] == "final":
                if self.verbose:
                    print(f"\n✅ 最终答案：{parsed['answer']}")
                return parsed["answer"]

            # 4. 如果是错误
            if parsed["type"] == "error":
                if self.verbose:
                    print(f"⚠️ 解析错误：{parsed['message']}")
                # 把错误作为观察反馈
                step = AgentStep(
                    thought=parsed.get("thought", ""),
                    action="",
                    action_input="",
                    observation=f"解析错误，请按格式输出。{parsed['message']}",
                )
                self.history.append(step)
                continue

            # 5. 执行工具
            if self.verbose:
                print(f"\n🔧 执行工具：{parsed['action']}({parsed['action_input']})")

            observation = self._execute_tool(parsed["action"], parsed["action_input"])

            if self.verbose:
                print(f"📤 观察结果：{observation[:200]}...")

            # 6. 记录步骤
            step = AgentStep(
                thought=parsed["thought"],
                action=parsed["action"],
                action_input=parsed["action_input"],
                observation=observation,
            )
            self.history.append(step)

        # 超过最大步数
        if self.verbose:
            print(f"\n⚠️ 达到最大步数 {self.max_steps}，强制结束")
        return f"Agent 未能在 {self.max_steps} 步内完成任务。最后状态：{self.history[-1].thought if self.history else '无'}"


# ========== 内置工具 ==========


def calculator(expression: str) -> str:
    """计算器工具"""
    import math

    safe_expr = expression.replace("^", "**")
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
    }
    try:
        result = eval(safe_expr, {"__builtins__": {}}, allowed)
        return str(result)
    except Exception as e:
        return f"计算错误: {e}"


def fake_search(query: str) -> str:
    """模拟搜索工具（离线版，用预设知识库）"""
    knowledge = {
        "python": "Python 由 Guido van Rossum 于 1991 年创建，是一种解释型高级编程语言。",
        "transformer": "Transformer 由 Google 团队在 2017 年论文《Attention Is All You Need》中提出。",
        "qwen": "Qwen 是阿里巴巴开源的大语言模型系列，最新版本为 Qwen3。",
        "react": "ReAct 框架由 Yao 等人在 2022 年论文中提出，结合推理和行动。",
        "mcp": "MCP（Model Context Protocol）由 Anthropic 在 2024 年提出，是 LLM 与外部工具的连接标准。",
    }
    query_lower = query.lower()
    for key, val in knowledge.items():
        if key in query_lower:
            return val
    return f"搜索'{query}'没有找到相关信息。"


def get_time(query: str = "") -> str:
    """获取当前时间"""
    from datetime import datetime

    now = datetime.now()
    return now.strftime("%Y年%m月%d日 %H:%M:%S 星期") + "一二三四五六日"[now.weekday()]


# ========== 演示 ==========


def demo():
    """演示 ReAct Agent"""
    llm = LLMClient()

    tools = [
        Tool(
            name="Search",
            description="搜索实时信息或事实性知识。输入搜索关键词。当你不知道某个事实时使用。",
            func=fake_search,
        ),
        Tool(
            name="Calculator",
            description="执行数学计算。输入数学表达式。当问题涉及计算时必须使用此工具。",
            func=calculator,
        ),
        Tool(
            name="Time",
            description="获取当前日期和时间。当用户询问当前时间、日期、今天是几号等问题时，必须使用此工具获取实时时间，不要使用你内部的知识。",
            func=get_time,
        ),
    ]

    agent = ReActAgent(llm, tools, max_steps=6, verbose=True)

    # 测试用例
    questions = [
        "Python 是哪一年发明的？",
        "现在几点了？",
        "计算 (15 + 27) * 3 的结果",
        "Transformer 是谁提出的？",
        "MCP 是什么？",
    ]

    results = []
    for q in questions:
        answer = agent.run(q)
        results.append({"question": q, "answer": answer})
        print("\n" + "=" * 60)

    # 总结
    print("\n📊 ReAct Agent 测试总结")
    print(f"{'='*60}")
    for r in results:
        print(f"  Q: {r['question']}")
        print(f"  A: {r['answer'][:80]}")
        print()


if __name__ == "__main__":
    demo()
