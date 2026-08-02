"""第一个完整 Agent：ReAct + Memory + Tools"""

import os
import sys
import re
import json
from dataclasses import dataclass
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm_client import LLMClient
from src.tools.builtin import create_default_registry, ToolRegistry
from src.memory import AgentMemory, ShortTermMemory
from src.embedder import Embedder


@dataclass
class AgentConfig:
    """Agent 配置"""

    max_steps: int = 8
    verbose: bool = True
    system_prompt: str = """你是一个智能助手，能够通过工具完成任务。

你有短期记忆（当前对话）和长期记忆（过往知识）。

工作流程：
1. 分析用户问题
2. 思考需要什么信息或操作
3. 调用合适的工具
4. 根据工具返回结果继续思考
5. 如果信息足够，给出最终答案

重要规则：
- 如果用户只是打招呼（如"你好"、"hi"），直接友好回应，不需要调用工具
- 如果用户有明确问题，按照 ReAct 格式输出 Thought -> Action -> Observation -> Final Answer
- 当你确定答案时，输出 Final Answer

请始终先思考（Thought），再决定行动（Action）。"""


class CompleteAgent:
    """完整 Agent：ReAct 循环 + 记忆 + 工具"""

    def __init__(
        self,
        llm: LLMClient,
        registry: ToolRegistry,
        memory: Optional[AgentMemory] = None,
        config: AgentConfig = AgentConfig(),
    ):
        self.llm = llm
        self.registry = registry
        self.memory = memory or AgentMemory(Embedder(), llm)
        self.config = config
        self.short_memory = ShortTermMemory(max_messages=20)

    def _build_prompt(self, question: str) -> str:
        """构建完整 Prompt"""
        # 1. 工具描述
        tools_desc = self.registry.get_descriptions()

        # 2. 记忆上下文
        memory_context = (
            self.memory.get_context_for_query(question) if self.memory else ""
        )

        # 3. 短期记忆（当前对话历史）
        conversation = self.short_memory.get_context()

        # 4. 构建 Prompt
        prompt = f"""{self.config.system_prompt}

【可用工具】
{tools_desc}

【工具调用格式】
当你需要使用工具时，输出：
Thought: 你的思考
Action: 工具名称
Action Input: {{"参数名": "参数值"}}

当你确定答案时，输出：
Thought: 我现在知道答案了
Final Answer: 最终答案

【记忆上下文】
{memory_context if memory_context != "（无相关记忆）" else "（无）"}

【当前对话】
{conversation if conversation else "（开始新对话）"}

【用户问题】
{question}

请开始："""

        return prompt

    def _call_llm(self, prompt: str) -> str:
        """调用 LLM"""
        if self.config.verbose:
            print(f"\n{'='*60}")
            print("📡 思考中...")
            print(f"{'='*60}")

        response = self.llm.simple_chat(prompt)
        return response

    def _parse_response(self, response: str) -> dict:
        """解析 LLM 输出"""
        # 检查最终答案
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
            return {"type": "error", "message": "无法解析 Action", "thought": thought}

        action = action_match.group(1).strip()

        # 解析 Action Input（JSON 格式）
        input_match = re.search(
            r"Action Input:\s*(.+?)(?:\nObservation:|\nThought:|$)", response, re.DOTALL
        )
        if input_match:
            input_str = input_match.group(1).strip()
            try:
                action_input = json.loads(input_str)
            except json.JSONDecodeError:
                # 容错：当作纯字符串
                action_input = {"input": input_str}
        else:
            action_input = {}

        return {
            "type": "step",
            "thought": thought,
            "action": action,
            "action_input": action_input,
        }

    def run(self, question: str) -> dict:
        """运行 Agent"""
        if self.config.verbose:
            print("\n🤖 Agent 启动")
            print(f"❓ 问题：{question}")
            print(f"🛠 工具：{self.registry.list_tools()}")

        # 记录到短期记忆
        self.short_memory.add("user", question)

        steps = []

        for step_num in range(self.config.max_steps):
            # 1. 构建 Prompt
            prompt = self._build_prompt(question)

            # 2. 调用 LLM
            response = self._call_llm(prompt)

            if self.config.verbose:
                print(f"\n🤖 LLM 输出：\n{response[:500]}")

            # 3. 解析
            parsed = self._parse_response(response)

            # 4. 处理
            if parsed["type"] == "final":
                answer = parsed["answer"]
                if self.config.verbose:
                    print(f"\n✅ 最终答案：{answer}")

                # 记录到记忆
                self.short_memory.add("assistant", answer)
                if self.memory:
                    self.memory.store_fact(f"Q: {question} A: {answer}")
                    self.memory.record_episode(
                        task=question,
                        actions=[s["action"] for s in steps if "action" in s],
                        result=answer,
                        success=True,
                    )

                return {
                    "question": question,
                    "answer": answer,
                    "steps": steps,
                    "success": True,
                }

            if parsed["type"] == "error":
                if self.config.verbose:
                    print(f"⚠️ 解析错误：{parsed['message']}")
                steps.append(
                    {"thought": parsed.get("thought", ""), "error": parsed["message"]}
                )
                continue

            # 5. 执行工具
            action = parsed["action"]
            action_input = parsed["action_input"]

            if self.config.verbose:
                print(f"\n🔧 执行：{action}({action_input})")

            observation = self.registry.execute(action, **action_input)

            if self.config.verbose:
                print(f"📤 结果：{observation[:200]}")

            steps.append(
                {
                    "thought": parsed["thought"],
                    "action": action,
                    "action_input": action_input,
                    "observation": observation,
                }
            )

            # 6. 把观察结果加入短期记忆（作为系统消息）
            self.short_memory.add("system", f"工具 {action} 返回：{observation[:500]}")

        # 超过最大步数
        if self.config.verbose:
            print(f"\n⚠️ 达到最大步数 {self.config.max_steps}")

        fallback = "抱歉，我无法在限定步数内完成任务。"
        return {
            "question": question,
            "answer": fallback,
            "steps": steps,
            "success": False,
        }

    def chat(self):
        """交互式对话"""
        print("🤖 Agent 已就绪（输入 quit 退出）")
        print("=" * 60)

        while True:
            question = input("\n🙋 我：")
            if question.strip().lower() in ("quit", "exit", "q"):
                print("👋 再见！")
                break

            if not question.strip():
                print("⚠️ 请输入有效问题！")
                continue

            result = self.run(question)
            print(f"\n🤖 Agent：{result['answer']}")


# ========== 评测任务 ==========
def run_eval():
    """运行 5 个评测任务"""
    llm = LLMClient()
    registry = create_default_registry()

    # 不用长期记忆（评测要独立）
    agent = CompleteAgent(
        llm,
        registry,
        memory=None,  # 评测时关闭长期记忆
        config=AgentConfig(max_steps=6, verbose=True),
    )

    eval_tasks = [
        {
            "question": "现在几点了？",
            "expected_keyword": "202",
            "description": "时间查询",
        },
        {
            "question": "计算 123 * 456 + 789 的结果",
            "expected_keyword": "57297",
            "description": "数学计算",
        },
        {
            "question": "帮我列出当前目录的文件",
            "expected_keyword": "📁",
            "description": "文件系统操作",
        },
        {
            "question": "帮我创建一个文件 test_agent.txt，内容是'Agent 测试成功'",
            "expected_keyword": "已写入",
            "description": "文件写入",
        },
        {
            "question": "搜索 Python 编程语言的信息",
            "expected_keyword": "Python",
            "description": "网络搜索",
        },
    ]

    results = []
    for i, task in enumerate(eval_tasks, 1):
        print(f"\n{'#'*60}")
        print(f"# 评测任务 {i}/5：{task['description']}")
        print(f"# 问题：{task['question']}")
        print(f"{'#'*60}")

        result = agent.run(task["question"])

        success = task["expected_keyword"] in result["answer"] or any(
            task["expected_keyword"] in str(s.get("observation", ""))
            for s in result["steps"]
        )

        results.append(
            {
                "task": task["description"],
                "question": task["question"],
                "answer": result["answer"][:100],
                "success": success,
                "steps_count": len(result["steps"]),
            }
        )

        print(f"\n{'✅' if success else '❌'} 结果：{'成功' if success else '失败'}")

    # 总结
    success_count = sum(r["success"] for r in results)
    print(f"\n{'='*60}")
    print(
        f"📊 评测总结：{success_count}/{len(results)} 成功（{success_count/len(results):.0%}）"
    )
    print(f"{'='*60}")
    for r in results:
        print(f"  {'✅' if r['success'] else '❌'} {r['task']}: {r['answer'][:60]}")

    return results


def main():
    """主入口：评测 or 交互"""
    if len(sys.argv) > 1 and sys.argv[1] == "eval":
        run_eval()
    else:
        llm = LLMClient()
        registry = create_default_registry()
        agent = CompleteAgent(llm, registry)
        agent.chat()


if __name__ == "__main__":
    main()
