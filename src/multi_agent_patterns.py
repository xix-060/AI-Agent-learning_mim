"""Multi-Agent 三种范式实现（简化版，不用框架）"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from dataclasses import dataclass
from src.llm_client import LLMClient
from src.models import Message, RoleEnum


@dataclass
class Agent:
    """单个 Agent 定义"""

    name: str
    role: str  # 角色描述
    system_prompt: str
    llm: LLMClient

    def run(self, input_text: str, context: str = "") -> str:
        """执行 Agent"""
        messages = [
            Message(role=RoleEnum.SYSTEM, content=self.system_prompt),
        ]
        if context:
            messages.append(
                Message(role=RoleEnum.SYSTEM, content=f"上下文：\n{context}")
            )
        messages.append(Message(role=RoleEnum.USER, content=input_text))

        response = self.llm.chat(messages)
        return response.content


# ========== 1. Supervisor 模式 ==========


class SupervisorPattern:
    """监督者模式：Supervisor 调度多个 Worker"""

    def __init__(self, llm: LLMClient):
        self.llm = llm

        # 创建 Supervisor
        self.supervisor = Agent(
            name="Supervisor",
            role="任务调度",
            system_prompt="""你是一个任务调度器。分析用户任务，决定由哪个 Agent 处理。
可用 Agent：
- Researcher：搜索信息、查资料
- Calculator：数学计算、数据分析
- Writer：写文章、总结、翻译

请输出 JSON：{"agent": "Agent名称", "task": "分配给该Agent的具体任务"}
如果任务完成，输出：{"agent": "done", "task": "最终答案"}""",
            llm=llm,
        )

        # 创建 Workers
        self.workers = {
            "Researcher": Agent(
                name="Researcher",
                role="搜索",
                system_prompt="你是一个搜索专家。根据问题提供相关信息。简洁回答，200字以内。",
                llm=llm,
            ),
            "Calculator": Agent(
                name="Calculator",
                role="计算",
                system_prompt="你是一个数学专家。解决数学问题，展示计算过程。",
                llm=llm,
            ),
            "Writer": Agent(
                name="Writer",
                role="写作",
                system_prompt="你是一个写作专家。根据信息写文章、总结或翻译。",
                llm=llm,
            ),
        }

    def run(self, task: str, max_rounds: int = 5) -> dict:
        """运行 Supervisor 模式"""
        print(f"\n{'=' * 60}")
        print("👔 Supervisor 模式启动")
        print(f"📝 任务：{task}")
        print(f"{'=' * 60}")

        context = ""
        history = []

        for round_num in range(max_rounds):
            print(f"\n--- 第 {round_num + 1} 轮 ---")

            # 1. Supervisor 决策
            decision = self.supervisor.run(
                f"任务：{task}\n当前上下文：{context or '（无）'}\n\n请决定下一步。"
            )
            print(f"👔 Supervisor：{decision}")

            # 解析决策
            try:
                # 提取 JSON
                import re

                json_match = re.search(r"\{[^}]+\}", decision)
                if json_match:
                    decision_data = json.loads(json_match.group())
                else:
                    decision_data = {"agent": "done", "task": decision}
            except json.JSONDecodeError:
                decision_data = {"agent": "done", "task": decision}

            agent_name = decision_data.get("agent", "done")
            subtask = decision_data.get("task", "")

            # 2. 检查是否完成
            if agent_name.lower() == "done":
                print(f"\n✅ 任务完成：{subtask}")
                return {
                    "task": task,
                    "result": subtask,
                    "rounds": round_num + 1,
                    "history": history,
                }

            # 3. Worker 执行
            worker = self.workers.get(agent_name)
            if not worker:
                print(f"⚠️ 未知 Agent：{agent_name}")
                context += f"\n错误：未知 Agent {agent_name}"
                continue

            print(f"\n🤖 {agent_name} 正在处理：{subtask}")
            result = worker.run(subtask, context=context)
            print(f"📤 {agent_name} 结果：{result[:150]}...")

            # 4. 更新上下文
            context += f"\n{agent_name} 的结果：{result}"
            history.append(
                {
                    "round": round_num + 1,
                    "supervisor_decision": decision,
                    "agent": agent_name,
                    "subtask": subtask,
                    "result": result,
                }
            )

        return {
            "task": task,
            "result": context,
            "rounds": max_rounds,
            "history": history,
        }


# ========== 2. Group Chat 模式 ==========


class GroupChatPattern:
    """群聊模式：多个 Agent 轮流发言"""

    def __init__(self, llm: LLMClient):
        self.llm = llm
        self.agents = [
            Agent(
                name="产品经理",
                role="PM",
                system_prompt="你是一个产品经理。从用户需求和可行性角度分析问题。每次发言100字以内。",
                llm=llm,
            ),
            Agent(
                name="工程师",
                role="开发",
                system_prompt="你是一个资深工程师。从技术实现角度分析问题。每次发言100字以内。",
                llm=llm,
            ),
            Agent(
                name="设计师",
                role="设计",
                system_prompt="你是一个 UI/UX 设计师。从用户体验角度分析问题。每次发言100字以内。",
                llm=llm,
            ),
        ]
        self.moderator = Agent(
            name="主持人",
            role="Moderator",
            system_prompt="""你是一个讨论主持人。根据讨论内容，决定下一个发言者。
可用发言者：产品经理、工程师、设计师
请只输出发言者名字，如"工程师"。如果讨论充分可以输出"结束"。""",
            llm=llm,
        )

    def run(self, topic: str, max_turns: int = 6) -> dict:
        """运行群聊"""
        print(f"\n{'=' * 60}")
        print("💬 Group Chat 模式启动")
        print(f"📝 话题：{topic}")
        print(f"{'=' * 60}")

        transcript = []
        context = f"讨论话题：{topic}\n\n"

        for turn in range(max_turns):
            # 1. 主持人决定下一个发言者
            if turn == 0:
                speaker_name = "产品经理"  # 第一轮固定
            else:
                next_speaker = self.moderator.run(
                    f"{context}\n\n谁应该下一个发言？（只输出名字）"
                )
                speaker_name = next_speaker.strip()
                if "结束" in speaker_name:
                    print("\n🎤 主持人：讨论结束")
                    break

            # 找到发言者
            speaker = next((a for a in self.agents if a.name in speaker_name), None)
            if not speaker:
                speaker = self.agents[turn % len(self.agents)]

            # 2. 发言
            print(f"\n🎤 {speaker.name}：")
            speech = speaker.run(
                f"请就以下话题发表你的观点（100字以内）：\n{topic}\n\n之前的讨论：\n{context[-500:]}"
            )
            print(f"   {speech}")

            # 3. 记录
            transcript.append(
                {"turn": turn + 1, "speaker": speaker.name, "speech": speech}
            )
            context += f"{speaker.name}：{speech}\n"

        # 总结
        summary = self.llm.simple_chat(
            f"请总结以下讨论的结论（200字以内）：\n{context}",
            system_prompt="你是一个讨论总结者。",
        )

        return {
            "topic": topic,
            "transcript": transcript,
            "summary": summary,
            "turns": len(transcript),
        }


# ========== 3. Workflow 模式 ==========


class WorkflowPattern:
    """工作流模式：固定流程，Agent 按顺序处理"""

    def __init__(self, llm: LLMClient):
        self.llm = llm

        # 定义工作流步骤
        self.steps = [
            Agent(
                name="调研员",
                role="Research",
                system_prompt="你是一个调研员。根据主题收集相关信息，列出要点。200字以内。",
                llm=llm,
            ),
            Agent(
                name="分析师",
                role="Analysis",
                system_prompt="你是一个分析师。基于调研信息，分析关键洞察。200字以内。",
                llm=llm,
            ),
            Agent(
                name="撰稿人",
                role="Writing",
                system_prompt="你是一个撰稿人。基于调研和分析，写一篇结构清晰的文章。300字左右。",
                llm=llm,
            ),
            Agent(
                name="审稿人",
                role="Review",
                system_prompt="你是一个审稿人。检查文章质量，提出修改建议或确认通过。100字以内。",
                llm=llm,
            ),
        ]

    def run(self, task: str) -> dict:
        """运行工作流"""
        print(f"\n{'=' * 60}")
        print("🔄 Workflow 模式启动")
        print(f"📝 任务：{task}")
        print(f"{'=' * 60}")

        context = ""
        results = []

        for i, agent in enumerate(self.steps):
            print(f"\n--- 步骤 {i + 1}/{len(self.steps)}: {agent.name} ---")

            # 每个 Agent 处理上一步的结果
            if i == 0:
                input_text = task
            else:
                input_text = f"原始任务：{task}\n\n上一步结果：{context}"

            result = agent.run(input_text)
            print(f"📤 {agent.name} 输出：{result[:150]}...")

            context = result
            results.append(
                {
                    "step": i + 1,
                    "agent": agent.name,
                    "output": result,
                }
            )

        return {
            "task": task,
            "final_result": context,
            "steps": results,
        }


# ========== 演示 ==========
def demo():
    llm = LLMClient()

    # 1. Supervisor
    print("\n" + "#" * 60)
    print("# 范式 1：Supervisor")
    print("#" * 60)
    sup = SupervisorPattern(llm)
    sup.run("帮我研究一下 RAG 技术，并写一段 100 字的介绍")

    # 2. Group Chat
    print("\n" + "#" * 60)
    print("# 范式 2：Group Chat")
    print("#" * 60)
    gc = GroupChatPattern(llm)
    result = gc.run("是否应该用 LangGraph 还是 CrewAI 来构建 Agent？")
    print(f"\n📋 讨论总结：{result['summary']}")

    # 3. Workflow
    print("\n" + "#" * 60)
    print("# 范式 3：Workflow")
    print("#" * 60)
    wf = WorkflowPattern(llm)
    result = wf.run("AI Agent 的发展趋势")
    print(f"\n📄 最终文章：{result['final_result'][:200]}...")


if __name__ == "__main__":
    demo()
