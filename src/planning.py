"""Agent 规划 4 种模式实现"""

import os
import sys
import re
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm_client import LLMClient


class PlanningPatterns:
    """4 种规划模式"""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    # ========== 1. CoT（思维链）==========

    def chain_of_thought(self, question: str) -> str:
        """线性推理"""
        prompt = f"""请一步一步思考并回答问题。

问题：{question}

请按以下格式回答：
思考步骤1：...
思考步骤2：...
...
最终答案：..."""
        return self.llm.simple_chat(prompt)

    # ========== 2. ToT（思维树）==========

    def tree_of_thoughts(
        self,
        question: str,
        num_branches: int = 3,
        max_depth: int = 2,
    ) -> str:
        """树状搜索：每步生成多个分支，评估选优"""
        print(f"🌳 ToT 启动：{num_branches} 分支 × {max_depth} 层")

        current_thoughts = [""]

        for depth in range(max_depth):
            print(f"\n--- 第 {depth+1} 层 ---")

            # 1. 生成分支
            all_branches = []
            for thought in current_thoughts:
                branches = self._generate_branches(question, thought, num_branches)
                all_branches.extend(branches)

            # 2. 评估并选优
            scored = self._evaluate_branches(question, all_branches)
            # 取 Top-N
            current_thoughts = [s["thought"] for s in scored[:num_branches]]

            for s in scored[:num_branches]:
                print(f"  [{s['score']:.2f}] {s['thought'][:60]}...")

        # 3. 基于最优思路生成答案
        best_thought = current_thoughts[0]
        prompt = f"""基于以下思考过程，回答问题：

问题：{question}
思考过程：{best_thought}

最终答案："""
        return self.llm.simple_chat(prompt)

    def _generate_branches(
        self, question: str, current_thought: str, n: int
    ) -> list[str]:
        """生成 n 个分支"""
        prompt = f"""问题：{question}
已有思考：{current_thought or "(开始)"}

请生成{n}个不同的下一步思考方向，每行一个。
重要规则：
- 每行只能包含一个思考方向
- 不要加任何编号（如 1. 2. 或 -）
- 不要加引号或其他符号
- 用简洁的短语描述

示例输出格式（生成3个）：
考虑用户画像和使用场景
选择适合的技术栈
设计核心对话流程

你的输出："""
        result = self.llm.simple_chat(prompt)
        branches = [b.strip() for b in result.strip().split("\n") if b.strip()]
        # 清理可能的编号
        branches = [re.sub(r"^[\d\.\-\*\•]+\s*", "", b) for b in branches]
        return branches[:n]

    def _evaluate_branches(self, question: str, branches: list[str]) -> list[dict]:
        """评估分支（打分 0-1）"""
        branches_text = "\n".join(f"[{i+1}] {b}" for i, b in enumerate(branches))
        prompt = f"""问题：{question}

以下是几个可能的思考方向：
{branches_text}

请为每个方向打分（0.0-1.0，越高越好）。

评分标准：
- 相关性：是否与问题相关
- 可行性：是否容易实施
- 完整性：是否覆盖关键方面

请严格按格式输出（每行一个）：
1: 0.85
2: 0.60
3: 0.90

你的评分："""
        result = self.llm.simple_chat(prompt)

        # 解析分数 - 支持多种格式
        scores = []
        for i, branch in enumerate(branches):
            # 尝试多种格式: "1: 0.85", "1. 0.85", "[1] 0.85"
            patterns = [
                rf"{i+1}\s*[:：.\]]\s*([\d.]+)",  # 1: 0.85, 1. 0.85, [1] 0.85
                rf"编号{i+1}\s*[:：]\s*([\d.]+)",  # 编号1: 0.85
            ]
            score = 0.5  # 默认分数
            for pattern in patterns:
                match = re.search(pattern, result)
                if match:
                    score = float(match.group(1))
                    break
            scores.append({"thought": branch, "score": min(max(score, 0.0), 1.0)})
        return sorted(scores, key=lambda x: x["score"], reverse=True)

    # ========== 3. Plan-and-Execute（先规划后执行）==========

    def plan_and_execute(self, task: str, executor=None) -> dict:
        """先规划后执行"""
        # 1. 生成计划
        print("📋 生成计划...")
        plan_prompt = f"""请将以下任务分解为 3-5 个具体步骤，用 JSON 数组格式输出：

任务：{task}

格式：
["步骤1", "步骤2", "步骤3"]

只输出 JSON，不要其他内容。"""

        plan_response = self.llm.simple_chat(plan_prompt)

        try:
            steps = json.loads(plan_response)
        except json.JSONDecodeError:
            # 容错：按行解析
            steps = [s.strip() for s in plan_response.strip().split("\n") if s.strip()]

        print(f"📋 计划：{len(steps)} 步")
        for i, step in enumerate(steps, 1):
            print(f"  {i}. {step}")

        # 2. 逐步执行
        results = []
        for i, step in enumerate(steps, 1):
            print(f"\n▶️ 执行步骤 {i}/{len(steps)}：{step}")

            if executor:
                # 用传入的执行器
                result = executor(step)
            else:
                # 默认用 LLM 执行
                result = self.llm.simple_chat(
                    f"请执行以下步骤：\n{step}\n\n任务背景：{task}"
                )

            results.append({"step": step, "result": result})
            print(f"  ✅ 结果：{result[:100]}...")

        # 3. 汇总
        summary = self.llm.simple_chat(
            f"任务：{task}\n执行结果：\n"
            + "\n".join(f"步骤'{r['step']}': {r['result'][:200]}" for r in results)
            + "\n\n请总结任务完成情况："
        )

        return {"plan": steps, "results": results, "summary": summary}

    # ========== 4. Reflexion（反思）==========

    def reflexion(
        self,
        question: str,
        max_attempts: int = 3,
        evaluator=None,
    ) -> dict:
        """反思模式：执行 → 评估 → 反思 → 改进"""
        print(f"🔄 Reflexion 启动（最多 {max_attempts} 次）")

        history = []
        best_answer = ""
        best_score = 0

        for attempt in range(max_attempts):
            print(f"\n--- 第 {attempt+1} 次尝试 ---")

            # 1. 生成回答（第一次直接答，后续基于反思改进）
            if attempt == 0:
                answer = self.llm.simple_chat(question)
            else:
                # 基于反思改进
                improve_prompt = f"""问题：{question}

之前的回答：{history[-1]['answer']}
反思意见：{history[-1]['reflection']}

请根据反思改进回答："""
                answer = self.llm.simple_chat(improve_prompt)

            print(f"  📝 回答：{answer[:100]}...")

            # 2. 评估
            if evaluator:
                score = evaluator(question, answer)
            else:
                score = self._self_evaluate(question, answer)

            print(f"  ⭐ 评分：{score:.2f}")

            history.append({"answer": answer, "score": score, "reflection": ""})

            if score > best_score:
                best_score = score
                best_answer = answer

            # 3. 如果够好，提前停止
            if score >= 0.95:  # 提高阈值，让更多迭代发生
                print(f"  ✅ 评分 {score:.2f} 优秀，停止迭代")
                break

            # 4. 反思
            if attempt < max_attempts - 1:
                reflection_prompt = f"""请审视以下回答，找出不足并提出改进建议：

问题：{question}
回答：{answer}
评分：{score:.2f}

反思（指出问题和改进方向）："""
                reflection = self.llm.simple_chat(reflection_prompt)
                history[-1]["reflection"] = reflection
                print(f"  💭 反思：{reflection[:100]}...")

        return {
            "best_answer": best_answer,
            "best_score": best_score,
            "attempts": len(history),
            "history": history,
        }

    def _self_evaluate(self, question: str, answer: str) -> float:
        """自我评估，返回 0-1 分数"""
        prompt = f"""请严格评估以下回答的质量（0.0-1.0）：

问题：{question}
回答：{answer}

评分标准（每项占25%）：
1. 准确性：内容是否准确无误？有没有事实性错误？
2. 完整性：是否完整回答了问题？有没有遗漏重要内容？
3. 清晰度：表达是否清晰？逻辑是否连贯？
4. 深度：是否有足够的深度？是否有独特见解？

严格要求：
- 90分以上需要非常优秀，几乎完美
- 70-89分是良好，有小瑕疵
- 60-69分是及格，有明显不足
- 60分以下需要大幅改进

请只输出一个分数（0-100的数字），如 85："""
        result = self.llm.simple_chat(prompt)
        match = re.search(r"(\d+)", result)
        if match:
            score = float(match.group(1)) / 100.0  # 转换为 0-1
            return min(max(score, 0.0), 1.0)
        return 0.5


# ========== 演示 ==========
def demo():
    llm = LLMClient()
    pp = PlanningPatterns(llm)

    print("=" * 60)
    print("模式 1：CoT（思维链）")
    print("=" * 60)
    print(
        pp.chain_of_thought(
            "一个水池进水管3小时注满，出水管5小时排空，同时开几小时注满？"
        )
    )

    print("\n" + "=" * 60)
    print("模式 2：ToT（思维树）")
    print("=" * 60)
    print(pp.tree_of_thoughts("如何设计一个客服机器人？", num_branches=3, max_depth=2))

    print("\n" + "=" * 60)
    print("模式 3：Plan-and-Execute")
    print("=" * 60)
    result = pp.plan_and_execute("写一篇关于 AI Agent 的科普文章")
    print(f"\n📋 总结：{result['summary'][:200]}...")

    print("\n" + "=" * 60)
    print("模式 4：Reflexion")
    print("=" * 60)
    # 用更难的问题来体现迭代过程
    result = pp.reflexion(
        "请详细解释量子计算的原理，并对比经典计算说明其优势", max_attempts=3
    )
    print(
        f"\n🏆 最佳回答（{result['attempts']}次尝试，评分 {result['best_score']:.2f}）："
    )
    print(result["best_answer"][:300])


if __name__ == "__main__":
    demo()
