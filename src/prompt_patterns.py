"""提示工程 6 大模式实战"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm_client import LLMClient
from src.models import Message, RoleEnum


class PromptPatterns:
    """6 种提示模式的实现"""

    def __init__(self):
        self.client = LLMClient()

    def _ask(
        self, messages: list[dict], temperature: float = 0.7, n: int = 1
    ) -> list[str]:
        """底层调用 LLM"""
        msg_objects = [
            Message(role=RoleEnum(m["role"]), content=m["content"]) for m in messages
        ]

        results = []
        for _ in range(n):
            resp = self.client.chat(msg_objects, temperature=temperature)
            results.append(resp.content)
        return results

    # ========== 模式 1：Zero-shot ==========

    def zero_shot(self, question: str) -> str:
        """零样本：直接提问"""
        messages = [
            {"role": "system", "content": "你是一个有用的助手。"},
            {"role": "user", "content": question},
        ]
        return self._ask(messages, temperature=0.0)[0]

    # ========== 模式 2：Few-shot ==========

    def few_shot(self, task: str, examples: list[tuple[str, str]]) -> str:
        """少样本：给几个示例让模型模仿"""
        prompt = "请根据以下示例完成任务：\n\n"
        for i, (input_text, output_text) in enumerate(examples, 1):
            prompt += f"示例{i}：\n输入：{input_text}\n输出：{output_text}\n\n"
        prompt += f"现在请处理：\n输入：{task}\n输出："

        messages = [{"role": "user", "content": prompt}]
        return self._ask(messages, temperature=0.0)[0]

    # ========== 模式 3：Chain-of-Thought ==========

    def chain_of_thought(self, question: str) -> str:
        """思维链：让模型一步一步推理"""
        messages = [
            {
                "role": "system",
                "content": "你是一个逻辑推理专家。请一步一步思考，最后给出答案。",
            },
            {"role": "user", "content": f"{question}\n\n请一步一步思考。"},
        ]
        return self._ask(messages, temperature=0.0)[0]

    # ========== 模式 4：Self-Consistency ==========

    def self_consistency(self, question: str, n: int = 5) -> dict:
        """自洽性：多次采样取多数投票"""
        messages = [
            {
                "role": "system",
                "content": "你是一个数学专家。请一步一步思考并给出最终答案。最终答案用 '答案是：X' 格式。",
            },
            {"role": "user", "content": f"{question}\n\n让我们一步一步思考。"},
        ]

        responses = self._ask(messages, temperature=0.8, n=n)

        import re
        from collections import Counter

        answers = []
        for resp in responses:
            match = re.search(r"答案是[：:]\s*(.+?)(?:\n|$)", resp)
            if match:
                answers.append(match.group(1).strip())

        vote = Counter(answers).most_common()

        return {
            "question": question,
            "num_samples": n,
            "answers": answers,
            "vote": vote,
            "final_answer": vote[0][0] if vote else responses[0],
            "all_responses": responses,
        }

    # ========== 模式 5：ReAct（简化版，第 4 周深入）==========

    def react_simple(self, question: str) -> str:
        """ReAct 简化版：展示 Thought-Action-Observation 格式"""
        prompt = f"""请用以下格式回答问题：

问题：{question}

思考（Thought）：分析需要什么信息
行动（Action）：决定要做什么
观察（Observation）：行动的结果
...（可重复多次）
答案（Answer）：最终答案

请严格按照这个格式回答。"""

        messages = [{"role": "user", "content": prompt}]
        return self._ask(messages, temperature=0.3)[0]

    # ========== 模式 6：Reflection ==========

    def reflection(self, question: str) -> dict:
        """反思：生成 → 自我批评 → 改进"""
        messages1 = [{"role": "user", "content": question}]
        initial = self._ask(messages1, temperature=0.7)[0]

        critique_prompt = f"""请审视以下回答，找出存在的问题（事实错误、逻辑漏洞、遗漏等）：

问题：{question}
回答：{initial}

请列出问题，如果没有问题请说明。"""
        messages2 = [{"role": "user", "content": critique_prompt}]
        critique = self._ask(messages2, temperature=0.3)[0]

        improve_prompt = f"""根据批评意见改进回答：

原问题：{question}
原回答：{initial}
批评意见：{critique}

请给出改进后的回答。"""
        messages3 = [{"role": "user", "content": improve_prompt}]
        improved = self._ask(messages3, temperature=0.5)[0]

        return {
            "initial": initial,
            "critique": critique,
            "improved": improved,
        }

    # ========== 演示 ==========

    def main(self):
        """测试 6 种模式"""
        pp = PromptPatterns()

        print("=" * 60)
        print("模式 1：Zero-shot")
        print("=" * 60)
        print(pp.zero_shot("用一句话解释什么是量子计算"))

        print("\n" + "=" * 60)
        print("模式 2：Few-shot")
        print("=" * 60)
        result = pp.few_shot(
            "这部电影太烂了，浪费我两小时",
            [
                ("这家餐厅味道真好", "正面"),
                ("服务态度很差", "负面"),
                ("性价比超高，推荐", "正面"),
            ],
        )
        print(f"情感分类结果：{result}")

        print("\n" + "=" * 60)
        print("模式 3：Chain-of-Thought")
        print("=" * 60)
        print(
            pp.chain_of_thought(
                "一个水池，进水管 3 小时注满，出水管 5 小时排空，同时开几小时注满？"
            )
        )

        print("\n" + "=" * 60)
        print("模式 4：Self-Consistency")
        print("=" * 60)
        result = pp.self_consistency(
            "一个班 40 人，其中 60% 是女生，女生中有 3/4 喜欢运动，喜欢运动的女生有几人？",
            n=5,
        )
        print(f"投票结果：{result['vote']}")
        print(f"最终答案：{result['final_answer']}")

        print("\n" + "=" * 60)
        print("模式 5：ReAct（简化）")
        print("=" * 60)
        print(pp.react_simple("北京和上海今天的气温差是多少度？"))

        print("\n" + "=" * 60)
        print("模式 6：Reflection")
        print("=" * 60)
        result = pp.reflection("请解释 HTTP 和 HTTPS 的区别")
        print(f"初始回答：\n{result['initial'][:200]}...")
        print(f"\n批评：\n{result['critique'][:200]}...")
        print(f"\n改进后：\n{result['improved'][:200]}...")


if __name__ == "__main__":
    PromptPatterns().main()
