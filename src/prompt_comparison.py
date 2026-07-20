"""对比不同提示模式在数学题上的表现"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.prompt_patterns import PromptPatterns


def main():
    pp = PromptPatterns()

    math_questions = [
        "小明有 10 颗糖，吃了 3 颗，又得到 5 颗，现在有几颗？",
        "一个长方形长 8 米宽 5 米，周长是多少？",
        "如果 3 个苹果 15 元，7 个苹果多少钱？",
    ]

    for q in math_questions:
        print(f"\n{'='*60}")
        print(f"问题：{q}")
        print(f"{'='*60}")

        print("\n[Zero-shot]")
        print(pp.zero_shot(q))

        print("\n[Chain-of-Thought]")
        print(pp.chain_of_thought(q))


if __name__ == "__main__":
    main()
