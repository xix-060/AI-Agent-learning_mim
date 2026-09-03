"""
查询改写：把口语化问题改写成"实体 + 关系"明确的检索友好形式
- 实体名补全：'Attention 那篇' → 《Attention Is All You Need》
- 关系显式化：'还写过啥' → 查询作者的其他论文
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class QueryRewriter:
    SYSTEM_PROMPT = """你是学术问答系统的查询改写器。把用户的口语化问题改写成实体和关系都明确的形式。

规则：
1. 论文/作者名补全为全称（可参考已知实体列表）
2. 隐含的关系要显式写出来
3. 只输出改写后的问题，不要解释

示例：
输入: Attention 那篇的作者还写过啥
输出: 查询论文《Attention Is All You Need》的所有作者，以及这些作者发表的其他论文

输入: 哪两个GAN论文的作者是同一拨人
输出: 查询 GAN 相关论文中，存在共同作者关系的论文对

输入: BERT被谁引用得最多
输出: 查询引用了论文《BERT》的论文，按被引次数排序
"""

    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("LLM_API_KEY"),
            base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com"),
        )
        self.model = os.getenv("LLM_MODEL", "deepseek-chat")

    def rewrite(self, question: str) -> str:
        """把口语化问题改写为实体+关系明确的检索友好形式。

        改写层是检索增强环节，失败不阻塞主流程：
        LLM 调用失败 / 无 API key / 返回空时，降级返回原始问题。
        """
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ],
                temperature=0.0,
            )
            rewritten = resp.choices[0].message.content.strip()
            if not rewritten:
                print("⚠️ 查询改写返回空，使用原始问题")
                return question
            return rewritten
        except Exception as e:
            print(f"⚠️ 查询改写失败（{e}），使用原始问题")
            return question


if __name__ == "__main__":
    rw = QueryRewriter()
    for q in [
        "Attention 那篇的作者还写过啥",
        "谁和Yann LeCun合著过论文",
        "transformer出来之前的预训练是怎么做的",
    ]:
        print(f"原始: {q}")
        print(f"改写: {rw.rewrite(q)}\n")
