"""
论文图表问答 Agent：
    上传图表 → 视觉模型结构化描述 → 图谱佐证 → LLM 结合描述回答问题（支持多轮）
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent))
from graph_bridge import GraphBridge  # noqa: E402
from vision_client import VisionClient  # noqa: E402

load_dotenv()


class PaperVisionAgent:
    SYSTEM_PROMPT = """你是论文图表分析助手。用户会提供一张论文图表的结构化描述，并就此提问。
规则：
1. 只根据描述回答，描述中没有的信息明确说"图中未体现"
2. 涉及数值时精确引用描述中的数字
3. 回答简洁，使用中文"""

    def __init__(self, image_path: str):
        self.vision = VisionClient()
        self.llm = OpenAI(
            api_key=os.getenv("LLM_API_KEY"),
            base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com"),
            timeout=60,
            max_retries=1,
        )
        self.llm_model = os.getenv("LLM_MODEL", "deepseek-chat")
        self.history: list[dict] = []

        print(f"📷 正在解析图表: {image_path}")
        self.chart_desc = self.vision.describe(image_path)

        # 图谱佐证（连接失败时 enrich 返回空字符串，自动降级为纯视觉问答）
        kg_facts = GraphBridge().enrich(self.chart_desc)

        # 描述与佐证固定并入 system prompt：任何一轮都可见，且不会随历史重复注入
        self.system_prompt = self.SYSTEM_PROMPT + "\n\n[图表描述]\n" + self.chart_desc
        if kg_facts:
            self.system_prompt += "\n\n[知识图谱佐证]\n" + kg_facts

        print("✅ 图表解析完成，可以开始提问了\n")
        print(self.chart_desc + "\n" + "-" * 50)

    def ask(self, question: str) -> str:
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.history)
        messages.append({"role": "user", "content": question})

        resp = self.llm.chat.completions.create(
            model=self.llm_model,
            messages=messages,
            temperature=0.2,
        )
        answer = resp.choices[0].message.content
        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": answer})
        return answer


if __name__ == "__main__":
    import sys

    # 默认图相对本文件定位，任意 CWD 启动均可
    default_img = str(Path(__file__).resolve().parent.parent / "test.png")
    agent = PaperVisionAgent(sys.argv[1] if len(sys.argv) > 1 else default_img)
    while True:
        q = input("🧑 你: ").strip()
        if q in ("exit", "quit", "q"):
            break
        print(f"🤖 助手: {agent.ask(q)}\n")
