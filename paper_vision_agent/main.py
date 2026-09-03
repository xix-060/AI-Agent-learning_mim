"""CLI 入口：python main.py --image 论文图.png"""

import argparse
from src.agent import PaperVisionAgent

parser = argparse.ArgumentParser()
parser.add_argument("--image", required=True, help="论文图表路径")
args = parser.parse_args()

agent = PaperVisionAgent(args.image)
while True:
    try:
        q = input("🧑 你: ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if not q:
        continue
    if q in ("exit", "quit", "q"):
        break
    print(f"🤖 助手: {agent.ask(q)}\n")
