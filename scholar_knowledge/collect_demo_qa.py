"""临时采集脚本：项目B 两跳问答 → 答案 + 推理路径（GIF 数据源）

用法:
    conda activate ai-agent
    python scholar_knowledge/collect_demo_qa.py
输出:
    scholar_knowledge/docs/demo_qa.json
"""

import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "src"))  # scholar_knowledge/src（graph_builder 等）
sys.path.insert(0, str(BASE.parent))  # 仓库根（src.llm_client 等）

from hybrid_rag import HybridRAG  # noqa: E402
from graph_builder import ScholarGraph  # noqa: E402
from src.llm_client import LLMClient  # noqa: E402
from src.embedder import Embedder  # noqa: E402

QUESTION = "从 ReAct 论文出发，沿着引用关系两步内能追溯到哪些奠基性论文？"


def main() -> None:
    """跑两跳问答，收集答案正文与推理路径实体链。"""
    rag = HybridRAG(ScholarGraph(), LLMClient(), Embedder())
    answer = rag.query(QUESTION, verbose=True)
    path = rag.last_path
    print("\n===== 答案 =====")
    print(answer)
    print("\n===== 推理路径 =====")
    print(path)

    out = BASE / "docs" / "demo_qa.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {"question": QUESTION, "answer": answer, "path": path},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n已写入 {out}")


if __name__ == "__main__":
    main()
