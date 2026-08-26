"""准备 DPO 偏好数据"""

import json
from pathlib import Path


# 偏好数据：prompt + chosen（好回答） + rejected（差回答）
DPO_DATA = [
    {
        "prompt": "如何学习 AI？",
        "chosen": "建议分三步：1）学 Python 基础；2）学机器学习经典算法（线性回归、决策树）；3）学深度学习（PyTorch + Transformer）。推荐资源：吴恩达课程、李宏毅课程。动手做项目最重要。",
        "rejected": "AI 很难，你学不会的。",
    },
    {
        "prompt": "Python 和 Java 选哪个？",
        "chosen": "取决于你的目标。Python 适合数据科学、AI、快速原型开发，语法简洁。Java 适合企业级应用、Android 开发，类型安全。如果你做 AI/数据方向，选 Python；如果做后端/企业开发，选 Java。",
        "rejected": "Python 好用。",
    },
    {
        "prompt": "什么是 RAG？",
        "chosen": "RAG（检索增强生成）是一种结合检索和生成的技术。它先从知识库检索相关文档，然后把文档作为上下文发给 LLM 生成答案。优点是减少幻觉、知识可更新、可溯源。",
        "rejected": "RAG 就是一个搜索。",
    },
    {
        "prompt": "如何写好代码？",
        "chosen": "写好代码的关键：1）命名清晰（变量名表达意图）；2）函数单一职责；3）写注释解释为什么，不是什么；4）写测试；5）持续重构。推荐读《Clean Code》。",
        "rejected": "多写就行了。",
    },
    {
        "prompt": "什么是 Agent？",
        "chosen": "AI Agent 是能感知环境、做决策、执行行动的智能体。基于 LLM 的 Agent 通常包含四个组件：规划（Planning）、记忆（Memory）、工具使用（Tool Use）、行动循环（Action Loop）。代表框架有 LangGraph、AutoGen。",
        "rejected": "Agent 就是机器人。",
    },
    {
        "prompt": "如何提高编程效率？",
        "chosen": "提高编程效率的方法：1）使用 IDE 的快捷键和自动补全；2）用 AI 编程助手（Cursor、Copilot）；3）写可复用的函数/组件；4）学会调试技巧；5）保持专注，用番茄工作法。",
        "rejected": "多加班就行了。",
    },
    {
        "prompt": "什么是向量数据库？",
        "chosen": "向量数据库是专门存储和检索向量的数据库。它把文本/图片转成向量后存储，通过相似度计算实现语义检索。常见的有 Chroma（轻量）、Milvus（分布式）、Faiss（高性能库）。在 RAG 系统中是核心组件。",
        "rejected": "就是数据库。",
    },
    {
        "prompt": "LoRA 是什么？",
        "chosen": "LoRA（Low-Rank Adaptation）是一种参数高效微调方法。它冻结原始权重，只训练两个小矩阵 A 和 B（ΔW = BA）。由于 A、B 的秩远小于原始维度，参数量减少 99%+，但效果接近全参数微调。QLoRA = 量化 + LoRA。",
        "rejected": "LoRA 是一种模型。",
    },
]


def save_dpo_data(path="data/dpo_dataset.json"):
    """保存 DPO 数据"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(DPO_DATA, f, ensure_ascii=False, indent=2)
    print(f"✅ 保存 {len(DPO_DATA)} 条偏好数据到 {path}")


if __name__ == "__main__":
    save_dpo_data()
