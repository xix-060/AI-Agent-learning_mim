"""生成实验用样本数据"""

import os

sample_text = """
人工智能（Artificial Intelligence，AI）是计算机科学的一个分支，致力于研究和开发能够模拟、延伸和扩展人类智能的理论、方法、技术和应用系统。

人工智能的发展经历了三次浪潮。第一次浪潮（1956-1974）以符号主义为代表，Alan Turing 提出了图灵测试，John McCarthy 在达特茅斯会议上首次提出"人工智能"概念。第二次浪潮（1980-1987）以专家系统和连接主义为代表，出现了反向传播算法。第三次浪潮（2006年至今）以深度学习为代表，Geoffrey Hinton 提出了深度信念网络，GPU 的普及使大规模训练成为可能。

机器学习是人工智能的核心子领域，它使计算机系统能够从数据中学习并改进，而无需明确编程。机器学习主要分为三类：监督学习（分类、回归）、无监督学习（聚类、降维）和强化学习（通过奖励信号学习策略）。

深度学习是机器学习的一个子领域，使用多层神经网络。常见的深度学习模型包括：卷积神经网络（CNN，用于图像）、循环神经网络（RNN，用于序列）、Transformer（用于自然语言处理）。Transformer 由 Google 团队在 2017 年提出，成为了大语言模型的基础架构。

大语言模型（LLM）是基于 Transformer 架构的大规模语言模型。GPT 系列由 OpenAI 开发，包括 GPT-3、GPT-4 等版本。LLaMA 由 Meta 开源，Qwen 由阿里巴巴开发。这些模型通过预训练和微调两个阶段获得能力。

RAG（检索增强生成）技术通过结合外部知识库来增强大语言模型的能力。它解决了大模型知识截止、幻觉问题、领域知识不足等问题。RAG 系统通常包括文档处理、向量化、检索、生成四个步骤。

Agent（智能体）是能够感知环境、做出决策、执行行动的 AI 系统。基于 LLM 的 Agent 通常包含四个核心组件：规划（Planning）、记忆（Memory）、工具使用（Tool Use）和行动执行（Action）。ReAct 模式是 Agent 的经典范式，通过 Thought-Action-Observation 循环实现推理与行动的交替。

MCP（Model Context Protocol）是 Anthropic 在 2024 年提出的开放协议，旨在标准化 LLM 与外部工具和资源的连接方式。MCP 采用客户端-服务器架构，使 AI 应用能够以统一的方式访问文件系统、数据库、API 等外部资源。
""".strip()


def generate_sample(output_path: str = "data/sample_knowledge.txt"):
    """生成样本数据文件"""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(sample_text)
        print(f"✅ 生成样本数据：{len(sample_text)} 字")
        print(f"📄 输出文件：{output_path}")
    except OSError as e:
        print(f"❌ 写入失败：{e}")
        raise


if __name__ == "__main__":
    generate_sample()
