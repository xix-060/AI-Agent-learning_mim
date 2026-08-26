"""用 LLM 蒸馏生成 SFT 训练数据"""

import json
import re
import random
from pathlib import Path
from src.llm_client import LLMClient
from src.models import Message, RoleEnum


# ========== 1. 种子任务生成器 ==========


def generate_seed_tasks(domain: str, count: int = 20) -> list[str]:
    """生成种子任务（让 LLM 自己生成问题）"""
    llm = LLMClient()

    prompt = f"""请为"{domain}"领域生成{count}个多样化的指令任务。
这些任务应该：
1. 涵盖不同难度
2. 表述方式多样
3. 适合训练 AI 助手

请用 JSON 数组格式输出，每个元素是一个任务描述字符串。
只输出 JSON，不要其他内容。"""

    response = llm.chat(
        [
            Message(role=RoleEnum.SYSTEM, content="你是一个任务生成器，只输出 JSON。"),
            Message(role=RoleEnum.USER, content=prompt),
        ],
        temperature=0.8,
    ).content

    # 解析 JSON
    try:
        json_match = re.search(r"\[.*\]", response, re.DOTALL)
        if json_match:
            tasks = json.loads(json_match.group())
            return tasks[:count]
    except json.JSONDecodeError:
        pass

    # Fallback：用预设任务
    return get_fallback_tasks(domain, count)


def get_fallback_tasks(domain: str, count: int) -> list[str]:
    """预设任务模板"""
    templates = {
        "AI Agent": [
            "什么是 AI Agent？",
            "Agent 和聊天机器人有什么区别？",
            "ReAct 框架是什么？",
            "Agent 如何使用工具？",
            "解释 Agent 的记忆机制",
            "LangGraph 和 LangChain 的区别？",
            "Multi-Agent 有哪些协作模式？",
            "Agent 如何做规划？",
            "什么是 Function Calling？",
            "MCP 协议解决什么问题？",
        ],
        "RAG": [
            "什么是 RAG？",
            "RAG 如何减少幻觉？",
            "向量检索的原理是什么？",
            "如何选择 Embedding 模型？",
            "切块策略有哪些？",
            "什么是 Reranker？",
            "HyDE 是什么？",
            "如何评估 RAG 系统？",
            "GraphRAG 和普通 RAG 的区别？",
            "RAG 的常见失败原因？",
        ],
    }

    tasks = templates.get(domain, templates["AI Agent"])
    # 随机打乱并扩展
    random.shuffle(tasks)
    while len(tasks) < count:
        tasks = tasks * 2
    return tasks[:count]


# ========== 2. 蒸馏生成 ==========


def distill_answers(tasks: list[str], domain: str) -> list[dict]:
    """用 LLM 生成回答"""
    llm = LLMClient()

    system_prompt = f"""你是{domain}领域的专家。请简洁、准确地回答问题。
回答要求：
1. 50-200字
2. 结构清晰
3. 通俗易懂
4. 有具体例子更好"""

    dataset = []
    for i, task in enumerate(tasks):
        print(f"  [{i+1}/{len(tasks)}] {task}")

        answer = llm.simple_chat(task, system_prompt=system_prompt)

        dataset.append(
            {
                "instruction": task,
                "input": "",
                "output": answer,
            }
        )

    return dataset


# ========== 3. 数据清洗 ==========


def clean_dataset(data: list[dict]) -> list[dict]:
    """清洗数据"""
    cleaned = []
    seen = set()  # 去重

    for item in data:
        # 跳过空回答
        if not item["output"] or len(item["output"].strip()) < 10:
            continue

        # 跳过过长回答
        if len(item["output"]) > 2000:
            item["output"] = item["output"][:2000]

        # 去重
        key = item["instruction"] + item["output"][:50]
        if key in seen:
            continue
        seen.add(key)

        cleaned.append(item)

    print(f"  清洗：{len(data)} → {len(cleaned)} 条")
    return cleaned


# ========== 4. 主函数 ==========


def main():
    """蒸馏生成数据集"""
    domains = ["AI Agent", "RAG"]

    all_data = []

    for domain in domains:
        print(f"\n{'='*60}")
        print(f"📡 蒸馏 {domain} 领域数据")
        print(f"{'='*60}")

        # 1. 生成种子任务
        print("\n1. 生成种子任务...")
        tasks = generate_seed_tasks(domain, count=15)
        print(f"  生成 {len(tasks)} 个任务")

        # 2. 蒸馏回答
        print("\n2. 蒸馏回答...")
        data = distill_answers(tasks, domain)

        # 3. 清洗
        print("\n3. 清洗数据...")
        data = clean_dataset(data)

        all_data.extend(data)

    # 保存
    output_path = "data/distilled_dataset.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 蒸馏完成：{len(all_data)} 条数据保存到 {output_path}")

    # 统计
    print("\n📊 数据统计：")
    for domain in domains:
        domain_data = [
            d
            for d in all_data
            if domain.lower() in d["instruction"].lower()
            or any(
                kw in d["instruction"]
                for kw in {
                    "AI Agent": ["Agent", "agent", "ReAct", "MCP", "LangGraph"],
                    "RAG": ["RAG", "rag", "向量", "Embedding", "Reranker"],
                }[domain]
            )
        ]
        print(f"  {domain}: {len(domain_data)} 条")

    # 打印示例
    print("\n📝 示例数据：")
    for d in all_data[:3]:
        print(f"  Q: {d['instruction']}")
        print(f"  A: {d['output'][:100]}...")
        print()


if __name__ == "__main__":
    main()
