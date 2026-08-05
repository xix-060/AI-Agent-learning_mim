"""CrewAI 多 Agent 协作实战"""

import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM

load_dotenv()


# ========== 配置 LLM ==========
# CrewAI 1.15+ 使用内置的 LLM 类，支持 LiteLLM 兼容接口

llm = LLM(
    model=os.getenv("LLM_MODEL", "qwen-turbo"),
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv(
        "LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ),
    temperature=0.7,
)


# ========== 1. 定义 Agent ==========

researcher = Agent(
    role="资深研究员",
    goal="深入调研给定主题，收集全面、准确的信息",
    backstory="""你是一位经验丰富的研究员，擅长快速搜集和整理信息。
你总是能找到最有价值的洞察。""",
    verbose=True,
    llm=llm,
)

analyst = Agent(
    role="数据分析师",
    goal="分析调研结果，提炼关键洞察和趋势",
    backstory="""你是一位敏锐的数据分析师，善于从信息中发现模式和价值。
你的分析总是切中要害。""",
    verbose=True,
    llm=llm,
)

writer = Agent(
    role="技术撰稿人",
    goal="将调研和分析转化为通俗易懂的文章",
    backstory="""你是一位优秀的技术撰稿人，能把复杂的概念讲得简单有趣。
你的文章总是引人入胜。""",
    verbose=True,
    llm=llm,
)


# ========== 2. 定义任务 ==========


def create_crew(topic: str) -> Crew:
    """创建一个研究→分析→写作的工作流 Crew"""

    research_task = Task(
        description=f"""
        研究"{topic}"这个主题，包括：
        1. 基本概念和定义
        2. 发展历程
        3. 当前现状和主流方案
        4. 未来趋势

        请提供详细、准确的调研结果。
        """,
        agent=researcher,
        expected_output="一份详细的调研报告，包含概念、历程、现状、趋势",
    )

    analysis_task = Task(
        description=f"""
        基于研究员的调研结果，分析"{topic}"：
        1. 提炼 3 个关键洞察
        2. 分析优缺点
        3. 给出适用场景建议

        调研结果：{{previous_output}}
        """,
        agent=analyst,
        expected_output="3 个关键洞察 + 优缺点分析 + 适用场景",
        context=[research_task],  # 依赖研究任务
    )

    writing_task = Task(
        description=f"""
        基于调研和分析，写一篇关于"{topic}"的科普文章：
        1. 标题吸引人
        2. 开头引人入胜
        3. 正文结构清晰
        4. 结尾有启发
        5. 300-500字

        调研结果：{{research_output}}
        分析结果：{{analysis_output}}
        """,
        agent=writer,
        expected_output="一篇 300-500 字的科普文章",
        context=[research_task, analysis_task],
    )

    # 创建 Crew
    crew = Crew(
        agents=[researcher, analyst, writer],
        tasks=[research_task, analysis_task, writing_task],
        process=Process.sequential,  # 顺序执行
        verbose=True,
    )

    return crew


# ========== 3. 运行 ==========


def main():
    topics = [
        "AI Agent",
        "RAG 检索增强生成",
        "MCP 协议",
    ]

    for topic in topics:
        print(f"\n{'#'*60}")
        print(f"# CrewAI 协作：{topic}")
        print(f"{'#'*60}")

        crew = create_crew(topic)
        result = crew.kickoff()

        print(f"\n{'='*60}")
        print("📄 最终文章：")
        print(f"{'='*60}")
        print(result)


if __name__ == "__main__":
    main()
