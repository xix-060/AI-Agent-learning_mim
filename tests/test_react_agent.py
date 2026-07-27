"""ReAct Agent 测试"""

import pytest
from src.llm_client import LLMClient
from src.react_agent import ReActAgent, Tool, calculator, fake_search, get_time


@pytest.fixture(scope="module")
def agent():
    llm = LLMClient()
    tools = [
        Tool(name="Search", description="搜索信息", func=fake_search),
        Tool(name="Calculator", description="数学计算", func=calculator),
        Tool(name="Time", description="获取时间", func=get_time),
    ]
    return ReActAgent(llm, tools, max_steps=5, verbose=False)


def test_search_question(agent):
    answer = agent.run("Python 是哪一年发明的？")
    assert "1991" in answer


def test_calculator(agent):
    answer = agent.run("计算 2 + 3 * 4 的结果")
    assert "14" in answer


def test_time(agent):
    answer = agent.run("现在几点了？")
    assert "202" in answer
