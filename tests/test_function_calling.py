"""Function Calling 测试"""

import pytest
from src.function_calling import FunctionCallingAgent


@pytest.fixture(scope="module")
def agent():
    return FunctionCallingAgent()


def test_calculate(agent):
    result = agent.run("帮我算 2 + 3 * 4")
    assert "14" in result


def test_time(agent):
    result = agent.run("现在几点")
    assert "202" in result  # 年份


def test_unit_convert(agent):
    result = agent.run("1 公斤等于多少克")
    assert "1000" in result


def test_no_tool(agent):
    result = agent.run("你好")
    assert len(result) > 0
