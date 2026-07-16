"""chat_local 的单元测试（需要先下载模型）"""

import pytest
from src.chat_local import LocalChatModel


@pytest.fixture(scope="module")
def model():
    """所有测试共享一个模型实例（避免重复加载）"""
    return LocalChatModel()


def test_single_turn(model):
    response, history = model.chat("1+1等于几？")
    assert isinstance(response, str)
    assert len(response) > 0
    assert "2" in response
    assert len(history) == 2  # user + assistant


def test_multi_turn(model):
    history = []
    _, history = model.chat("我叫小明", history=history)
    response, history = model.chat("我叫什么名字？", history=history)
    assert "小明" in response
    assert len(history) == 4
