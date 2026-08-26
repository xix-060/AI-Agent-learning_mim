"""CI 冒烟测试：验证 pytest 能正常收集并运行。

曾用于演示 GitHub Actions 红灯→绿灯流程（故意写错等式触发失败，现已修复）。
"""


def test_ci_smoke_passing():
    """冒烟测试：基础断言通过，确认 CI 环境正常。"""
    assert 1 + 1 == 2, "基础算术应成立"
