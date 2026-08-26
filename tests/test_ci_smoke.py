"""CI 冒烟测试：故意失败，用于演示 GitHub Actions 红灯→绿灯流程。

提交后会触发 CI 失败（红灯），修复后恢复绿色。
"""


def test_will_fail_demo_red_light():
    """故意失败的断言：演示 CI 红灯流程。"""
    assert 1 + 1 == 3, "演示用：故意写错等式触发 CI 失败"
