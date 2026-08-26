---
description: Agent 开发规范
globs: ["**/agent*.py", "**/tools/**/*.py"]
---

# Agent 开发规范

## 模式
- Agent 循环必须设 max_steps 防死循环
- 工具返回值截断到 2000 字符内
- 工具描述要具体（给 LLM 看的，写清楚何时用）

## 记忆
- 短期记忆用消息列表，超限要压缩
- 长期记忆操作必须带 metadata

## 测试
- 涉及 LLM 调用的测试标记 @pytest.mark.llm
- 工具函数测试不打 LLM API
