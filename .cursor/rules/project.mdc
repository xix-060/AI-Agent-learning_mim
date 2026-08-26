---
description: AI Agent 学习项目规则
globs: ["src/**/*.py"]
alwaysApply: true
---

# 项目规则

## 技术栈
- Python 3.11，类型注解必写
- LLM 调用统一走 src/llm_client.py 的 LLMClient
- 测试用 pytest，放 tests/，文件名 test_*.py

## 代码规范
- 函数必须有 docstring（中文）
- 行宽 100
- 用 pathlib 不用 os.path
- 异常要捕获并给出有意义的错误信息

## 提交规范
- Conventional Commits: feat/fix/docs/refactor/test
- 中文描述

## 禁止
- 不要引入新的重量级依赖（先问我）
- 不要修改 .env
- 不要删除现有测试
