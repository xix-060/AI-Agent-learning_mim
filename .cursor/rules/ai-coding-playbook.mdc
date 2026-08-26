---
description: 我的 AI 编程作战手册（踩坑经验沉淀）
alwaysApply: false
globs: []
---

# AI 编程作战手册

## 流程铁律
1. 新功能 = 先 SPEC 后代码（模板见 templates/SPEC_TEMPLATE.md）
2. 一次只让 AI 做一件事，验收通过再下一件
3. 每完成一步就 git commit，方便回滚

## 喂上下文的套路
- 改哪个文件就 @哪个文件，不整目录乱喂
- 涉及项目约定时 @ .cursor/rules/project.mdc
- UI 问题贴截图，不写字描述

## 已知坑（真实踩过）
1. 让 AI 一次性生成 5 个文件 → import 全乱 → 改成逐文件生成
2. AI 修 Bug 时只贴报错不给复现步骤 → 它瞎猜 → 必须给复现步骤
3. AI 说"测试通过了" → 实际没跑 → 必须自己跑 pytest
4. 长会话后半段质量明显下降 → 40 分钟换新会话

## 验收标准
- ruff 过
- pytest 过（我自己跑，不信 AI 汇报）
- 我能向别人讲清这段代码每一行
