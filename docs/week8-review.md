# 第 8 周复盘

📅 **周期**: Cursor 深度配置 → AI 辅助开发 Todo App → Spec 驱动实战 → 多智能体协作 → CI/CD 兜底 → AI 辅助重构沉淀
📊 **代码统计**: 10 个 Commit（08-26） | 45 文件改动 | +3,211 / -223 行 | 5 个新源文件 + 12 篇文档 + 6 份规则

***

## 1. 三大范式我都会了吗

本周把 AI 编程的三种范式各打了一遍，不再停留在"听说过"：

### 1️⃣ Vibe Coding（会用，但知道只适合 Demo）

用 AI 直接生成 [todo_app/](file:///e:/git/AI-Agent-learning_mim/todo_app)（Flask + SQLite + 标签功能），一轮就跑起来。体会到它的甜：快、零门槛。也吃到它的苦：第二轮加"标签"功能时 AI 自由发挥、边界乱窜——于是同一天切到 Spec 驱动重做了一遍（[todo_app/SPEC_TAGS.md](file:///e:/git/AI-Agent-learning_mim/todo_app/SPEC_TAGS.md)）。

**结论**：Vibe 适合原型/一次性脚本，**正经功能别用**。

### 2️⃣ Spec 驱动（主力工作流，本周的核心）

把 Day 3 的 Spec 流程真正落进项目：
- 写 [refactor-spec.md](file:///e:/git/AI-Agent-learning_mim/docs/refactor-spec.md)：背景基线 + 范围（做什么/不做什么）+ 4 项验收 checklist + 量化目标 + 失败回退。
- AI 按 checklist 分步实现，每项改完跑对应测试再进下一项。
- 量化验收写进 [refactor-report.md](file:///e:/git/AI-Agent-learning_mim/docs/refactor-report.md)：eval 2→0、测试 45→66、diff +248/-204。

**关键对比**（[spec-driven-development.md](file:///e:/git/AI-Agent-learning_mim/docs/spec-driven-development.md) 已沉淀）：Vibe 平均返工 3-8 次，Spec 流程 0-2 次；本次重构一次通过、零返工。

### 3️⃣ 多智能体协作（理解适用边界：自评不可靠，需 CI 兜底）

写了 [multi_agent_dev.py](file:///e:/git/AI-Agent-learning_mim/src/multi_agent_dev.py)：PM → Coder → Tester 三角分工， Tester 跑 pytest 拿 PASS/FAIL 回传。

**踩到的边界**：
- Tester 用 `"PASS" in review` 子串匹配 → "PASSED with errors" 也判过 → 必须精确匹配 `"结论：PASS"`。
- LLM 自评不可信（呼应 [cicd-notes.md](file:///e:/git/AI-Agent-learning_mim/docs/cicd-notes.md) 的结论）：多智能体的"验收"必须有客观裁判（CI/pytest），不能让 LLM 自己说"通过了"。

***

## 2. 我的工作流沉淀

本周把零散经验固化成可复用的资产，分布在三个层面：

### 规则层（.cursor/rules + .trae/rules 双份同步）

| 规则 | 作用 |
| :--- | :--- |
| [project.md](file:///e:/git/AI-Agent-learning_mim/.trae/rules/project.md) | 项目硬约束（Python 3.11、类型注解、pathlib、Conventional Commits） |
| [agent-dev.md](file:///e:/git/AI-Agent-learning_mim/.trae/rules/agent-dev.md) | Agent 开发规范（globs 匹配 `**/agent*.py`） |
| [ai-coding-playbook.md](file:///e:/git/AI-Agent-learning_mim/.trae/rules/ai-coding-playbook.md) | 作战手册：流程铁律 + 喂上下文套路 + 已知坑 + 验收标准 |

> Trae 用 `.md`、Cursor 用 `.mdc`，frontmatter 字段兼容，两份同步维护。

### 模板层

- [templates/SPEC_TEMPLATE.md](file:///e:/git/AI-Agent-learning_mim/templates/SPEC_TEMPLATE.md)：新功能先写 SPEC 的骨架。
- [refactor-spec.md](file:///e:/git/AI-Agent-learning_mim/docs/refactor-spec.md)：SPEC 的实战样例（可当二次模板）。

### 知识层

- [ai-coding-best-practices.md](file:///e:/git/AI-Agent-learning_mim/docs/ai-coding-best-practices.md)：提炼 Anthropic Claude Code 最佳实践（7 原则 + 3 反模式）。
- [ai-coding-workflow.md](file:///e:/git/AI-Agent-learning_mim/docs/ai-coding-workflow.md)：Explore→Plan→Code→Commit 工作流。
- [ai-faq.md](file:///e:/git/AI-Agent-learning_mim/docs/ai-faq.md)：4 个高频问答（不敢用/改坏了/额度用完/不该用）。
- [cursor-skills.md](file:///e:/git/AI-Agent-learning_mim/docs/cursor-skills.md)：IDE 技巧库。

***

## 3. 最大认知转变

> **AI 编程的瓶颈不是 AI 能力，而是我描述问题和验收的能力。**

本周三次实证：

1. **Todo App 加标签**：Vibe 模式 AI 乱改边界 → 改用 Spec 模式，把"做什么/不做什么/验收"写清 → 一次过。AI 还是那个 AI，差别在我。
2. **工具模块重构**：先 review 列 5 类问题，再写 spec 把"前三项"框死范围 + checklist → AI 80 分钟搞定，零返工。没有 spec，AI 会顺手改 langgraph/react_agent 的 eval，改动扩散、不可控。
3. **CI 红灯绿灯**：故意写必挂测试 push → 红灯 13s 出结果。本地绿、远端红（UTF-16 编码、缺依赖）的"环境差异"是头号杀手——**不信 AI 汇报"测试通过了"，必须自己跑 pytest，且要在干净环境跑**。

衍生认知：
- **小步 commit 是回滚的底气**：AI 改坏了 `git checkout -- <file>` 单文件回退，所以每完成一步就 commit（playbook 铁律 3）。
- **上下文会越拖越脏**：长会话后半段质量明显下降，40 分钟换新会话（playbook 已知坑 4）。
- **喂上下文要精准**：改哪个文件 @ 哪个文件，不整目录乱喂——否则 AI 抓不住重点。

***

## 4. 下周（部署与高并发）预习

### 🎯 预习清单

| 概念 | 是什么 | 我现在的理解（待验证） |
| :--- | :--- | :--- |
| vLLM | 高吞吐 LLM 推理引擎 | PagedAttention 减少显存碎片，提升并发 |
| SGLang | 推理引擎 + 编程框架 | 比 vLLM 更激进的结构化生成优化 |
| QPS | 每秒请求数 | 吞吐上限指标 |
| P99 延迟 | 99 分位响应时间 | 长尾用户体验指标，比平均值更诚实 |
| TTFT | 首 token 延迟 | 流式场景体感最关键的指标 |

### 💡 想搞清楚的问题

1. **vLLM 和 SGLang 是竞争还是互补？** 各自的杀手锏是什么？本周的 0.5B 小模型该用哪个？
2. **QPS / P99 怎么实测？** 用 locust 还是 wrk？压测时怎么构造真实 prompt 分布？
3. **本周的 CI 流水线能不能扩展到 CD？** 测试过了 → 自动部署到哪（本地 Docker？云函数？）
4. **RAG/微调后的模型部署有什么坑？** 第 7 周的 LoRA adapter 在推理时怎么加载才不拖慢 QPS？

### 🔗 前置知识（本周已具备）

| 下周需要 | 本周学的 | 关联 |
| :--- | :--- | :--- |
| 模型服务化 | Todo App 的 Flask API | 都是 HTTP 服务，部署套路可迁移 |
| 压测脚本 | CI 的 pytest 自动化 | 自动化思维一致 |
| 性能观测 | CI 红绿灯 + 耗时记录 | 指标思维可迁移 |
| LoRA adapter 加载 | 第 7 周 PeftModel | 推理路径已知 |

***

## 5. 本周代码统计

### 提交记录（10 个 Commit）

| Commit | 核心产出 | 类型 |
| :--- | :--- | :--- |
| f426990 | Cursor 深度配置（rules 三份 + 技巧库 + logger） | feat |
| 0a59a21 | AI 辅助开发 Todo 应用（Flask + SQLite） | feat |
| 88fb378 | Spec 驱动开发实战（标签功能）+ SPEC 模板 | feat |
| 984787b | AI 编程最佳实践 + 多智能体协作开发实验 | feat |
| 2a44fb8 | GitHub Actions 配置 + LLM 测试标记 | ci |
| 72624dd | 故意失败的冒烟测试（演示 CI 红灯） | test |
| c24a9c1 | requirements.txt 改 UTF-8 + 忽略 test_naive_rag | fix(ci) |
| 602509b | 修复 test_ci_smoke 冒烟测试（演示 CI 绿灯） | fix |
| 1a2584a | GitHub Actions 自动化流水线（lint + 测试） | ci |
| 643a466 | AI 辅助重构工具模块 + 沉淀编程手册 | refactor |

### 新增/改动文件树

```
src/
├── function_calling.py        # 重构：_parse_tool_args + _llm_call + 类型注解
├── tools/builtin.py           # 重构：_safe_eval(ast) + 常量抽取 + _convert_by_ratio
├── tools.py                   # 删除（死代码，0 引用）
├── logger.py                  # 新：统一日志工具
├── multi_agent_dev.py         # 新：PM→Coder→Tester 多智能体
└── lora_train.py              # 改：接入 logger
todo_app/                       # 新：Flask+SQLite Todo App
├── app.py, db.py, templates/
└── SPEC_TAGS.md               # Spec 实战样例
tests/                          # +3 测试文件，3 个加 llm 标记
├── test_calculator_safe.py    # 新：ast 安全求值 10 case
├── test_unit_converter.py     # 新：单位换算 5 case
├── test_parse_tool_args.py    # 新：JSON 容错 6 case
└── test_ci_smoke.py           # 新：CI 冒烟
docs/                           # 12 篇
├── ai-coding-best-practices.md
├── ai-coding-workflow.md
├── ai-faq.md
├── cicd-notes.md              # + img/ 4 张红绿灯截图
├── cursor-skills.md
├── refactor-spec.md / refactor-report.md
└── spec-driven-development.md  # 补：返工对比 + 一次通过率
.github/workflows/ci.yml        # 新：CI 流水线
.cursor/rules/ + .trae/rules/   # 各 3 份规则
templates/SPEC_TEMPLATE.md      # 新：SPEC 骨架
```

### 📈 代码量

```
本周新增约 3,211 行（10 个 commit，45 文件）
```

***

## 6. 本周踩坑沉淀（已进项目记忆）

| 坑 | 教训 |
| :--- | :--- |
| `ruff` / `gh` 不在 PATH | 用 `python -m ruff`；gh 改用浏览器截图 Actions 页面 |
| `requirements.txt` UTF-16 编码 | Linux pip 解析失败 → 改 UTF-8（CI 红灯 #1 根因） |
| `test_naive_rag` 依赖 numpy/pdfplumber 未进 requirements | CI 里 `--ignore`，或补依赖 |
| 工具名漂移 `calculate` vs `calculator` | tools.py 是死代码 → 删除对齐；prompt 与 registry 名必须一致 |
| `multi_agent_dev` PASS 子串匹配误判 | `"PASS" in review` → `"结论：PASS"` 精确匹配 |
| `eval({"__builtins__":{}})` 不能防沙箱逃逸 | `().__class__.__bases__` 可逃逸 → ast 白名单节点校验 |
| `isinstance(x, frozenset)` 报错 | 第二参数必须是 type / tuple of types，不能用 frozenset |
| LLM 自评"测试通过了"不可信 | 必须自己跑 pytest，CI 是客观裁判 |

***

## 📝 本周反思

### ✅ 做得好的地方

1. **范式闭环**：一周内 Vibe → Spec → 多智能体 → CI → 重构沉淀，不是单独练招式，而是串成一条工作流。
2. **Spec 驱动真用起来**：refactor-spec.md 是第一次拿 Spec 改自己的代码，量化验收（eval 2→0、测试 +21）让"AI 一次通过"从口号变成可验证的事实。
3. **踩坑即沉淀**：每个坑都进了 playbook/项目记忆/FAQ，下周不重蹈。
4. **双 IDE 规则同步**：Trae + Cursor 两份规则并行维护，换 IDE 不丢经验。

### ⚠️ 需要改进

1. **单日信息密度过高**：10 个 commit 全挤在 08-26 一天，复盘时回溯吃力。下周应分散到多日，每天留简短日志。
2. **量化指标偏代码侧**：本周量化了 eval/测试/lint，但没量化"Spec 写多长 → AI 一次通过率"。下周可记录 spec 字数 vs 返工次数。
3. **压测/部署完全空白**：下周进入部署与高并发，目前对 vLLM/SGLang 只有概念印象，需尽早动手跑一次。

### 🎯 下周期待

> 本周从"用 AI 写代码"推进到"管 AI 写代码"——把 Vibe 的随意换成 Spec 的可控，把 LLM 自评换成 CI 的客观。
>
> 下周进入部署与高并发，把模型从"能跑"推到"扛得住压"——vLLM/SGLang 怎么选、QPS/P99 怎么测、本周的 CI 能不能扩成 CD。

***

**文档生成**: 2026-08-26
**下周目标**: 部署与高并发（vLLM / SGLang / QPS / P99 / 压测）
