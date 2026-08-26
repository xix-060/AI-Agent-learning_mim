# 重构 SPEC：tools.py + function_calling.py

> 按 Day 3 Spec 驱动流程执行（见 [spec-driven-development.md](./spec-driven-development.md)）：人写规格 + 验收 checklist → AI 按规格分步实现 → 人按 checklist 验收。

## 1. 背景与基线

经 review（详见对话），`src/tools.py` 与 `src/function_calling.py` 存在 3 类问题：安全（eval）、代码重复、错误处理缺失。本 SPEC 重构前三项（安全、重复、错误处理），第 4 项（类型注解）顺手补。

### 基线数据（before-refactor tag）

| 指标 | 值 |
| :--- | :--- |
| `ruff check src/`（默认规则） | 0 问题（不抓 eval） |
| `ruff check --select S src/`（安全规则） | 17 个安全问题 |
| 其中 `S307`（eval） | 6 处（本次范围内 2 处） |
| 本次范围内 eval 调用 | `src/tools.py:25`、`src/tools/builtin.py:43` |
| `pytest tests/ -m "not llm"` | 45 通过 / 7 deselected |
| `src/tools.py` 引用数 | 0（死代码，TOOLS_SCHEMA/TOOL_MAP 已被 `src/tools/builtin.py` 的 ToolRegistry 取代） |

### 关键事实

- `function_calling.py` 实际调用的 `calculator` 在 [src/tools/builtin.py:24](../src/tools/builtin.py#L24)，**不是** [src/tools.py](../src/tools.py) 的 `calculate`。
- `src/tools.py` 已无任何 `import`，是旧残留；其 `eval`（[tools.py:25](../src/tools.py#L25)）是死代码，但为消除风险一并处理。
- 工具名漂移：`src/tools.py` 的 `calculate`（TOOLS_SCHEMA）vs `src/tools/builtin.py` 的 `calculator`（registry 注册名）vs `function_calling.py` system_prompt 的 `calculator`——三者不一致。

## 2. 范围

### 做什么（in scope）

1. **eval → ast 安全解析**：`src/tools/builtin.py` 的 `calculator` + 删除 `src/tools.py` 死代码。
2. **白名单抽常量 + 工具名漂移修复 + LLM 调用抽取**。
3. **JSON 解析容错 + 边界防御**：`function_calling.py` 的 `json.loads` / `choices[0]` / 参数解构。
4. 顺手补类型注解。

### 不做什么（out of scope，留作后续）

- 不重构 `src/langgraph_advanced`、`src/langgraph_agent`、`src/mcp_server`、`src/react_agent` 的 eval（超出本次两个文件范围，避免改动扩散）。
- 不重构 `src/tools/builtin.py` 的 `web_search`（S110 except-pass）、`exec`（S102，run_python 工具）——不在 review 报告前三项。
- 不动 `LLMClient` 内部实现。
- 不改 `.env`、不加新依赖（`ast` 是标准库）。

## 3. 重构项与验收 checklist

### 项 1：eval → ast 安全解析

**做什么**

- 在 `src/tools/builtin.py` 新增 `_safe_eval(expr: str) -> str`：用 `ast.parse(expr, mode="eval")` + `NodeVisitor` 白名单遍历，仅允许 `BinOp/UnaryOp/Call(Name in allowed)/Constant/Name(in allowed)/BoolOp/Compare`。访问 `__class__`/`__builtins__`/属性 → 抛 `ValueError`。
- `calculator` 改为调用 `_safe_eval`，`allowed` 常量提到模块级 `_SAFE_FUNCS`。
- 删除 `src/tools.py`（死代码）。
- `src/langgraph_agent/basic_graph.py:34`、`src/react_agent.py:275` 等**不在范围**，但 spec 里登记为后续项。

**验收 checklist**

- [ ] `src/tools/builtin.py` 不再出现 `eval(` / `exec(`。
- [ ] `src/tools.py` 文件已删除，`git status` 显示 deleted。
- [ ] `_safe_eval("2+3*4")` == `"2+3*4 = 14"`。
- [ ] `_safe_eval("sin(pi/2)")` == `"sin(pi/2) = 1.0"`。
- [ ] `_safe_eval("__import__('os')")` 抛异常被捕获，返回含"计算错误"的字符串（不抛到调用方）。
- [ ] `_safe_eval("().__class__.__bases__")` 同上被拦。
- [ ] 新增 `tests/test_calculator_safe.py` 覆盖以上 5 个 case + 正常表达式 3 个。

### 项 2：白名单抽常量 + 工具名漂移 + LLM 调用抽取

**做什么**

- 把 `calculator` 的 `allowed` dict 提为模块级常量 `_SAFE_FUNCS: dict[str, Any]`（含 sin/cos/tan/sqrt/log/log10/pi/e/abs/round/min/max/sum）。
- 把 `unit_converter` 的长度/重量换算表提为模块级常量 `_LENGTH_UNITS`、`_WEIGHT_UNITS`，长度/重量两个同构 `if` 合并为一个 `_convert_by_ratio(value, from_unit, to_unit, table)`。
- `function_calling.py`：抽 `_llm_call(messages, *, tools=None, temperature=0.0)` 收敛两次 `self.client.client.chat.completions.create` 重复。
- 工具名漂移：删除 `src/tools.py` 后，`calculate` 残留消失；`function_calling.py` 的 system_prompt 已经是 `calculator`，与 registry 一致——无需改 prompt，确认即可。

**验收 checklist**

- [ ] `src/tools/builtin.py` 顶部新增 `_SAFE_FUNCS`、`_LENGTH_UNITS`、`_WEIGHT_UNITS` 三个模块级常量。
- [ ] `unit_converter` 函数体 ≤ 20 行（合并分支后）。
- [ ] `function_calling.py` 的 `run` 内不再直接出现 `self.client.client.chat.completions.create`，统一走 `_llm_call`。
- [ ] 新增 `tests/test_unit_converter.py` 覆盖 m→km、kg→lb、C→F、不支持换算 4 个 case。
- [ ] 现有 `tests/test_function_calling.py`（llm 标记）不变仍可收集。

### 项 3：JSON 解析容错 + 边界防御

**做什么**

- 在 `function_calling.py` 新增 `_parse_tool_args(raw: str | None) -> dict`：
  - `raw` 为空 → 返回 `{}`。
  - `json.loads` 失败 → 尝试用正则修复常见错误（单引号→双引号、尾逗号），仍失败则返回 `{}` 并记日志。
  - 返回值若不是 dict → 返回 `{}`。
- `run` 内 `func_args = _parse_tool_args(tool_call.function.arguments)`。
- `response.choices` 为空 → 返回 `( Assistant "API 返回空，请重试"` 兜底，不 `IndexError`。
- `tool_calls` 为 None 时跳过工具循环（现有逻辑已 OK，补注释确认）。

**验收 checklist**

- [ ] `function_calling.py` 不再出现裸 `json.loads(tool_call.function.arguments)`。
- [ ] `_parse_tool_args(None)` == `{}`。
- [ ] `_parse_tool_args("")` == `{}`。
- [ ] `_parse_tool_args("{'a': 1,}")`（单引号+尾逗号）== `{"a": 1}`。
- [ ] `_parse_tool_args("not json")` == `{}` 且记一条 warning 日志。
- [ ] `_parse_tool_args("[1,2]")`（非 dict）== `{}`。
- [ ] 新增 `tests/test_parse_tool_args.py` 覆盖以上 6 个 case。

### 项 4：补类型注解（顺手）

- `function_calling.py`：`__init__(self) -> None`、`main() -> None`、实例属性 `client: LLMClient`/`registry: ToolRegistry`/`system_prompt: str`、`messages: list[dict[str, Any]]`、`run(...) -> str` 已有。
- `src/tools/builtin.py`：`_SAFE_FUNCS: dict[str, Any]`、`_LENGTH_UNITS: dict[str, float]`、`_WEIGHT_UNITS: dict[str, float]`、`_safe_eval(expr: str) -> str`、`_convert_by_ratio(...) -> str`、`ToolRegistry.execute(...) -> str`（已有）、`ToolRegistry.get_openai_tools_schema() -> list[dict[str, Any]]`。

**验收 checklist**

- [ ] `ruff check src/tools/builtin.py src/function_calling.py` 无新增告警。
- [ ] `python -m mypy --strict`（若可用）或人工抽查：上述符号均有注解。

## 4. 量化验收（写进 refactor-report.md）

| 指标 | before | after（目标） |
| :--- | :--- | :--- |
| 本次范围 eval 调用 | 2 | 0 |
| `ruff --select S` 范围内 | 2（tools.py + builtin.py） | 0（tools.py 删，builtin.py 改 ast） |
| 测试用例（`-m "not llm"`） | 45 | ≥ 45 + 13（新增 3 个测试文件） |
| `src/tools.py` | 119 行死代码 | 删除 |
| `function_calling.py` LLM 调用重复 | 2 处裸调用 | 0（走 `_llm_call`） |

## 5. 执行顺序

1. 项 1（eval → ast + 删 tools.py）→ 跑 `pytest tests/test_calculator_safe.py`。
2. 项 2（常量抽取 + `_llm_call`）→ 跑 `pytest tests/test_unit_converter.py`。
3. 项 3（`_parse_tool_args`）→ 跑 `pytest tests/test_parse_tool_args.py`。
4. 项 4（类型注解）→ 跑 `ruff check src/`。
5. 全量回归：`pytest tests/ -m "not llm" --ignore=tests/test_chat_local.py --ignore=tests/test_naive_rag.py`。
6. 写 [refactor-report.md](./refactor-report.md)，填实际 after 值。

## 6. 失败回退

- 任一项验收 checklist 不过 → 当项打回重做，不进入下一项。
- 回退命令：`git reset --hard before-refactor`。
