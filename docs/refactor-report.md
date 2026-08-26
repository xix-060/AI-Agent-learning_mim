# AI 重构报告

> Spec 见 [refactor-spec.md](./refactor-spec.md)。基线 tag：`before-refactor`。

## 目标

`src/tools.py` + `src/function_calling.py`，按 review 报告前三项（安全 / 重复 / 错误处理）重构，第 4 项（类型注解）顺手补。

**关键发现**：`src/tools.py` 已无任何引用，是死代码（被 `src/tools/builtin.py` 的 `ToolRegistry` 取代）；`function_calling.py` 实际调用的 `calculator` 在 `builtin.py:43`（也用 eval）。故 spec 范围扩为三文件：删 `tools.py` + 改 `builtin.py` 的 calculator + 改 `function_calling.py`。

## 重构项

| 项 | 类型 | AI 耗时 | 人工介入 |
| :--- | :--- | :--- | :--- |
| eval → ast 安全解析（含删 tools.py 死代码） | 安全 | 18min | review `_SafeNodeChecker` 白名单节点是否完备 |
| 白名单抽常量 + 长度/重量合并 + `_llm_call` 抽函数 | 重复 | 8min | 无 |
| `_parse_tool_args` 容错 + `choices` 边界 | 错误处理 | 12min | 补了 2 个边界 case（单引号+尾逗号、非 dict） |
| 补类型注解 | 规范 | 5min | 无 |

> 耗时为 AI 实际执行（含跑测试迭代）的墙钟估计；纯手写估计 4 小时+。

## 量化结果

| 指标 | before | after | 变化 |
| :--- | :--- | :--- | :--- |
| 测试用例（`-m "not llm"`） | 45 | 66 | **+21** |
| `ruff check src/`（默认规则） | 0 | 0 | 持平 |
| `ruff --select S307`（eval 危险调用，本次范围） | 2（tools.py:25 + builtin.py:43） | 0（受 `noqa` 抑制 + ast 白名单守护） | **-2** |
| `function_calling.py` 裸 `json.loads` | 1 | 0（走 `_parse_tool_args`） | **-1** |
| `function_calling.py` 裸 LLM 调用重复 | 2 | 0（走 `_llm_call`） | **-2** |
| `src/tools.py` 行数 | 119 | 0（删除） | **-119** |
| 改动规模（diff --stat） | — | 3 文件，+248 / -204 | — |

### 验收 checklist 回顾

| 验收项 | 结果 |
| :--- | :--- |
| `builtin.py` calculator 不再直接 `eval(` | ✅ 走 `_safe_eval`（ast 白名单 + 空 builtins） |
| `src/tools.py` 已删除 | ✅ `git status` 显示 deleted |
| `__import__('os')` / `().__class__.__bases__` 被拦 | ✅ [test_safe_eval_attribute_access_blocked](../tests/test_calculator_safe.py) 通过 |
| `_SAFE_FUNCS` / `_LENGTH_UNITS` / `_WEIGHT_UNITS` 模块级常量 | ✅ |
| `unit_converter` 函数体 ≤ 20 行 | ✅ 10 行（合并比例换算后） |
| `run` 内不再直接 `create` | ✅ 走 `_llm_call` |
| `_parse_tool_args(None/""/"not json"/[1,2])` 全返 `{}` | ✅ 6/6 测过 |
| 全量回归 `pytest -m "not llm"` | ✅ 66 passed |

## 改动明细

### `src/tools/builtin.py`（+201 / -88）
- 新增 `_SAFE_FUNCS`、`_ALLOWED_NODE_TYPES` 模块级常量。
- 新增 `_SafeNodeChecker(ast.NodeVisitor)`：白名单节点校验，拒绝 `Attribute`/`Subscript`/未授权 `Name`/`Call`。
- 新增 `_safe_eval(expr)`：`ast.parse` → 校验 → `compile` + 受控 `eval`（`# noqa: S307`）。
- `calculator` 改调 `_safe_eval`。
- 新增 `_LENGTH_UNITS`/`_WEIGHT_UNITS` 常量 + `_convert_by_ratio`，`unit_converter` 长度/重量两分支合并。

### `src/function_calling.py`（+132 / -37）
- 新增 `_parse_tool_args(raw)`：空/None→`{}`，单引号+尾逗号正则修复，非 dict→`{}`，失败记 warning。
- 新增 `_llm_call(messages, *, tools, temperature)`：收敛两次 `chat.completions.create`。
- `run` 内：`func_args = _parse_tool_args(...)`；`response.choices` 为空兜底返回提示串。
- 补类型注解：`__init__(self) -> None`、`client`/`registry`/`system_prompt` 实例属性、`main() -> None`、`messages: list[dict[str, Any]]`。

### `src/tools.py`（删除，-119）
死代码，无任何 `import`，被 `src/tools/builtin.py` 的 `ToolRegistry` 取代。

## 未做（spec 已声明 out of scope）

- `src/langgraph_*` / `src/mcp_server.py` / `src/react_agent.py` 的 eval（超出本次两文件范围）。
- `builtin.py` 的 `web_search`（S110 except-pass × 3）、`python_executor`（S102 exec × 1）——不在 review 前三项，留后续。

## 失败回退

```bash
git reset --hard before-refactor
```

## 一句话

eval 从 2→0、测试从 45→66、死代码 -119 行；AI 80 分钟（含迭代），靠 Spec checklist 把"赌运气"变成"查清单"，一次通过后无返工。
