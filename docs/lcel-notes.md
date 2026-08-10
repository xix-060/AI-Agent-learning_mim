# LangChain LCEL（Expression Language）

## 1. 什么是 LCEL？

LangChain Expression Language，用 `|` 管道符把组件串成链。
类比 Unix 管道：`cat file | grep "error" | wc -l`

## 2. 核心接口：Runnable

所有组件都实现 Runnable 接口：

- invoke()：同步调用
- batch()：批量调用
- stream()：流式调用
- ainvoke() / abatch() / astream()：异步版

## 3. LCEL 四大原语

| 原语 | 作用 | 示例 |
| :--- | :--- | :--- |
| `\|` | 管道，前一个输出作为后一个输入 | `prompt \| llm \| parser` |
| `RunnablePassthrough` | 透传输入 | `{"context": retriever, "question": RunnablePassthrough()}` |
| `RunnableParallel` | 并行执行 | `RunnableParallel(a=..., b=...)` |
| `RunnableLambda` | 包装普通函数 | `RunnableLambda(lambda x: x.upper())` |

## 4. LCEL vs 旧版 Chain

```python
# 旧版（已废弃）
from langchain.chains import LLMChain
chain = LLMChain(llm=llm, prompt=prompt)

# LCEL（推荐）
chain = prompt | llm | parser
```
