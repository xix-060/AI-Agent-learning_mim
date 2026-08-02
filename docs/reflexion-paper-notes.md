# Reflexion 论文精读笔记

**论文**: Reflexion: Language Agents with Verbal Reinforcement Learning
**作者**: Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, Shunyu Yao
**发表**: NeurIPS 2023 (Oral)
**arXiv**: https://arxiv.org/abs/2303.11366

---

## 1. 核心思想：自我反思 + 语言强化

### 问题背景
- 现有 Agent（如 ReAct）即使失败也重复相同策略
- 传统强化学习需要大量训练数据和梯度更新
- 人类能从错误中学习（"下次应该先查资料再写代码"）

### Reflexion 的核心创新
**用自然语言（而非标量分数）作为强化信号**

1. **自我反思 (Self-Reflection)**: Agent 失败后，用语言描述"哪里错了"、"下次怎么做"
2. **情节记忆 (Episodic Memory)**: 将反思存入短期记忆，下次尝试时读取
3. **无需微调**: 只靠 Prompt 传递反思，不改变模型权重

---

## 2. 和 ReAct 的关系

### ReAct 的局限
```
ReAct: Thought → Action → Observation → ... → Final Answer
```
- 失败后不会改进策略
- 可能陷入重复错误的循环

### Reflexion = ReAct + 反思循环
```
Reflexion:
  Trial 1: ReAct (失败) → Self-Reflection → 存入记忆
  Trial 2: ReAct (读取记忆) → (成功?) → 继续反思...
```

### 三组件架构

| 组件 | 功能 | 实现 |
|------|------|------|
| **Actor** | 执行 ReAct 循环 | 读取反思记忆 + 执行任务 |
| **Evaluator** | 判断成功/失败 | 环境反馈或 LLM 自评 |
| **Self-Reflection** | 生成反思文本 | LLM 分析失败原因 |

---

## 3. 评测结果

### 三大任务性能提升

| 任务 | 指标 | ReAct/CoT | Reflexion | 提升 |
|------|------|-----------|-----------|------|
| **HumanEval** (代码生成) | pass@1 | 80% | **91%** | +11% |
| **HotpotQA** (多跳推理) | 精确匹配 | ~35% | **51%** | +16% |
| **AlfWorld** (决策任务) | 完成率 | 75% | **97%** | +22% |

### 关键发现
1. **迭代次数**: 通常 3-5 次尝试后收敛
2. **记忆窗口**: 只保留最近 3-5 条反思，避免上下文过长
3. **通用 LLM 即可**: GPT-3.5-turbo 就能工作，无需特殊模型

---

## 4. 技术细节

### 4.1 语言强化的形式化
```
r_t = LLM反思(s_t, a_t, o_t)  # 第 t 步的反思文本

M = [r_1, r_2, ..., r_k]      # 记忆（滑动窗口）

new_prompt = original_prompt + "\n\n" + "\n".join(M) + "\n\n当前任务..."
```

### 4.2 反思触发条件
- **幻觉检测**: LLM 输出了不存在的信息
- **重复动作**: 连续执行相同的 Action
- **循环检测**: 在同一个问题上迭代超过 N 次

### 4.3 示例（HumanEval）
```python
# 第一次尝试（失败）
Thought: 需要计算两点距离
Action: 生成代码（缺少 sqrt）
Observation: 测试失败

# 反思
Reflection: "缺少导入 math 模块，且没有对坐标差开平方"

# 第二次尝试（成功）
Thought: 这次导入 math，使用 math.sqrt
Action: 生成正确代码
Observation: 所有测试通过
```

---

## 5. 实现要点

```python
class ReflexionAgent:
    def __init__(self, llm, tools, max_trials=5):
        self.memory = []      # 反思记忆
        self.max_trials = max_trials

    def run(self, task):
        for trial in range(self.max_trials):
            # 1. 执行 ReAct（读取记忆）
            result = self.react_loop(task, self.memory)

            # 2. 评估结果
            if self.evaluate(result):
                return result  # 成功！

            # 3. 生成反思
            reflection = self.reflect(result)

            # 4. 更新记忆（滑动窗口）
            self.memory.append(reflection)
            if len(self.memory) > 5:
                self.memory = self.memory[-5:]

        return result  # 返回最后一次结果

    def react_loop(self, task, memory):
        prompt = self.build_prompt(task, memory)
        # ... 标准 ReAct 循环 ...

    def reflect(self, result):
        """让 LLM 分析失败原因"""
        prompt = f"""
        任务: {result['task']}
        执行轨迹: {result['trajectory']}
        失败原因: {result['error']}

        请反思: 哪里做错了？下次应该怎么做？
        """
        return self.llm.generate(prompt)
```

---

## 6. 局限性与改进

### 已知问题
1. **单模型偏见**: 同一个 LLM 生成 Action 和 Reflection，可能维持偏见
2. **反思质量**: 低质量反思反而有害
3. **成本**: 多次尝试增加 API 调用

### 改进方向
- **多 Agent 反思**: 用不同角色（审查员、专家）产生多角度反思
- **结构化反思**: 预定义反思模板
- **早停机制**: 检测到无改进时停止

---

## 7. 与本项目的关系

### 可借鉴点
- 为我们的 `agent_v1.py` 添加反思机制
- 在 `agent_eval.py` 中测试多次尝试的效果

### 简化实现建议
```python
# 在 CompleteAgent 中添加
def run_with_reflection(self, question, max_trials=3):
    memory = []
    for trial in range(max_trials):
        result = self.run(question, extra_context=memory)
        if self.is_successful(result):
            return result
        # 生成反思并添加到记忆
        reflection = self.llm.generate_reflection(question, result)
        memory.append(reflection)
    return result
```
