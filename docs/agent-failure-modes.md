# Agent 常见失败模式

## 1. 工具选择错误

症状：该用 Calculator 却用了 Search
原因：Prompt 中工具描述不清晰
解决：优化工具 description，加 Few-shot 示例

## 2. 参数格式错误

症状：Action Input 不是有效 JSON
原因：LLM 没按格式输出
解决：强化 Prompt + 容错解析

## 3. 循环调用

症状：反复调用同一工具，不收敛
原因：LLM 忘记之前的结果（短期记忆丢失）
解决：把观察结果加入 Prompt + 限制最大步数

## 4. 幻觉

症状：不调用工具，直接编造答案
原因：LLM 太自信
解决：Prompt 强制"必须调用工具验证"

## 5. Token 爆炸

症状：步骤太多，Prompt 超长
原因：每步都把全部历史拼进 Prompt
解决：摘要压缩 + 限制步数

## 优化建议

1. 优化 system\_prompt
2. 加 Few-shot 示例
3. 优化工具描述
4. 加容错解析
5. 限制单工具返回长度
