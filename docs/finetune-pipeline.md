# LLM 微调全流程

## 1. 确定任务

- 什么任务？（分类/生成/对话）
- 有没有数据？
- 需要什么效果？

## 2. 数据准备

- 收集/蒸馏/标注
- 格式化（instruction/input/output 或 messages）
- 清洗（去重/过滤/长度控制）
- 划分（训练/验证/测试）

## 3. 选择方法

| 数据量        | 显存      | 推荐方法                |
| :--------- | :------ | :------------------ |
| < 1000 条   | < 8GB   | LoRA + 0.5B/1.5B 模型 |
| 1000-10000 | 8-16GB  | LoRA + 7B 模型        |
| > 10000    | 16-24GB | QLoRA + 7B/14B      |
| > 50000    | 24GB+   | 全参数微调               |

## 4. 训练

- 加载模型 + Tokenizer
- 配置 LoRA（r/alpha/target\_modules）
- 配置训练参数（lr/batch/epoch）
- 用 SFTTrainer 训练
- 监控 loss 和显存

## 5. 评估

- 自动指标（BLEU/ROUGE/覆盖率）
- 人工评估（抽查）
- 对比 base vs 微调

## 6. 部署

- 合并 LoRA 权重
- 量化（4bit/8bit）
- 用 vLLM/SGLang 部署
- API 服务化（FastAPI）

## 7. 迭代

- 分析 bad case
- 补充数据
- 调整参数
- 重新训练
