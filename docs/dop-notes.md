# DPO（Direct Preference Optimization）

## 1. 解决什么问题？

SFT 后的模型可能生成"有害"或"低质量"的回答。
需要用人类偏好来"对齐"模型。

## 2. RLHF vs DPO

### RLHF（传统方法）

流程：SFT → 训练奖励模型 → PPO 强化学习

缺点：复杂、不稳定、需要两个模型

### DPO（2023 年新方法）

流程：SFT → 直接用偏好数据训练

优点：简单、稳定、不需要奖励模型

核心：把偏好优化转化为分类问题

## 3. DPO 数据格式

```json
{
  "prompt": "如何学习编程？",
  "chosen": "建议从 Python 开始，因为它语法简洁...（好的回答）",
  "rejected": "编程很难，放弃吧。（差的回答）"
}
```

## **4. DPO 原理（简化）**

给定一个 prompt，模型有两个回答：

- chosen（人类偏好的）
- rejected（人类不偏好的）

DPO 的目标：让模型对 chosen 的概率 > rejected 的概率。\
用 Bradley-Terry 模型把偏好转化为概率。

## **5. 何时用 DPO？**

- SFT 后想进一步提升质量
- 有偏好数据（A/B 测试、人工标注）
- 不想搞复杂的 RLHF

## **6. TRL 库的 DPO 实现**

```Python
from trl import DPOTrainer, DPOConfig

trainer = DPOTrainer(
    model=model,
    args=DPOConfig(...),
    train_dataset=preference_dataset,
    processing_class=tokenizer,
)
trainer.train()
```
