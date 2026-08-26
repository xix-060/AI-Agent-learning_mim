# SFT 数据格式

## 1. 指令格式（最常用）

```json
{
  "instruction": "把以下句子翻译成英文",
  "input": "今天天气真好",
  "output": "Today's weather is nice."
}
```

## 2.对话格式(Chat模式)

```JSON
{
  "messages": [
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！有什么可以帮你的？"}
  ]
}
```

## **3. 数据集要求**

- 数量：100-10000 条（任务越具体，需要越少）

* 质量：> 数量（100 条高质量 > 10000 条低质量）
* 多样性：指令表述要多样
* 长度：输入+输出控制在 512-2048 token

## **4. 数据来源**

- 公开数据集：Alpaca Belle Firefly
- 自己蒸馏：用 GPT-4/DeepSeek 生成（Day 3）
- 人工标注：质量最高但成本高
