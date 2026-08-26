# 模型量化

## 1. 什么是量化？

把高精度浮点数（FP32/FP16）转为低精度（INT8/INT4）。
减少显存占用，轻微损失精度。

## 2. 量化对比

| 精度   | 7B 模型显存 | 精度损失 |
| :--- | :------ | :--- |
| FP32 | 28GB    | 0%   |
| FP16 | 14GB    | \~0% |
| INT8 | 7GB     | \~1% |
| INT4 | 3.5GB   | \~3% |

## 3. QLoRA 的三个创新

1. 4-bit NormalFloat：新的量化数据类型，比 INT4 更准
2. Double Quantization：量化常数也量化，再省 0.5GB
3. Paged Optimizer：用 CPU 内存兜底 GPU 显存峰值

## 4. bitsandbytes 库

HuggingFace 的量化库，配合 PEFT 使用：

```python
from transformers import BitsAndBytesConfig
config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
model = AutoModelForCausalLM.from_pretrained(..., quantization_config=config)
```

## **5. QLoRA 微调流程**

1. 加载 4bit 量化的 base 模型
2. 添加 LoRA 适配器（FP16 精度训练）
3. 只训练 LoRA 参数
4. 推理时可合并或保持量化
