"""LoRA 微调 Qwen2.5-0.5B 做情感分类"""

import os

# Hugging Face 国内镜像（必须在 import transformers/datasets 前设置）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import json  # noqa: E402
import torch  # noqa: E402
from datasets import Dataset  # noqa: E402
from transformers import (  # noqa: E402
    AutoTokenizer,
    AutoModelForCausalLM,
)
from peft import LoraConfig, get_peft_model, TaskType  # noqa: E402
from trl import SFTTrainer, SFTConfig  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from src.logger import get_logger  # noqa: E402

load_dotenv()

logger = get_logger("lora_train")


# ========== 配置 ==========

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DATA_PATH = "data/sft_sentiment.json"
OUTPUT_DIR = "data/lora_output"
MAX_SEQ_LENGTH = 256


# ========== 1. 加载数据 ==========


def load_sft_data(path: str) -> Dataset:
    """加载 SFT 数据"""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    # 转成对话格式
    formatted = []
    for item in data:
        text = f"指令：{item['instruction']}\n输入：{item['input']}\n输出：{item['output']}"
        formatted.append({"text": text})

    return Dataset.from_list(formatted)


# ========== 2. 加载模型 ==========


def load_model():
    """加载模型和 tokenizer"""
    logger.info("加载模型 %s", MODEL_NAME)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

    logger.info("模型参数量：%.1fM", model.num_parameters() / 1e6)
    return model, tokenizer


# ========== 3. 配置 LoRA ==========


def create_lora_config():
    """LoRA 配置"""
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,  # 低秩维度
        lora_alpha=16,  # 缩放因子（通常 = 2r）
        lora_dropout=0.05,  # dropout
        bias="none",  # 不训练 bias
        target_modules=[  # 应用到哪些层
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
    )


# ========== 4. 训练 ==========


def train(
    data_path: str = DATA_PATH,
    output_dir: str = OUTPUT_DIR,
    epochs: int = 3,
    max_samples: int | None = None,
):
    """微调主函数

    Args:
        data_path: 训练数据路径。
        output_dir: 模型保存目录。
        epochs: 训练轮数，默认 3。
        max_samples: 截取的数据条数（冒烟测试用），None 表示用全量。
    """
    # 1. 加载数据
    logger.info("加载训练数据：%s", data_path)
    dataset = load_sft_data(data_path)
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        logger.info("冒烟模式：截取 %d 条", len(dataset))
    logger.info("数据量：%d", len(dataset))

    # 2. 加载模型
    model, tokenizer = load_model()

    # 3. 配置 LoRA
    lora_config = create_lora_config()
    model = get_peft_model(model, lora_config)

    # 打印可训练参数（PEFT 内部输出，保留）
    model.print_trainable_parameters()

    # 4. 训练参数
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=1,  # 内存瘦身：4→1
        gradient_accumulation_steps=16,  # 等效 batch_size=16 不变
        learning_rate=2e-4,
        use_cpu=True,  # 纯 CPU 训练必须显式声明（transformers 5.x 校验）
        bf16=False,  # CPU 不用 bf16，强制 fp32
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=3,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        max_length=MAX_SEQ_LENGTH,  # trl 1.10+ 用 max_length（旧版叫 max_seq_length）
        dataset_text_field="text",
        report_to="none",  # 不上报到 wandb
    )

    # 5. 创建 Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # 6. 训练
    logger.info("开始训练（%d epoch）", epochs)
    if torch.cuda.is_available():
        logger.info("GPU: %s", torch.cuda.get_device_name())
        logger.info("显存: %.1fGB", torch.cuda.get_device_properties(0).total_mem / 1e9)

    trainer.train()

    # 7. 保存
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("模型已保存到 %s", output_dir)


def train_with_distilled_data():
    """用蒸馏数据微调（领域 SFT）"""
    DISTILLED_DATA_PATH = "data/distilled_dataset.json"
    DISTILLED_OUTPUT_DIR = "data/lora_distilled_output"
    logger.info("用蒸馏数据微调")
    train(data_path=DISTILLED_DATA_PATH, output_dir=DISTILLED_OUTPUT_DIR)


# ========== 5. 测试 ==========


def test_model():
    """测试微调后的模型"""
    print("\n🧪 测试微调效果...")

    # 加载 base 模型（不用 device_map="auto"，否则 CPU 上 meta 卸载会和 PEFT 加载 adapter 冲突）
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )

    # 加载 LoRA 权重
    from peft import PeftModel

    model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
    model = model.merge_and_unload()  # 合并 LoRA 权重

    # 测试用例
    test_cases = [
        ("这家餐厅太好吃了", "正面"),
        ("质量太差了", "负面"),
        ("今天气温25度", "中性"),
        ("服务态度很棒", "正面"),
        ("等了两个小时", "负面"),
    ]

    correct = 0
    for text, expected in test_cases:
        prompt = f"指令：判断情感\n输入：{text}\n输出："
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,  # 贪心解码，无需 temperature
            )

        response = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        ).strip()
        success = expected in response
        correct += success
        print(
            f"  {'✅' if success else '❌'} '{text}' → 期望:{expected} 实际:{response}"
        )

    print(f"\n📊 准确率：{correct}/{len(test_cases)} = {correct/len(test_cases):.0%}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_model()
    elif len(sys.argv) > 1 and sys.argv[1] == "distill":
        train_with_distilled_data()
    elif len(sys.argv) > 1 and sys.argv[1] == "smoke":
        # 冒烟测试：1 epoch + 5 条数据，验证流程与 logger 不跑完整训练
        train(
            epochs=1,
            max_samples=5,
            output_dir="data/lora_smoke_output",
        )
    else:
        train()
        test_model()
