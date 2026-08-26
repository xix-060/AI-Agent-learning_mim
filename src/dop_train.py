"""DPO 偏好对齐训练"""

import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import DPOTrainer, DPOConfig
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DATA_PATH = "data/dpo_dataset.json"
OUTPUT_DIR = "data/dpo_output"


def load_dpo_data(path: str) -> Dataset:
    """加载 DPO 数据"""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    # DPO 格式
    formatted = []
    for item in data:
        formatted.append(
            {
                "prompt": item["prompt"],
                "chosen": item["chosen"],
                "rejected": item["rejected"],
            }
        )

    return Dataset.from_list(formatted)


def train_dpo():
    """DPO 训练"""
    # 1. 加载数据
    print("📋 加载 DPO 数据...")
    dataset = load_dpo_data(DATA_PATH)
    print(f"  数据量：{len(dataset)}")

    # 2. 加载模型
    print(f"\n📦 加载模型 {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

    # 3. LoRA 配置（DPO 也可以用 LoRA）
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "v_proj"],
    )

    # 4. DPO 训练参数
    training_args = DPOConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,  # DPO 用更小的学习率
        beta=0.1,  # DPO 的温度参数
        logging_steps=2,
        save_strategy="epoch",
        max_length=512,
        max_prompt_length=256,
        report_to="none",
    )

    # 5. 创建 DPO Trainer
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    # 6. 训练
    print("\n🚀 开始 DPO 训练...")
    trainer.train()

    # 7. 保存
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\n✅ DPO 模型已保存到 {OUTPUT_DIR}")


def test_dpo():
    """测试 DPO 效果"""
    print("\n🧪 测试 DPO 模型...")

    from peft import PeftModel

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)

    questions = ["如何学习 AI？", "什么是 Agent？", "LoRA 是什么？"]

    for q in questions:
        messages = [{"role": "user", "content": q}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer([text], return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=200, temperature=0.7, do_sample=True
            )

        response = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
        print(f"\n❓ {q}")
        print(f"🤖 {response[:200]}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_dpo()
    else:
        train_dpo()
