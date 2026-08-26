"""QLoRA 4bit 量化微调 7B 模型"""

import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from trl import SFTTrainer, SFTConfig
from dotenv import load_dotenv

load_dotenv()


# ========== 配置 ==========

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"  # 7B 模型
DATA_PATH = "data/distilled_dataset.json"
OUTPUT_DIR = "data/qlora_output"
MAX_SEQ_LENGTH = 512


# ========== 1. 4bit 量化配置 ==========


def get_quantization_config():
    """QLoRA 量化配置"""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",  # NormalFloat 4-bit
        bnb_4bit_compute_dtype=torch.float16,  # 计算精度
        bnb_4bit_use_double_quant=True,  # 双重量化
    )


# ========== 2. 加载量化模型 ==========


def load_quantized_model():
    """加载 4bit 量化模型"""
    print(f"📦 加载 4bit 量化模型 {MODEL_NAME}...")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 量化配置
    bnb_config = get_quantization_config()

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # 准备 kbit 训练
    model = prepare_model_for_kbit_training(model)

    # 打印显存
    if torch.cuda.is_available():
        mem = torch.cuda.memory_allocated() / 1e9
        print(f"✅ 模型已加载，显存占用：{mem:.1f}GB")

    return model, tokenizer


# ========== 3. LoRA 配置 ==========


def create_qlora_config():
    """QLoRA 的 LoRA 配置"""
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,  # 7B 模型可以用更大的 r
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",  # 7B 模型可以应用到所有线性层
        ],
    )


# ========== 4. 显存监控 ==========


def print_gpu_memory():
    """打印 GPU 显存"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"  GPU 显存 - 已分配: {allocated:.2f}GB, 已保留: {reserved:.2f}GB")


# ========== 5. 训练 ==========


def train():
    """QLoRA 训练"""
    # 1. 加载数据
    print("📋 加载数据...")
    with open(DATA_PATH, encoding="utf-8") as f:
        data = json.load(f)

    formatted = []
    for item in data:
        text = f"指令：{item['instruction']}\n输入：{item.get('input', '')}\n输出：{item['output']}"
        formatted.append({"text": text})

    dataset = Dataset.from_list(formatted)
    print(f"  数据量：{len(dataset)}")

    # 2. 加载模型
    model, tokenizer = load_quantized_model()

    # 3. LoRA
    lora_config = create_qlora_config()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 4. 训练参数（针对低显存优化）
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=2,
        per_device_train_batch_size=1,  # 小 batch
        gradient_accumulation_steps=8,  # 梯度累积，等效 batch=8
        learning_rate=1e-4,
        logging_steps=3,
        save_strategy="epoch",
        save_total_limit=2,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
        fp16=True,  # 混合精度
        optim="paged_adamw_8bit",  # 分页优化器（省显存）
        report_to="none",
    )

    # 5. 训练
    print("\n🚀 开始 QLoRA 训练...")
    print_gpu_memory()

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # 训练前显存
    print("\n训练前：")
    print_gpu_memory()

    trainer.train()

    # 训练后显存
    print("\n训练后：")
    print_gpu_memory()

    # 6. 保存
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\n✅ QLoRA 模型已保存到 {OUTPUT_DIR}")


# ========== 6. 推理测试 ==========


def inference():
    """推理测试"""
    from peft import PeftModel

    print("\n🧪 测试 QLoRA 模型...")

    # 加载 base 模型（4bit）
    bnb_config = get_quantization_config()
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # 加载 LoRA
    model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)

    # 测试
    questions = [
        "什么是 AI Agent？",
        "RAG 如何减少幻觉？",
        "ReAct 框架是什么？",
    ]

    for q in questions:
        messages = [{"role": "user", "content": q}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer([text], return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
            )

        response = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
        print(f"\n❓ {q}")
        print(f"🤖 {response[:200]}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        inference()
    else:
        train()
