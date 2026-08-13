"""HuggingFace 生态实战"""

import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

# 在 import transformers / datasets 前设置 HF 镜像（国内直连 hf.co 会超时）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch  # noqa: E402
from transformers import (  # noqa: E402
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
)
from datasets import load_dataset  # noqa: E402


HF_OFFLINE_MSG = "  (网络连接 HuggingFace 超时，跳过此 demo；若已缓存模型会自动走缓存)"


def _is_connect_error(e: Exception) -> bool:
    return "timeout" in str(type(e).__name__).lower() or "10060" in str(e)


# ========== 1. 加载模型 ==========


def demo_load_model():
    """加载小型模型"""
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    print(f"[LOAD] {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"  !! 加载失败：{type(e).__name__}: {e}")
        if _is_connect_error(e):
            print(HF_OFFLINE_MSG)
        return

    print(f"[OK] 模型参数量：{model.num_parameters() / 1e6:.1f}M")

    messages = [
        {"role": "system", "content": "你是助手"},
        {"role": "user", "content": "你好"},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100)

    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
    )
    print(f"[AI] {response}")


# ========== 2. Pipeline 快速推理 ==========


def demo_pipeline():
    """用 pipeline 做情感分析"""
    try:
        classifier = pipeline(
            "text-classification",
            model="distilbert-base-uncased-finetuned-sst-2-english",
        )
    except Exception as e:
        print(f"  !! pipeline 初始化失败：{type(e).__name__}: {e}")
        if _is_connect_error(e):
            print(HF_OFFLINE_MSG)
        return

    texts = [
        "I love this product!",
        "This is terrible.",
        "It's okay, nothing special.",
    ]

    for text in texts:
        result = classifier(text)
        print(f"  {text} -> {result}")


# ========== 3. Datasets 数据集 ==========


def demo_datasets():
    """加载和探索数据集"""
    try:
        ds = load_dataset("stanfordnlp/imdb", split="train[:100]")
    except Exception as e:
        print(f"  !! load_dataset 失败：{type(e).__name__}: {e}")
        if _is_connect_error(e):
            print(HF_OFFLINE_MSG)
        return None

    print(f"数据集大小：{len(ds)}")
    print(f"列名：{ds.column_names}")

    for i in range(3):
        item = ds[i]
        label = "正面" if item["label"] == 1 else "负面"
        print(f"\n[{i + 1}] {label}")
        print(f"    {item['text'][:100]}...")

    return ds


# ========== 4. Tokenizer 探索 ==========


def demo_tokenizer():
    """探索 Tokenizer"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True
        )
    except Exception as e:
        print(f"  !! Tokenizer 加载失败：{type(e).__name__}: {e}")
        if _is_connect_error(e):
            print(HF_OFFLINE_MSG)
        return

    texts = [
        "Hello, world!",
        "你好，世界！",
        "AI Agent 是未来的方向。",
    ]

    for text in texts:
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        print(f"\n  原文：{text}")
        print(f"  Tokens：{tokens}")
        print(f"  Token 数：{len(tokens)}")
        print(f"  解码：{decoded}")


# ========== 演示 ==========
def main():
    mirror = os.environ.get("HF_ENDPOINT")
    print(f"HuggingFace 镜像端点：{mirror}")

    print("=" * 60)
    print("1. 加载模型")
    print("=" * 60)
    demo_load_model()

    print("\n" + "=" * 60)
    print("2. Pipeline 推理")
    print("=" * 60)
    demo_pipeline()

    print("\n" + "=" * 60)
    print("3. Datasets")
    print("=" * 60)
    demo_datasets()

    print("\n" + "=" * 60)
    print("4. Tokenizer")
    print("=" * 60)
    demo_tokenizer()


if __name__ == "__main__":
    main()
