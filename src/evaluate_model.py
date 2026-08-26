"""评估微调效果"""

import os

# Hugging Face 国内镜像（必须在 import transformers/datasets 前设置）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import json  # noqa: E402
import torch  # noqa: E402
from transformers import AutoTokenizer, AutoModelForCausalLM  # noqa: E402
from peft import PeftModel  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv()

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
LORA_PATH = "data/lora_output"  # 或 qlora_output


def load_base_model():
    """加载 base 模型"""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    # 不用 device_map="auto"，否则 CPU 上 meta 卸载会和 PEFT 加载 adapter 冲突
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )
    return model, tokenizer


def load_finetuned_model():
    """加载微调后模型"""
    base, tokenizer = load_base_model()
    model = PeftModel.from_pretrained(base, LORA_PATH)
    model = model.merge_and_unload()
    return model, tokenizer


def generate_response(model, tokenizer, question, max_tokens=200):
    """生成回答"""
    messages = [{"role": "user", "content": question}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
    )


# ========== 评估指标 ==========


def calculate_bleu(reference: str, candidate: str) -> float:
    """简化版 BLEU 分数"""
    ref_words = set(reference.split())
    cand_words = set(candidate.split())

    if not cand_words:
        return 0.0

    overlap = len(ref_words & cand_words)
    return overlap / len(cand_words)


def calculate_rouge_l(reference: str, candidate: str) -> float:
    """简化版 ROUGE-L（最长公共子序列）"""
    ref_words = reference.split()
    cand_words = candidate.split()

    if not ref_words or not cand_words:
        return 0.0

    # LCS
    m, n = len(ref_words), len(cand_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_words[i - 1] == cand_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs = dp[m][n]
    recall = lcs / m
    precision = lcs / n
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )
    return f1


def keyword_coverage(response: str, keywords: list[str]) -> float:
    """关键词覆盖率"""
    if not keywords:
        return 0.0
    covered = sum(1 for kw in keywords if kw in response)
    return covered / len(keywords)


# ========== 评估 ==========


def evaluate():
    """对比 base vs 微调"""
    # 评估集
    eval_cases = [
        {
            "question": "判断情感：这家餐厅太好吃了",
            "keywords": ["正面"],
        },
        {
            "question": "判断情感：质量太差了",
            "keywords": ["负面"],
        },
        {
            "question": "判断情感：今天气温25度",
            "keywords": ["中性"],
        },
        {
            "question": "什么是 AI Agent？",
            "keywords": ["感知", "决策", "行动", "LLM", "工具"],
        },
        {
            "question": "LoRA 是什么？",
            "keywords": ["低秩", "微调", "参数", "高效"],
        },
    ]

    print("📊 加载 base 模型...")
    base_model, tokenizer = load_base_model()

    print("📊 加载微调模型...")
    ft_model, _ = load_finetuned_model()

    results = []
    for case in eval_cases:
        print(f"\n❓ {case['question']}")

        # Base 回答
        base_resp = generate_response(base_model, tokenizer, case["question"])
        base_coverage = keyword_coverage(base_resp, case["keywords"])

        # 微调回答
        ft_resp = generate_response(ft_model, tokenizer, case["question"])
        ft_coverage = keyword_coverage(ft_resp, case["keywords"])

        print(f"  Base:    [{base_coverage:.0%}] {base_resp[:80]}...")
        print(f"  微调:    [{ft_coverage:.0%}] {ft_resp[:80]}...")

        results.append(
            {
                "question": case["question"],
                "base_coverage": base_coverage,
                "ft_coverage": ft_coverage,
                "improvement": ft_coverage - base_coverage,
            }
        )

    # 汇总
    print(f"\n{'='*60}")
    print("📊 评估总结")
    print(f"{'='*60}")

    avg_base = sum(r["base_coverage"] for r in results) / len(results)
    avg_ft = sum(r["ft_coverage"] for r in results) / len(results)

    print(f"  Base 模型关键词覆盖率：{avg_base:.1%}")
    print(f"  微调模型关键词覆盖率：{avg_ft:.1%}")
    print(f"  提升：{avg_ft - avg_base:+.1%}")

    # 保存
    with open("docs/finetune-eval-report.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "avg_base_coverage": avg_base,
                "avg_ft_coverage": avg_ft,
                "improvement": avg_ft - avg_base,
                "details": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    if avg_ft > avg_base:
        print(f"\n✅ 微调有效！覆盖率提升 {avg_ft - avg_base:+.1%}")
    else:
        print("\n⚠️ 微调效果不明显，可能需要更多数据或调参")


if __name__ == "__main__":
    evaluate()
