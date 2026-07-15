# 第一行！必须放在所有代码、导入前面
from __future__ import annotations

# 【关键】镜像环境变量必须放在所有导入最前面
import os

# 强制覆盖镜像，不要用setdefault，避免不生效
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 开启下载日志，能看到卡住在哪一步
os.environ["HF_HUB_VERBOSITY"] = "debug"

import torch

# 新增hub下载工具，支持断点续传
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer


class LocalChatModel:
    """本地大模型对话封装"""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        local_model_dir="./qwen2.5-0.5b",
    ):
        print(f"📦 正在检查/下载模型 {model_name}...")

        # 自动选择设备
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"💻 使用设备: {self.device}")

        # 1. 先使用snapshot_download断点续传下载全部文件（解决无进度卡死）
        snapshot_download(
            repo_id=model_name,
            local_dir=local_model_dir,
            # local_dir_use_symlinks=False,
            # resume_download=True,  # 断点续传，断网下次继续
            force_download=False,
        )
        print("✅ 模型文件下载完成，开始加载...")

        # 2. 从本地文件夹加载，不再联网
        self.tokenizer = AutoTokenizer.from_pretrained(
            local_model_dir,
            trust_remote_code=True,
            local_files_only=True,  # 强制只读取本地，杜绝联网卡死
        )
        # CPU优化加载参数
        load_dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            local_model_dir,
            dtype=load_dtype,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
            low_cpu_mem_usage=True,  # CPU大幅降低内存占用，防止卡死
        )
        print(f"✅ 模型加载完成！参数量: {self.model.num_parameters() / 1e6:.1f}M")

    def chat(
        self,
        user_input: str,
        system_prompt: str = "你是一个有用的助手",
        history: list[dict] | None = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
    ) -> tuple[str, list[dict]]:
        messages = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_input})

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[
            0
        ]

        new_history = (history or []) + [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": response},
        ]

        return response, new_history


def main():
    print("🤖 本地大模型对话（输入 quit 退出）")
    print("=" * 50)

    model = LocalChatModel()
    history = []

    while True:
        user_input = input("\n🙋 我：")
        if user_input.strip().lower() in ("quit", "exit", "q"):
            break

        response, history = model.chat(user_input, history=history)
        print(f"\n🤖 AI：{response}")
        print(f" 📊 历史轮数: {len(history) // 2}")


if __name__ == "__main__":
    main()
