"""Temperature / TopP / TopK 参数实验"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm_client import LLMClient


def run_experiment(
    client: LLMClient,
    prompt: str,
    param_name: str,
    param_values: list,
    num_runs: int = 10,
    **kwargs,
) -> dict:
    """
    对同一 prompt 用不同参数值运行多次，记录结果

    参数:
        param_name: "temperature" / "top_p"
        param_values: [0.0, 0.5, 1.0, 1.5]
        num_runs: 每个参数值运行几次
    """
    results = {}
    from src.models import Message, RoleEnum

    for value in param_values:
        responses = []
        for i in range(num_runs):
            # 构造请求参数
            params = {"temperature": 0.7, "top_p": 1.0, "max_tokens": 100}
            params[param_name] = value
            params.update(kwargs)

            chat_kwargs = {
                "messages": [Message(role=RoleEnum.USER, content=prompt)],
                "temperature": params["temperature"],
                "max_tokens": params["max_tokens"],
            }
            if "top_p" in params:
                chat_kwargs["top_p"] = params["top_p"]
            response = client.chat(**chat_kwargs)
            responses.append(
                {
                    "run": i + 1,
                    "content": response.content,
                    "tokens": response.usage["total_tokens"],
                    "time": response.elapsed_seconds,
                }
            )

        # 统计唯一回复数（衡量多样性）
        unique_contents = set(r["content"] for r in responses)
        results[str(value)] = {
            "param_value": value,
            "num_runs": num_runs,
            "num_unique": len(unique_contents),
            "diversity_ratio": len(unique_contents) / num_runs,
            "responses": responses,
        }
        print(
            f" {param_name}={value}: {len(unique_contents)}/{num_runs} 唯一回复 (多样性={len(unique_contents)/num_runs:.1%})"
        )

    return results


def main():
    client = LLMClient()

    # 实验 1：Temperature 实验
    print("📊 实验 1:Temperature 对生成多样性的影响")
    print("=" * 60)
    prompt = "用一句话描述秋天"
    temp_results = run_experiment(
        client, prompt, "temperature", [0.0, 0.3, 0.7, 1.0, 1.5], num_runs=10
    )

    # 实验 2：Top-P 实验
    print("\n📊 实验 2:Top-P 对生成多样性的影响")
    print("=" * 60)
    topp_results = run_experiment(
        client, prompt, "top_p", [0.1, 0.5, 0.9, 1.0], num_runs=10
    )

    # 实验 3：不同 prompt 类型
    print("\n📊 实验 3:不同 prompt 类型在 temperature=0.7 的表现")
    print("=" * 60)
    prompts = {
        "代码": "用 Python 写一个 hello world",
        "创意": "写一句关于月亮的诗",
        "事实": "中国的首都是哪里？",
    }
    for name, p in prompts.items():
        print(f"\n Prompt 类型: {name}")
        run_experiment(client, p, "temperature", [0.0, 0.7, 1.2], num_runs=5)

    # 保存结果
    all_results = {"temperature": temp_results, "top_p": topp_results}
    with open("docs/sampling-experiment-results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print("\n✅ 结果已保存到 docs/sampling-experiment-results.json")


if __name__ == "__main__":
    main()
