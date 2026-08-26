"""多智能体协作开发实验：PM → 开发 → 测试"""

import json
import subprocess
import sys
from pathlib import Path
from src.llm_client import LLMClient


class DevTeam:
    """三角色开发团队"""

    def __init__(self, llm: LLMClient):
        self.llm = llm
        self.roles = {
            "pm": """你是产品经理。把需求转成清晰的功能规格（输入/输出/边界情况）。
输出格式：功能规格（不超过 10 行）。""",
            "dev": """你是开发工程师。按规格写 Python 代码。
要求：类型注解、中文 docstring、只用标准库。
只输出代码，不要解释。""",
            "tester": """你是测试工程师。审查代码是否满足规格。
输出格式：
1. 每条规格是否满足（逐条 ✅/❌）
2. 发现的 bug 或边界问题（没有则写"无"）
3. 结论：PASS 或 FAIL(原因)""",
        }

    def _ask(self, role: str, content: str) -> str:
        return self.llm.simple_chat(content, system_prompt=self.roles[role])

    def develop(self, requirement: str, max_rounds: int = 3) -> dict:
        """完整协作流程"""
        log = []

        # 1. PM 出规格
        spec = self._ask("pm", f"用户需求：{requirement}")
        log.append({"role": "PM", "output": spec})
        print(f"📋 PM 规格如下：\n{spec}\n")

        # 2. 开发按规格写码（可多轮返工）
        code = ""
        for round_num in range(1, max_rounds + 1):
            prompt = f"按以下规格实现：\n{spec}"
            if code:
                prompt += f"\n\n这是上一版代码（测试未通过，请修复）：\n```python\n{code}\n```"
            code = self._ask("dev", prompt).strip()
            code = code.removeprefix("```python").removesuffix("```").strip()
            log.append(
                {"role": "DEV", "round": round_num, "output": code[:200] + "..."}
            )

            # 3. 测试审查
            review = self._ask(
                "tester", f"规格：\n{spec}\n\n代码：\n```python\n{code}\n```"
            )
            log.append({"role": "TESTER", "round": round_num, "output": review})
            print(f"🧪 第 {round_num} 轮测试审查：\n{review}\n")

            if "PASS" in review:
                break
        else:
            print("⚠️ 达到最大轮数")

        # 4. 真实验证：写入文件并执行
        result = self._actually_run(code)

        return {
            "spec": spec,
            "code": code,
            "review": review,
            "run_result": result,
            "log": log,
        }

    def _actually_run(self, code: str) -> str:
        """真实执行代码验证"""
        test_file = Path("data/multi_agent_output.py")
        test_file.parent.mkdir(exist_ok=True)
        test_file.write_text(code, encoding="utf-8")

        try:
            r = subprocess.run(
                [sys.executable, str(test_file)],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return f"exit={r.returncode}\nstdout:\n{r.stdout}\nstderr:\n{r.stderr}"
        except Exception as e:
            return f"执行失败: {e}"


def main():
    llm = LLMClient()
    team = DevTeam(llm)

    requirement = "写一个函数 markdown_to_html(md_text: str) -> str，支持 # 标题、**加粗**、- 列表、段落四种语法转换。main 里写 3 个自测示例并 print 结果。"

    result = team.develop(requirement)

    print(f"\n{'='*60}")
    print("🏁 最终结果")
    print(f"{'='*60}")
    print(f"真实执行输出：\n{result['run_result']}")

    # 保存实验记录
    Path("docs").mkdir(exist_ok=True)
    Path("docs/multi-agent-dev-log.json").write_text(
        json.dumps(result["log"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n📄 协作日志已保存到 docs/multi-agent-dev-log.json")


if __name__ == "__main__":
    main()
