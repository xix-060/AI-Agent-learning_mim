"""
视觉模型客户端 —— OpenAI 兼容接口
支持智谱 GLM-4V-Flash（免费）/ 阿里 Qwen-VL，通过 VISION_PROVIDER 环境变量切换
"""

import base64
import os
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

PROVIDERS = {
    "zhipu": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4/",
        "model": "glm-4v-flash",
        "key_env": "ZHIPU_API_KEY",
    },
    "dashscope": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-vl-plus",
        "key_env": "DASHSCOPE_API_KEY",
    },
}

MIME_MAP = {"png": "png", "jpg": "jpeg", "jpeg": "jpeg", "webp": "webp", "gif": "gif"}

DESCRIBE_PROMPT = """你是论文阅读助手。请仔细观察这张论文图表，输出结构化描述：
1. 【类型】折线图/柱状图/架构图/流程图/表格/其他
2. 【内容】坐标轴含义、图例、关键数值（尽量精确读出数字）
3. 【结论】这张图传达的核心信息（1-2 句）
4. 【实体】图中出现的论文、方法、模型名称（没有则写"无"）
直接输出四项内容，不要额外解释。"""


class VisionClient:
    def __init__(self, provider: str = None):
        provider = provider or os.getenv("VISION_PROVIDER", "zhipu")
        if provider not in PROVIDERS:
            raise ValueError(
                f"未知的视觉模型提供商: {provider}，可选: {', '.join(PROVIDERS)}"
            )
        cfg = PROVIDERS[provider]
        self.provider = provider
        self.model = cfg["model"]
        api_key = os.getenv(cfg["key_env"])
        if not api_key:
            raise ValueError(
                f"缺少环境变量 {cfg['key_env']}，请在 .env 中配置 {provider} 的 API key"
            )
        self.client = OpenAI(
            api_key=api_key,
            base_url=cfg["base_url"],
            timeout=45,
            max_retries=1,
        )
        print(f"👁️ 视觉模型已连接: {provider} / {self.model}")

    def describe(self, image_path: str, prompt: str = None) -> str:
        """看图说话：输入图片路径，返回结构化描述"""
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        ext = image_path.rsplit(".", 1)[-1].lower()
        mime = MIME_MAP.get(ext, "png")

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt or DESCRIBE_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/{mime};base64,{b64}"},
                        },
                    ],
                }
            ],
            temperature=0.1,
        )
        return resp.choices[0].message.content


if __name__ == "__main__":
    # 自测：test.png 相对本文件定位（paper_vision_agent/test.png），任意 CWD 启动均可
    test_img = Path(__file__).resolve().parent.parent / "test.png"
    vc = VisionClient()
    desc = vc.describe(str(test_img))
    print(desc)
