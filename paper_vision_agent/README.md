# Paper Vision Agent —— 论文图表多模态问答

上传论文图表，AI 读懂图中数据并结合学术知识图谱回答问题。

![演示：柱状图数值读取 / Transformer 架构理解 / BLEU 表格解析，三图各 3 问](docs/demo.gif)

> GIF 由 [make_gif.py](make_gif.py) 从真实问答卡片合成（[问答采集脚本](run_demo_qa.py) → [卡片渲染](gen_qa_card.py) → GIF），可一键重现。

## 架构

图片 → GLM-4V 结构化描述 → (知识图谱佐证) → LLM 多轮问答

## 快速开始

```bash
pip install -r requirements.txt
# 在仓库根目录 .env 中配置（已有则跳过）：
#   ZHIPU_API_KEY=xxx   # 智谱免费视觉模型
#   LLM_API_KEY=xxx     # 文本模型（DeepSeek 等）
python main.py --image test.png
```
