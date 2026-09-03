"""临时脚本：把 qa_results.json 渲染成问答卡片 HTML（GIF 素材用）

用法:
    python paper_vision_agent/gen_qa_card.py
输出:
    paper_vision_agent/test_images/qa_card.html
"""

import base64
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent / "test_images"
META = {
    "1_line_chart.png": ("📊 柱状图 · ReAct 论文 Figure 3", "数值读取 + 图谱联动"),
    "2_architecture_transformer.png": (
        "🏗 架构图 · Attention 论文 Figure 1",
        "结构理解 + 知识库联动",
    ),
    "3_table.png": ("📋 表格 · Attention 论文 Table 2", "表格解析"),
}


def img_b64(name: str) -> str:
    """图片转 base64 内嵌"""
    return base64.b64encode((BASE / name).read_bytes()).decode()


def md2html(text: str) -> str:
    """轻量 markdown：粗体/列表/换行"""
    import re

    t = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?m)^\s*[-*]\s+", "• ", t)
    return t.replace("\n", "<br>")


def qa_block(r: dict) -> str:
    """单个问答对"""
    return f"""
    <div class="qa">
      <div class="q"><span class="who">你</span>{r["question"]}</div>
      <div class="a"><span class="bot">🤖 Agent</span>{md2html(r["answer"])}</div>
    </div>"""


def card(name: str, rows: list) -> str:
    """一张图 + 3 组问答的卡片"""
    title, tag = META[name]
    qas = "\n".join(qa_block(r) for r in rows)
    return f"""
  <section class="card">
    <div class="head">
      <span class="title">{title}</span>
      <span class="tag">{tag}</span>
    </div>
    <img class="fig" src="data:image/png;base64,{img_b64(name)}" alt="{name}"/>
    {qas}
    <div class="foot">PaperVision Agent · GLM-4V-Flash 视觉理解 + ScholarGraph 图谱佐证</div>
  </section>"""


def main() -> None:
    """渲染全部卡片为一个 HTML 文件"""
    results = json.loads((BASE / "qa_results.json").read_text(encoding="utf-8"))
    cards = "\n".join(
        card(name, [r for r in results if r["image"] == name]) for name in META
    )
    html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<style>
  body {{ background:#eef1f6; font-family:"Microsoft YaHei",sans-serif; margin:0; padding:16px; }}
  .card {{ width:600px; margin:0 auto 24px; background:#fff; border-radius:16px;
          box-shadow:0 2px 12px rgba(0,0,0,.08); overflow:hidden; }}
  .head {{ display:flex; justify-content:space-between; align-items:center; padding:14px 18px 0; }}
  .title {{ font-size:17px; font-weight:700; color:#1a2233; }}
  .tag {{ font-size:11px; color:#2563eb; background:#eff6ff; border-radius:10px; padding:3px 10px; }}
  .fig {{ width:100%; max-height:250px; object-fit:contain; background:#f8fafc; padding:10px 0; }}
  .qa {{ padding:6px 18px; }}
  .q, .a {{ position:relative; margin:10px 0; padding:10px 14px; border-radius:12px;
           font-size:13px; line-height:1.65; color:#2a2f3a; }}
  .q {{ background:#2563eb; color:#fff; border-top-left-radius:2px; }}
  .a {{ background:#f4f6fa; border:1px solid #e5e9f2; border-top-right-radius:2px; }}
  .who, .bot {{ display:block; font-size:10px; opacity:.75; margin-bottom:4px; }}
  .foot {{ text-align:center; font-size:11px; color:#98a2b3; padding:10px 0 16px; }}
</style>
</head>
<body>
{cards}
</body>
</html>"""
    out = BASE / "qa_card.html"
    out.write_text(html, encoding="utf-8")
    print(f"已生成 {out}")


if __name__ == "__main__":
    main()
