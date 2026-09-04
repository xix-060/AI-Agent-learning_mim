"""项目1 GIF 生成：知识库 Agent 对话流（导入→RAG问答→工具调用）

用法:
    conda activate ai-agent
    python knowledge_agent/make_demo_gif.py
输入:
    knowledge_agent/docs/demo_chat.json（由 collect_demo_chat.py 采集）
输出:
    knowledge_agent/docs/demo.gif
"""

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

BASE = Path(__file__).resolve().parent
DATA = BASE / "docs" / "demo_chat.json"
OUT = BASE / "docs" / "demo.gif"

# 画布（终端风格）
W, H = 800, 520
BG = (30, 30, 40)
TITLE_BAR = (50, 50, 65)
BORDER = (70, 80, 100)
TEXT_USER = (100, 200, 255)
TEXT_AGENT = (120, 230, 130)
TEXT_SYS = (160, 160, 180)
TEXT_DIM = (90, 90, 110)

FONT_SIZE = 15
FONT_TITLE = 14


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """加载支持中文的等宽感字体，优先微软雅黑，回退黑体/宋体。"""
    candidates = (
        ["msyhbd.ttc", "msyh.ttc"] if bold else ["msyh.ttc", "simhei.ttf", "simsun.ttc"]
    )
    for name in candidates:
        try:
            return ImageFont.truetype(name, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def _wrap_text(text: str, max_chars: int = 62) -> list[str]:
    """按字符数换行（等宽字体近似）。"""
    lines = []
    for paragraph in text.split("\n"):
        if not paragraph.strip():
            lines.append("")
            continue
        while len(paragraph) > max_chars:
            lines.append(paragraph[:max_chars])
            paragraph = paragraph[max_chars:]
        lines.append(paragraph)
    return lines


def render_frame(title: str, lines: list[tuple[str, str, str]]) -> Image.Image:
    """渲染终端风格帧。lines = [(speaker, text, color), ...]"""
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)

    f = _font(FONT_SIZE)
    f_title = _font(FONT_TITLE, bold=True)

    # 终端标题栏
    draw.rectangle([0, 0, W, 30], fill=TITLE_BAR)
    # 红黄绿圆点
    for i, color in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        cx = 18 + i * 22
        draw.ellipse([cx, 9, cx + 12, 21], fill=color)
    draw.text((W // 2 - 60, 6), title, fill=TEXT_SYS, font=f_title)

    # 内容区
    y = 45
    for speaker, text, color in lines:
        if speaker:
            prefix = f"[{speaker}] " if speaker != "sys" else "  "
            draw.text((20, y), prefix, fill=color, font=f)
            prefix_w = draw.textbbox((0, 0), prefix, font=f)[2]
        else:
            prefix_w = 0

        wrapped = _wrap_text(text, 56 if speaker else 62)
        for j, line in enumerate(wrapped):
            if j == 0:
                draw.text(
                    (20 + prefix_w, y),
                    line,
                    fill=TEXT_SYS if speaker == "sys" else (200, 200, 210),
                    font=f,
                )
            else:
                draw.text(
                    (20, y),
                    line,
                    fill=TEXT_SYS if speaker == "sys" else (200, 200, 210),
                    font=f,
                )
            y += 22
        y += 6  # 段间距

    # 光标
    draw.rectangle([20, y, 30, y + 16], fill=(200, 200, 210))

    return img


def main() -> None:
    """从 demo_chat.json 生成逐帧 GIF。"""
    records = json.loads(DATA.read_text(encoding="utf-8"))

    frames = []
    durations = []

    # 帧序列：逐轮追加对话
    base_lines: list[tuple[str, str, str]] = []

    for rec in records:
        if rec["kind"] == "stats":
            stats = rec["stats"]
            base_lines.append(
                (
                    "sys",
                    f"知识库: {stats.get('chunk_count', '?')} 分块 | "
                    f"{stats.get('document_count', '?')} 文档",
                    TEXT_DIM,
                )
            )
            base_lines.append(("sys", "输入 'help' 查看帮助，'quit' 退出", TEXT_DIM))
            base_lines.append(("", "", TEXT_SYS))
            f = render_frame("个人知识库 Agent", list(base_lines))
            frames.append(f)
            durations.append(1500)

        elif rec["kind"] == "rag":
            base_lines.append(("您", rec["input"], TEXT_USER))
            base_lines.append(("", "", TEXT_SYS))
            f = render_frame("个人知识库 Agent", list(base_lines))
            frames.append(f)
            durations.append(800)

            base_lines.append(("Agent", rec["answer"], TEXT_AGENT))
            base_lines.append(("", "", TEXT_SYS))
            f = render_frame("个人知识库 Agent", list(base_lines))
            frames.append(f)
            durations.append(2500)

        elif rec["kind"] == "tool":
            base_lines.append(("您", rec["input"], TEXT_USER))
            base_lines.append(("", "", TEXT_SYS))
            f = render_frame("个人知识库 Agent", list(base_lines))
            frames.append(f)
            durations.append(800)

            base_lines.append(("Agent", rec["answer"], TEXT_AGENT))
            base_lines.append(("", "", TEXT_SYS))
            f = render_frame("个人知识库 Agent", list(base_lines))
            frames.append(f)
            durations.append(2500)

    # 末帧长停留
    durations[-1] = 4000

    OUT.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        OUT,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
        colors=128,
    )
    print(f"已生成 {OUT}（{len(frames)} 帧，{OUT.stat().st_size // 1024}KB）")


if __name__ == "__main__":
    main()
