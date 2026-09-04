"""项目B GIF 生成：两跳问答 → 答案 + 推理路径节点逐跳点亮

用法:
    conda activate ai-agent
    python scholar_knowledge/make_demo_gif.py
输入:
    scholar_knowledge/docs/demo_qa.json（由 collect_demo_qa.py 采集）
输出:
    scholar_knowledge/docs/demo.gif
"""

import json
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

BASE = Path(__file__).resolve().parent
DATA = BASE / "docs" / "demo_qa.json"
OUT = BASE / "docs" / "demo.gif"

# 画布
W, H = 800, 500
BG = (248, 250, 252)
CARD_BG = (255, 255, 255)
BORDER = (74, 144, 217)
TEXT_DARK = (30, 41, 59)
TEXT_MUTED = (100, 116, 139)
NODE_BG = (74, 144, 217)
NODE_BG_LIT = (34, 197, 94)
ARROW_COLOR = (148, 163, 184)
Q_BG = (239, 246, 255)

FONT_SIZE = 16
FONT_SMALL = 13
FONT_BOLD = 18


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """加载字体，优先微软雅黑（中文支持），回退黑体/宋体。"""
    candidates = (
        ["msyhbd.ttc", "msyh.ttc"] if bold else ["msyh.ttc", "simhei.ttf", "simsun.ttc"]
    )
    for name in candidates:
        try:
            return ImageFont.truetype(name, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def _wrap(draw: ImageDraw.ImageDraw, text: str, font, max_w: int) -> list[str]:
    """按像素宽度换行文本。"""
    lines = []
    for paragraph in text.split("\n"):
        if not paragraph.strip():
            lines.append("")
            continue
        wrapped = textwrap.wrap(paragraph, width=50)
        for w in wrapped:
            bbox = draw.textbbox((0, 0), w, font=font)
            if bbox[2] - bbox[0] <= max_w:
                lines.append(w)
            else:
                # 逐字缩减
                cur = ""
                for ch in w:
                    if (
                        draw.textbbox((0, 0), cur + ch, font=font)[2]
                        - draw.textbbox((0, 0), cur, font=font)[0]
                        <= max_w
                    ):
                        cur += ch
                    else:
                        lines.append(cur)
                        cur = ch
                if cur:
                    lines.append(cur)
    return lines


def _draw_node(draw, x: int, y: int, w: int, h: int, label: str, font, lit: bool):
    """画一个路径节点圆角矩形。"""
    bg = NODE_BG_LIT if lit else (220, 230, 240)
    text_color = (255, 255, 255) if lit else TEXT_DARK
    draw.rounded_rectangle(
        [x, y, x + w, y + h], radius=10, fill=bg, outline=BORDER, width=2
    )
    # 文字居中换行
    lines = _wrap(draw, label, font, w - 20)
    line_h = font.size + 4
    total_h = len(lines) * line_h
    ty = y + (h - total_h) // 2
    for ln in lines:
        bbox = draw.textbbox((0, 0), ln, font=font)
        tw = bbox[2] - bbox[0]
        draw.text((x + (w - tw) // 2, ty), ln, fill=text_color, font=font)
        ty += line_h


def _draw_arrow(draw, x1: int, y1: int, x2: int, y2: int):
    """画带箭头的连线。"""
    draw.line([(x1, y1), (x2, y2)], fill=ARROW_COLOR, width=3)
    # 箭头三角
    ah = 10
    draw.polygon(
        [(x2, y2), (x2 - ah, y2 - ah // 2), (x2 - ah, y2 + ah // 2)], fill=ARROW_COLOR
    )


def render_frame(
    question: str, answer_lines: list[str], path: list[str], lit_count: int
) -> Image.Image:
    """渲染单帧：问题 + 答案 + 路径节点（前 lit_count 个点亮）。"""
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)

    f_body = _font(FONT_SIZE)
    f_small = _font(FONT_SMALL)
    f_bold = _font(FONT_BOLD, bold=True)

    # 标题
    draw.text((30, 20), "学术知识图谱 GraphRAG", fill=BORDER, font=f_bold)

    # 问题区
    q_h = 60
    draw.rounded_rectangle(
        [30, 55, W - 30, 55 + q_h], radius=8, fill=Q_BG, outline=BORDER, width=1
    )
    draw.text((45, 62), "❓ " + question, fill=TEXT_DARK, font=f_body)

    # 答案区
    y = 135
    draw.text((30, y), "💡 回答：", fill=TEXT_MUTED, font=f_small)
    y += 22
    for line in answer_lines[:8]:  # 最多 8 行
        draw.text((45, y), line, fill=TEXT_DARK, font=f_small)
        y += 18

    # 推理路径区
    py = H - 130
    draw.text(
        (30, py - 25), "🔍 推理路径（图谱多跳验证）", fill=TEXT_MUTED, font=f_small
    )

    if path:
        n = len(path)
        node_w = 200
        node_h = 50
        gap = 40
        total_w = n * node_w + (n - 1) * gap
        start_x = (W - total_w) // 2
        for i, label in enumerate(path):
            nx = start_x + i * (node_w + gap)
            lit = i < lit_count
            _draw_node(draw, nx, py, node_w, node_h, label, f_small, lit)
            if i < n - 1 and i < lit_count - 1:
                # 已点亮的节点之间画箭头
                _draw_arrow(
                    draw,
                    nx + node_w + 2,
                    py + node_h // 2,
                    nx + node_w + gap - 2,
                    py + node_h // 2,
                )
        # 跳数标注
        if lit_count > 1:
            draw.text(
                (W // 2 - 20, py + node_h + 12),
                f"{lit_count - 1} 跳",
                fill=NODE_BG_LIT,
                font=f_small,
            )

    return img


def main() -> None:
    """从 demo_qa.json 生成逐帧 GIF。"""
    data = json.loads(DATA.read_text(encoding="utf-8"))
    question = data["question"]
    # 答案去掉末尾"路径:"行
    answer = data["answer"].split("\n路径:")[0].split("\n依据：")[0].strip()
    path = data["path"]

    draw_tmp = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    answer_lines = _wrap(draw_tmp, answer, _font(FONT_SMALL), W - 90)

    frames = []
    durations = []

    # 帧 1: 只显示问题（悬念）
    f = render_frame(question, ["思考中…"], path, 0)
    frames.append(f)
    durations.append(1200)

    # 帧 2: 显示答案
    f = render_frame(question, answer_lines, path, 0)
    frames.append(f)
    durations.append(2000)

    # 帧 3+: 逐个点亮路径节点
    for lit in range(1, len(path) + 1):
        f = render_frame(question, answer_lines, path, lit)
        frames.append(f)
        durations.append(900)

    # 末帧停留
    durations[-1] = 3000

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
