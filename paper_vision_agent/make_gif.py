"""把 qa_shots/ 的三张问答卡片截图合成为 GIF（README 首图演示素材）

用法:
    python paper_vision_agent/make_gif.py
输出:
    paper_vision_agent/docs/demo.gif
"""

from pathlib import Path

from PIL import Image

BASE = Path(__file__).resolve().parent
SHOTS = BASE / "test_images" / "qa_shots"
OUT_DIR = BASE / "docs"
FRAMES = ["1_line_chart_qa.png", "2_architecture_qa.png", "3_table_qa.png"]

HOLD_MS = 2600  # 每张卡静止停留时长
FADE_STEPS = 4  # 淡入过渡帧数
FADE_MS = 70  # 每过渡帧时长（淡入共约 280ms）
TARGET_W = 600  # GIF 目标宽度（README 内嵌显示无需原始分辨率，控制体积）
COLORS = 48  # 调色板颜色数（GIF 压缩）


def main() -> None:
    """加载三张卡片，统一尺寸（白底），加淡入过渡，导出循环 GIF。"""
    cards = [Image.open(SHOTS / f).convert("RGB") for f in FRAMES]
    # 统一缩放到目标宽度（LANCZOS 高质量），显著降低 GIF 体积
    cards = [
        im.resize((TARGET_W, int(im.height * TARGET_W / im.width)), Image.LANCZOS)
        for im in cards
    ]
    w = max(im.width for im in cards)
    h = max(im.height for im in cards)

    frames, durations = [], []
    prev_hold = Image.new("RGB", (w, h), (255, 255, 255))  # 首张卡从白底淡入
    for im in cards:
        canvas = Image.new("RGB", (w, h), (255, 255, 255))
        canvas.paste(im, ((w - im.width) // 2, (h - im.height) // 2))
        # 淡入：从上一张卡的停留帧渐变到当前卡
        for s in range(1, FADE_STEPS + 1):
            frames.append(Image.blend(prev_hold, canvas, s / FADE_STEPS))
            durations.append(FADE_MS)
        frames.append(canvas)
        durations.append(HOLD_MS)
        prev_hold = canvas

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "demo.gif"
    frames[0].save(
        out,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
        colors=COLORS,
    )
    print(
        f"已生成 {out}（{len(frames)} 帧，{out.stat().st_size // 1024}KB，约 {sum(durations) // 1000}s）"
    )


if __name__ == "__main__":
    main()
