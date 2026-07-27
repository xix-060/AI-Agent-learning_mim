"""绘制 ReAct 循环图"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def draw_react_loop():
    # 设置中文字体
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_aspect("equal")
    ax.axis("off")

    # 定义节点位置
    nodes = {
        "start": (0, 5),
        "thought": (0, 2.5),
        "action": (-4, 0),
        "observation": (0, -2.5),
        "check": (4, 0),
        "end": (0, -5),
    }

    # 绘制节点
    def draw_node(ax, x, y, text, color, fontsize=14, shape="rect"):
        if shape == "rect":
            box = FancyBboxPatch(
                (x - 1.2, y - 0.5),
                2.4,
                1,
                boxstyle="round,pad=0.1",
                facecolor=color,
                edgecolor="black",
                linewidth=2,
            )
        elif shape == "diamond":
            box = plt.Polygon(
                [(x, y + 0.8), (x + 1.2, y), (x, y - 0.8), (x - 1.2, y)],
                facecolor=color,
                edgecolor="black",
                linewidth=2,
            )
        ax.add_patch(box)
        ax.text(
            x, y, text, ha="center", va="center", fontsize=fontsize, fontweight="bold"
        )

    # 绘制节点
    draw_node(ax, *nodes["start"], "Question\n(输入问题)", "#FFE0B2", fontsize=12)
    draw_node(ax, *nodes["thought"], "Thought\n(LLM 思考)", "#BBDEFB", fontsize=12)
    draw_node(ax, *nodes["action"], "Action\n(执行操作)", "#C8E6C9", fontsize=12)
    draw_node(
        ax, *nodes["observation"], "Observation\n(获取结果)", "#FFF9C4", fontsize=12
    )
    draw_node(
        ax, *nodes["check"], "继续?\n(是/否)", "#F8BBD9", fontsize=12, shape="diamond"
    )
    draw_node(ax, *nodes["end"], "Final Answer\n(输出答案)", "#FFCDD2", fontsize=12)

    # 绘制箭头
    def draw_arrow(ax, start, end, text="", color="black"):
        arrow = FancyArrowPatch(
            start, end, arrowstyle="->", mutation_scale=20, linewidth=2, color=color
        )
        ax.add_patch(arrow)
        if text:
            mid_x = (start[0] + end[0]) / 2
            mid_y = (start[1] + end[1]) / 2
            ax.text(mid_x, mid_y + 0.3, text, ha="center", fontsize=10, color=color)

    # 绘制流程箭头
    draw_arrow(ax, nodes["start"], nodes["thought"], color="#1565C0")
    draw_arrow(ax, nodes["thought"], nodes["action"], color="#2E7D32")
    draw_arrow(ax, nodes["action"], nodes["observation"], color="#2E7D32")
    draw_arrow(ax, nodes["observation"], nodes["check"], color="#2E7D32")
    draw_arrow(ax, nodes["check"], nodes["end"], "否", color="#C62828")

    # 绘制循环箭头 (从 check 回到 thought)
    # 用曲线连接
    ax.annotate(
        "",
        xy=nodes["thought"],
        xytext=nodes["check"],
        arrowprops=dict(
            arrowstyle="->",
            connectionstyle="arc3,rad=-0.4",
            linewidth=2,
            color="#FF6F00",
        ),
    )
    ax.text(2.5, 1.5, "是", fontsize=12, color="#FF6F00", fontweight="bold")

    # 添加标题
    ax.set_title("ReAct 循环流程图", fontsize=18, fontweight="bold", pad=20)

    # 添加图例说明
    legend_texts = [
        "🔵 蓝色箭头: 调用 LLM",
        "🟢 绿色箭头: 执行代码",
        "🔴 红色箭头: 终止条件",
        "🟠 橙色箭头: 循环继续",
    ]
    for i, text in enumerate(legend_texts):
        ax.text(-5.5, 4.5 - i * 0.5, text, fontsize=10, verticalalignment="top")

    plt.tight_layout()
    plt.savefig(
        "docs/react-loop.png",
        dpi=150,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    print("✅ 已保存到 docs/react-loop.png")
    plt.close()


if __name__ == "__main__":
    draw_react_loop()
