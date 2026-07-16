"""绘制采样参数实验图表"""

import json
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]  # 中文
matplotlib.rcParams["axes.unicode_minus"] = False

with open("docs/sampling-experiment-results.json", encoding="utf-8") as f:
    data = json.load(f)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Temperature 图
temp_data = data["temperature"]
temp_values = [float(v) for v in temp_data.keys()]
diversity = [temp_data[v]["diversity_ratio"] for v in temp_data.keys()]
axes[0].bar(range(len(temp_values)), diversity, color="steelblue")
axes[0].set_xticks(range(len(temp_values)))
axes[0].set_xticklabels([str(v) for v in temp_values])
axes[0].set_xlabel("Temperature")
axes[0].set_ylabel("多样性比率（唯一回复/总运行）")
axes[0].set_title("Temperature vs 生成多样性")

# Top-P 图
topp_data = data["top_p"]
topp_values = [float(v) for v in topp_data.keys()]
diversity = [topp_data[v]["diversity_ratio"] for v in topp_data.keys()]
axes[1].bar(range(len(topp_values)), diversity, color="coral")
axes[1].set_xticks(range(len(topp_values)))
axes[1].set_xticklabels([str(v) for v in topp_values])
axes[1].set_xlabel("Top-P")
axes[1].set_ylabel("多样性比率")
axes[1].set_title("Top-P vs 生成多样性")

plt.tight_layout()
plt.savefig("docs/sampling-experiment.png", dpi=150)
print("✅ 图表已保存到 docs/sampling-experiment.png")
