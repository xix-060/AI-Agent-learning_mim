"""用 numpy 演示 LoRA 低秩分解"""

import numpy as np


def demonstrate_lora():
    """演示 LoRA 原理"""
    np.random.seed(42)

    d = 512  # 原始维度
    r = 8  # 低秩维度

    print(f"原始维度 d={d}, 低秩维度 r={r}")
    print("参数量对比：")
    print(f"  原始 W：{d*d:,} = {d*d:,}")
    print(f"  LoRA A({r}×{d}) + B({d}×{r})：{r*d + d*r:,} = {2*r*d:,}")
    print(f"  减少比例：{1 - 2*r*d/(d*d):.1%}")

    # 原始权重（冻结）
    W = np.random.randn(d, d) * 0.02

    # LoRA 权重（可训练）
    A = np.random.randn(r, d) * 0.01  # 初始化用高斯
    B = np.zeros((d, r))  # B 初始化为 0（开始时 ΔW=0）

    # 模拟输入
    x = np.random.randn(d)

    # 原始输出
    original_output = W @ x

    # LoRA 输出（训练前，B=0 所以 ΔW=0）
    lora_output_before = W @ x + B @ A @ x
    print("\n训练前（B=0）：")
    print("  ΔW = BA = 0，输出 = 原始输出")
    print(f"  差异：{np.abs(original_output - lora_output_before).max():.6f}")

    # 模拟训练后（B 有了值）
    B = np.random.randn(d, r) * 0.01
    delta_W = B @ A  # (d, d)

    lora_output_after = W @ x + delta_W @ x
    print("\n训练后（B≠0）：")
    print(f"  ΔW = BA，形状 {delta_W.shape}")
    print(f"  ΔW 秩：{np.linalg.matrix_rank(delta_W)}（≤ r={r}）")
    print(f"  输出差异：{np.abs(original_output - lora_output_after).max():.4f}")

    # 验证 ΔW 是低秩的
    U, S, Vt = np.linalg.svd(delta_W)
    print("\nΔW 的奇异值（前10个）：")
    print(f"  {S[:10].round(4)}")
    print(f"  非零奇异值数量：{np.sum(S > 1e-10)}（≤ r={r}）")

    # 合并到 W（推理时无额外开销）
    W_merged = W + delta_W
    merged_output = W_merged @ x
    print(f"\n合并后输出差异：{np.abs(lora_output_after - merged_output).max():.10f}")
    print("→ 合并后推理无额外延迟")


if __name__ == "__main__":
    demonstrate_lora()
