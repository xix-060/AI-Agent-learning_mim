"""⽤ numpy 从零实现 Scaled Dot-Product Attention"""

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """数值稳定的 softmax"""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def scaled_dot_product_attention(
    Q: np.ndarray, K: np.ndarray, V: np.ndarray
) -> np.ndarray:
    """Scaled Dot-Product Attention
    公式：softmax(QK^T / √d_k) V
    参数：
        Q: (seq_len, d_k) 查询矩阵
        K: (seq_len, d_k) 键矩阵
        V: (seq_len, d_v) 值矩阵
    返回：
        output: (seq_len, d_v) 注意⼒输出
        weights: (seq_len, seq_len) 注意⼒权重
    """
    d_k = K.shape[-1]
    # 1. 计算点积 QK^T
    scores = Q @ K.T  # (seq_len, seq_len)
    # 2. 缩放（除以 √d_k）
    scaled_scores = scores / np.sqrt(d_k)
    # 3. softmax 归⼀化
    weights = softmax(scaled_scores, axis=-1)
    # 4. 加权求和
    output = weights @ V  # (seq_len, d_v)
    return output, weights


def multi_head_attention(
    x: np.ndarray,
    W_Q: np.ndarray,
    W_K: np.ndarray,
    W_V: np.ndarray,
    W_O: np.ndarray,
    num_heads: int,
) -> np.ndarray:
    """
    Multi-Head Attention
    """
    seq_len, d_model = x.shape
    d_k = d_model // num_heads

    # 1. 线性变换得到 Q/K/V
    Q = x @ W_Q  # (seq_len, d_model)
    K = x @ W_K
    V = x @ W_V

    # 2. 拆分成多个头
    Q = Q.reshape(seq_len, num_heads, d_k).transpose(1, 0, 2)  # (h, seq_len, d_k)
    K = K.reshape(seq_len, num_heads, d_k).transpose(1, 0, 2)
    V = V.reshape(seq_len, num_heads, d_k).transpose(1, 0, 2)

    # 3. 每个头独⽴做 attention
    outputs = []
    for i in range(num_heads):
        out, _ = scaled_dot_product_attention(Q[i], K[i], V[i])
        outputs.append(out)

    # 4. 拼接所有头
    concat = np.hstack(outputs)  # (seq_len, d_model)

    # 5. 最后线性变换
    return concat @ W_O


# ========== 演⽰ ==========
if __name__ == "__main__":
    np.random.seed(42)
    seq_len, d_model, num_heads = 4, 8, 2
    d_k = d_model // num_heads  # 4

    # 模拟输⼊（4 个 token，每个 8 维）
    x = np.random.randn(seq_len, d_model)

    # 随机初始化权重矩阵
    W_Q = np.random.randn(d_model, d_model) * 0.1
    W_K = np.random.randn(d_model, d_model) * 0.1
    W_V = np.random.randn(d_model, d_model) * 0.1
    W_O = np.random.randn(d_model, d_model) * 0.1

    # 单头 attention
    Q = x @ W_Q[:, :d_k]
    K = x @ W_K[:, :d_k]
    V = x @ W_V[:, :d_k]
    output, weights = scaled_dot_product_attention(Q, K, V)
    print("单头 Attention 输出形状:", output.shape)
    print("注意⼒权重矩阵 (4x4):")
    print(weights.round(3))
    print("每⾏和为 1（softmax 保证）:", weights.sum(axis=-1).round(3))

    # 多头 attention
    multi_output = multi_head_attention(x, W_Q, W_K, W_V, W_O, num_heads)
    print("\n多头 Attention 输出形状:", multi_output.shape)
