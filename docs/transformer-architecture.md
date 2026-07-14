# Transformer 完整架构笔记

> 基于 Lilian Weng 博客 *The Transformer Family*（前30%）及补充资料整理

---

## 1. 三大组件

### 1.1 Embedding 层

**Token Embedding：把 token id 转成向量**

每个 token（词/子词）经过查表操作，从一个 vocab_size × d_model 的嵌入矩阵中取出对应的行向量。就像给每个词发一张"身份证"，上面写着一串数字（维度 d_model = 512）。

```
输入: token id = [352, 117, 8, ...]  （一句话的每个词的编号）
输出: 每个id → 查表 → 得到 512维向量
形状: (n, 512)
```

**Positional Encoding：注入位置信息**

Transformer 的注意力是"一眼看全句"，天然没有顺序感。必须给每个词贴上"座位号"，让模型知道谁在前谁在后。

**原始方案：sin/cos 公式**

$$PE(i, \delta) = \begin{cases} \sin(\frac{i}{10000^{2\delta'/d}}) & \text{if } \delta = 2\delta' \\ \cos(\frac{i}{10000^{2\delta'/d}}) & \text{if } \delta = 2\delta' + 1 \end{cases}$$

- i = 位置编号（第几个词）
- δ = 维度编号（向量的第几维）
- 不同维度用不同频率的波形 → 低维度用高频波（区分近距离位置），高维度用低频波（区分远距离位置）

**直觉类比**：就像用不同频率的信号给每个位置一个独一无二的"身份证号"。既有绝对位置信息（我坐在第3位），又有相对位置信息（我和你差了2个座位）。

**现代方案：RoPE（旋转位置编码）**

LLaMA、Qwen 等现代模型都用 RoPE，它不再给向量"加"位置信息，而是"旋转"向量来编码位置：

$$\begin{aligned} q_m &= R_{\theta,m} \cdot W^q x_m \\ k_n &= R_{\theta,n} \cdot W^k x_n \end{aligned}$$

核心思想：Q 和 K 的点积 q_m · k_n **天然包含相对位置 m-n 的信息**，不需要额外计算。

RoPE 的优势：
- 天然编码相对位置（不像 sin/cos 只编码绝对位置）
- 支持更长上下文（调整 base frequency 即可）
- 数学优雅：旋转矩阵保证内积只依赖相对位置

---

### 1.2 Attention 层

**Self-Attention（Encoder/Decoder 都有）**

每个词同时看整句话的所有词，计算"我应该关注谁"，然后把关注到的信息加权汇总。

公式回顾：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

- Q："我想了解什么？"
- K："我是什么类型？"
- V："我的实际内容是什么？"
- QK^T → 匹配分数矩阵（n×n），这就是注意力矩阵
- softmax → 归一化为概率（每行和为1）
- ×V → 按概率加权混合各词内容

**Cross-Attention（只在 Decoder，关注 Encoder 输出）**

Decoder 在生成译文时，需要"回头看"原文。Cross-Attention 就是这条"回头线"：

- Q 来自 Decoder（"我当前要写什么？")
- K 和 V 来自 Encoder（"原文那边有什么信息？")
- 计算方式和 Self-Attention 一样，只是 Q/K/V 来源不同

直觉类比：翻译工厂里，写作组（Decoder）去问理解组（Encoder）"原文这句话到底什么意思？"

**Causal Mask（Decoder 用，防止看未来 token）**

Decoder 生成第 i 个词时，只能看到第 1~i 个词，不能偷看 i+1 及之后的词。

实现方式：在注意力矩阵上盖一个"遮罩"（mask），把未来位置的注意力分数强制设为 -∞，softmax 后变成 0（即完全不关注）。

```
注意力矩阵（4词句子）加上 causal mask:

        词1  词2  词3  词4
词1  [  0.3  -∞   -∞   -∞ ]  → softmax → [0.9, 0, 0, 0]
词2  [  0.2  0.5  -∞   -∞ ]  → softmax → [0.3, 0.7, 0, 0]
词3  [  0.1  0.3  0.4  -∞ ]  → softmax → [0.2, 0.3, 0.5, 0]
词4  [  0.1  0.2  0.3  0.4]  → softmax → [0.15, 0.25, 0.35, 0.25]
```

---

### 1.3 Feed-Forward 层（FFN）

**两层线性变换 + 激活函数**

每个词在经过注意力层（和其他词交互）之后，还要独自"深度加工"一下：

$$\text{FFN}(x) = W_2 \cdot \sigma(W_1 x + b_1) + b_2$$

- W₁: (d_model, d_ff) = (512, 2048) → 先升维到4倍
- σ: 激活函数
- W₂: (d_ff, d_model) = (2048, 512) → 再降维回来

**原始：ReLU**

$$\text{ReLU}(x) = \max(0, x)$$

简单粗暴：负数直接变0，正数保留。

**现代：SwiGLU（LLaMA/Qwen 用）**

$$\text{SwiGLU}(x, W_1, W_2, V) = \text{Swish}(xW_1) \odot (xV)$$

- Swish 函数：σ(x) = x · sigmoid(βx)，比 ReLU 更平滑
- ⊙：逐元素乘法（门控机制）
- 门控的本质：一个分支决定"放多少信息通过"，另一个分支决定"放什么信息通过"

直觉类比：FFN 就像每个词做完小组讨论后，自己回办公室再独立思考一下。SwiGLU 比 ReLU 更好，因为它不是简单一刀切（正/负），而是有个"闸门"精细调节。

---

## 2. 关键辅助组件

### 2.1 Residual Connection（残差连接，解决深层梯度消失）

$$y = x + F(x)$$

把原始输入 x 直接加到输出上，跳过中间的变换 F。

为什么需要？深层网络有梯度消失问题——信号经过很多层越来越弱。残差连接给信号一条"捷径"，让它可以直接从第1层跳到第64层，不经过中间衰减。

直觉类比：做菜时每加一步调料，都要尝一口原味做参照，防止味道走偏。

Transformer 中每个子层都有残差连接：

```
输出 = LayerNorm( x + SubLayer(x) )
```

### 2.2 Layer Normalization（层归一化）

对每个样本的特征维度做归一化（和 BatchNorm 不同，LayerNorm 不依赖 batch）：

$$\text{LayerNorm}(x) = \frac{x - \mu}{\sigma} \cdot \gamma + \beta$$

- μ, σ：该样本各维度的均值和标准差
- γ, β：可学习的缩放和偏移参数

**原始：LayerNorm**

每个 token 的 512 维向量，减均值除标准差 → 统一"量级"，防止某些维度特别大或特别小。

**现代：RMSNorm（LLaMA/Qwen 用，更高效）**

$$\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \cdot \gamma$$

- 只算 RMS（根均方），不算均值 → 省掉了减均值的计算
- 没有偏移参数 β → 更少的参数量
- 实测效果和 LayerNorm 几乎一样，但更快

**Pre-Normalization vs Post-Normalization**

原始 Transformer：先做子层变换，再做 LayerNorm（Post-Norm）

现代模型：先做 LayerNorm，再做子层变换（Pre-Norm）→ 训练更稳定

```
Post-Norm（原始）: output = LayerNorm( x + SubLayer(x) )
Pre-Norm（现代）:  output = x + SubLayer( LayerNorm(x) )
```

### 2.3 Dropout（防过拟合）

训练时随机"关掉"一部分神经元（设为0），概率通常为 p=0.1。

- 每次训练随机不同神经元被关掉 → 模型不能依赖任何单一路径 → 学到更鲁棒的特征
- 注意力权重和 FFN 输出都会加 Dropout
- 推理/测试时不用 Dropout（所有神经元都参与）

---

## 3. Encoder vs Decoder

| | Encoder | Decoder |
|---|---|---|
| **注意力** | 双向（看全句所有词） | 单向（causal mask，只看已出现词） |
| **注意力层数** | 1个 Self-Attention | 2个：Masked Self-Attention + Cross-Attention |
| **用途** | 理解（BERT） | 生成（GPT/LLaMA/Qwen） |
| **层数** | 6（原始） | 6（原始） |
| **能否看未来** | 能（它一次性读完整句） | 不能（逐词生成，不能偷看） |
| **代表模型** | BERT、RoBERTa | GPT系列、LLaMA、Qwen |

**Encoder 的双向注意力**：

处理"我 喝 了 一杯 咖啡"时，"咖啡"可以看"我"，"我"也可以看"咖啡" → 理解更全面 → 适合做分类、阅读理解等"理解型"任务。

**Decoder 的单向注意力（Causal Mask）**：

生成时逐词输出：已经写了"我 喝 了"，下一个词只能根据这3个词来预测 → 不能偷看后面 → 适合做翻译、对话、文本生成等"生成型"任务。

---

## 4. 现代 LLM 架构演进（Decoder-Only）

现代大语言模型几乎都用 Decoder-Only 架构，原因是生成任务（对话、写作、代码）是主流需求，不需要 Encoder 的双向理解能力。

### 4.1 GPT 系列：标准 Transformer Decoder

- 架构和原始 Transformer 的 Decoder 基本一样
- 只用 Masked Self-Attention（没有 Cross-Attention，因为没有 Encoder）
- 逐年增大：GPT-1 → GPT-2 → GPT-3 → GPT-4

### 4.2 LLaMA：RMSNorm + RoPE + SwiGLU

Meta 开源的代表模型，对原始 Transformer 做了3个关键改进：

| 改进点 | 原始 Transformer | LLaMA | 为什么改？ |
|--------|-----------------|-------|-----------|
| 归一化 | LayerNorm (Post-Norm) | RMSNorm (Pre-Norm) | 更快、更稳定 |
| 位置编码 | sin/cos 固定编码 | RoPE 旋转位置编码 | 天然编码相对位置，支持更长上下文 |
| 激活函数 | ReLU | SwiGLU | 更好的非线性表达 |

### 4.3 Qwen2.5：同 LLaMA + GQA（分组查询注意力）

在 LLaMA 的基础上，又加了一个关键改进——GQA：

**GQA（Grouped Query Attention）**

| | MHA（原始多头注意力） | GQA（分组查询注意力） | MQA（多查询注意力） |
|---|---|---|---|
| Q 头数 | h=8 | h=64（Qwen2.5-72B） | h=64 |
| K/V 头数 | h=8 | G=8（每8个Q头共享1组KV） | 1（所有Q头共享1组KV） |
| KV Cache | 大（每组Q都存KV） | 中（减少到1/8） | 小（极端压缩） |
| 性能 | 最高 | 接近MHA | 略有损失 |

直觉类比：
- **MHA**：8个人各有各的书架，书架占空间大
- **GQA**：8个人分成几组，每组共享一个书架，省空间又不至于信息太少
- **MQA**：8个人全共用1个书架，最省空间但信息可能不够

GQA 的核心优势：推理时 KV Cache 大幅减少 → 速度提升 4~6倍，性能几乎不掉。

**Qwen2.5 架构总结**：

```
Qwen2.5 = Transformer Decoder + GQA + SwiGLU + RoPE + RMSNorm + QKV Bias

具体参数（72B模型）:
- 80层
- 64个Query头 / 8个KV头（GQA，比例8:1）
- d_model = 未知（推测约5120）
- 上下文长度：128K（训练时32K，推理时扩展到128K）
- 词表大小：151,643（BBPE编码，中文支持好）
```

---

## 5. Transformer 家族演进速查表

| 模型/技术 | 年份 | 核心创新 | 解决的问题 |
|-----------|------|---------|-----------|
| Vanilla Transformer | 2017 | 多头自注意力 + sin/cos位置编码 | 并行序列处理，取代RNN |
| Transformer-XL | 2019 | 隐藏状态重用 + 相对位置编码 | 上下文分割/长程依赖 |
| Sparse Transformer | 2019 | 稀疏因式分解注意力 | O(n²) 复杂度问题 |
| Reformer | 2020 | LSH注意力 + 可逆残差层 | 内存与时间效率 |
| Universal Transformer | 2019 | 循环机制 + ACT | 动态计算步骤 |
| GPT系列 | 2018+ | Decoder-Only 大规模预训练 | 生成能力 |
| BERT | 2018 | Encoder 双向预训练 | 理解能力 |
| LLaMA | 2023 | RMSNorm + RoPE + SwiGLU | 更高效的Decoder架构 |
| Qwen2.5 | 2024 | GQA + SwiGLU + RoPE + RMSNorm | KV Cache优化 + 中文优化 |

---

## 6. 关键公式速查

| 公式 | 表达式 | 一句话 |
|------|--------|--------|
| Scaled Dot-Product Attention | softmax(QK^T / √d_k) V | Q和K匹配 → 概率 → 加权混合V |
| Multi-Head Attention | Concat(head_1,...,head_h) W^O | 8个视角并行看 → 拼接 → 融合 |
| sin/cos位置编码 | sin(i/10000^(2δ'/d)) | 不同频率波形给位置编身份证 |
| RoPE | R_{θ,m} · W^q x_m | 旋转Q/K向量 → 天然编码相对位置 |
| FFN (ReLU) | W₂ · ReLU(W₁x + b₁) + b₂ | 升维 → 激活 → 降维 |
| SwiGLU | Swish(xW₁) ⊙ (xV) | 门控机制精细调节信息流 |
| 残差连接 | y = x + F(x) | 给信号一条捷径，防梯度消失 |
| RMSNorm | x/RMS(x) · γ | 不算均值，更快的归一化 |
