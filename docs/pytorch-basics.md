# PyTorch 基础

## 1. Tensor（张量）

类似 numpy 的 ndarray，但能在 GPU 上运算、支持自动求导。

## 2. autograd（自动求导）

记录运算图，自动计算梯度。
x.requires\_grad\_(True) → 记录梯度
loss.backward() → 自动算梯度
optimizer.step() → 更新参数

## 3. nn.Module

所有模型的基类。

- **init**：定义层
- forward：定义前向传播

## 4. 训练循环 5 步

1. 前向传播：output = model(input)
2. 算损失：loss = criterion(output, target)
3. 清梯度：optimizer.zero\_grad()
4. 反向传播：loss.backward()
5. 更新参数：optimizer.step()

## 5. DataLoader

批量加载数据，支持 shuffle 和并行加载。
