"""手写 MLP 训练 MNIST"""

import gzip
import os
import struct
import sys
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.optim as optim  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from torchvision import transforms  # noqa: E402
from PIL import Image  # noqa: E402


# ========== 0. 手动下载 MNIST（国内可用的 MNIST 官方 Google 镜像）==========

MNIST_MIRROR = "https://storage.googleapis.com/cvdf-datasets/mnist/"
MNIST_FILES = {
    "train_images": ("train-images-idx3-ubyte.gz", 9_912_422),
    "train_labels": ("train-labels-idx1-ubyte.gz", 28_881),
    "test_images": ("t10k-images-idx3-ubyte.gz", 1_648_772),
    "test_labels": ("t10k-labels-idx1-ubyte.gz", 4_543),
}
MNIST_RAW_DIR = os.path.join("data", "MNIST", "raw")


def _download_mnist():
    """若本地缺失或不完整，从 GCS 镜像下载 4 个 MNIST 文件"""
    os.makedirs(MNIST_RAW_DIR, exist_ok=True)
    for _key, (fname, expected_size) in MNIST_FILES.items():
        dst = os.path.join(MNIST_RAW_DIR, fname)
        if os.path.exists(dst) and os.path.getsize(dst) == expected_size:
            print(f"  [CACHE] {fname}")
            continue
        url = MNIST_MIRROR + fname
        print(f"  [DL   ] {fname} <- {url}")
        urllib.request.urlretrieve(url, dst)


def _load_idx_images(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)
    return data


def _load_idx_labels(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data


# ========== 1. 数据准备 ==========

print("[DATA] 准备 MNIST...")
_download_mnist()

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # MNIST 均值和标准差
    ]
)


class MNISTNPZDataset(Dataset):
    """从本地 idx3-ubyte.gz 加载的 MNIST Dataset"""

    def __init__(self, split: str, transform=None):
        prefix = "train" if split == "train" else "t10k"
        images_path = os.path.join(MNIST_RAW_DIR, f"{prefix}-images-idx3-ubyte.gz")
        labels_path = os.path.join(MNIST_RAW_DIR, f"{prefix}-labels-idx1-ubyte.gz")
        self.images = _load_idx_images(images_path)  # (N, 28, 28) uint8
        self.labels = _load_idx_labels(labels_path)  # (N,) uint8
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_pil = Image.fromarray(self.images[idx], mode="L")
        label = int(self.labels[idx])
        if self.transform:
            img_pil = self.transform(img_pil)
        return img_pil, label


train_dataset = MNISTNPZDataset("train", transform=transform)
test_dataset = MNISTNPZDataset("test", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

print(f"训练集：{len(train_dataset)} 张")
print(f"测试集：{len(test_dataset)} 张")


# ========== 2. 定义模型 ==========


class MLP(nn.Module):
    """多层感知机"""

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 256)  # 输入层 → 隐藏层
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)  # 隐藏层 → 输出层（10 类）

    def forward(self, x):
        x = self.flatten(x)  # (B, 1, 28, 28) → (B, 784)
        x = self.fc1(x)  # → (B, 256)
        x = self.relu(x)  # 激活
        x = self.fc2(x)  # → (B, 10)
        return x


# ========== 3. 训练 ==========


def train(model, device, train_loader, optimizer, criterion, epoch):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    correct = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # 训练循环 5 步
        optimizer.zero_grad()  # 1. 清梯度
        output = model(data)  # 2. 前向传播
        loss = criterion(output, target)  # 3. 算损失
        loss.backward()  # 4. 反向传播
        optimizer.step()  # 5. 更新参数

        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()

        if batch_idx % 100 == 0:
            print(
                f"  Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] "
                f"Loss: {loss.item():.4f}"
            )

    accuracy = 100.0 * correct / len(train_loader.dataset)
    avg_loss = total_loss / len(train_loader)
    print(f"  Epoch {epoch} 完成：平均 Loss={avg_loss:.4f}, 准确率={accuracy:.2f}%")
    return avg_loss, accuracy


def test(model, device, test_loader, criterion):
    """测试"""
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

    accuracy = 100.0 * correct / len(test_loader.dataset)
    avg_loss = test_loss / len(test_loader)
    print(f"  [TEST] Loss={avg_loss:.4f}, 准确率={accuracy:.2f}%")
    return avg_loss, accuracy


# ========== 4. 主函数 ==========


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")

    model = MLP().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print(f"模型参数量：{sum(p.numel() for p in model.parameters()):,}")

    # 训练 5 个 epoch
    epochs = 5
    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    for epoch in range(1, epochs + 1):
        print(f"\n--- Epoch {epoch}/{epochs} ---")
        train_loss, train_acc = train(
            model, device, train_loader, optimizer, criterion, epoch
        )
        test_loss, test_acc = test(model, device, test_loader, criterion)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

    # 保存模型
    torch.save(model.state_dict(), "data/mnist_mlp.pth")
    print("\n[OK] 模型已保存到 data/mnist_mlp.pth")

    # 绘制训练曲线
    plot_training(history)


def plot_training(history):
    """绘制训练曲线"""
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], "b-", label="训练 Loss")
    axes[0].plot(epochs, history["test_loss"], "r-", label="测试 Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss 曲线")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], "b-", label="训练准确率")
    axes[1].plot(epochs, history["test_acc"], "r-", label="测试准确率")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("准确率 (%)")
    axes[1].set_title("准确率曲线")
    axes[1].legend()

    plt.tight_layout()
    os.makedirs("docs", exist_ok=True)
    plt.savefig("docs/mnist-training-curve.png", dpi=150)
    print("[OK] 训练曲线已保存到 docs/mnist-training-curve.png")


if __name__ == "__main__":
    main()
