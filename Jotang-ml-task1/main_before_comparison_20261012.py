import random
import numpy as np
import torch

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

from sklearn.datasets import make_moons

X, y = make_moons(
    n_samples=1000,
    noise=0.2,
    random_state=SEED,
)

print(X.shape)  # (1000, 2)
print(y.shape)  # (1000,)
print(X[:5])
print(y[:5])

import matplotlib.pyplot as plt

plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=10)
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()

from sklearn.model_selection import train_test_split

# 第一次：分出 60% 训练集，留下 40% 临时数据
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.4,
    random_state=42,
    stratify=y
)

# 第二次：把临时数据平分成验证集和测试集
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp
)

# 检查划分结果
print("训练集：", X_train.shape, y_train.shape)
print("验证集：", X_val.shape, y_val.shape)
print("测试集：", X_test.shape, y_test.shape)


import torch
from torch import nn

# 在创建模型前固定种子，让模型初始化可复现
torch.manual_seed(42)

device = torch.device("cpu")

X_train = torch.as_tensor(X_train, dtype=torch.float32).to(device)
X_val = torch.as_tensor(X_val, dtype=torch.float32).to(device)
X_test = torch.as_tensor(X_test, dtype=torch.float32).to(device)

# 标签从 (N,) 转为 (N, 1)，与模型输出保持一致
y_train = torch.as_tensor(y_train, dtype=torch.float32).reshape(-1, 1).to(device)
y_val = torch.as_tensor(y_val, dtype=torch.float32).reshape(-1, 1).to(device)
y_test = torch.as_tensor(y_test, dtype=torch.float32).reshape(-1, 1).to(device)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(2, 16),   # 输入层 → 隐藏层
            nn.ReLU(),         # 引入非线性
            nn.Linear(16, 1),   # 隐藏层 → 输出层
        )

    def forward(self, x):
        return self.network(x)


model = MLP().to(device)
print(model)


criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 300

def evaluate(model, X, y):
    model.eval()  # 切换到评估模式

    with torch.no_grad():  # 评估时不需要计算梯度
        logits = model(X)
        loss = criterion(logits, y)

        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= 0.5).float()
        accuracy = (predictions == y).float().mean()

    return loss.item(), accuracy.item()


best_val_loss = float("inf")
save_path = "best_mlp.pth"

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(epochs):
    model.train()  # 每轮训练前切回训练模式

    # 本实验使用全批次训练：每轮使用全部训练样本
    optimizer.zero_grad()              # 清空之前的梯度
    logits = model(X_train)            # 前向传播
    train_loss = criterion(logits, y_train)
    train_loss.backward()              # 反向传播，计算梯度
    optimizer.step()                   # 更新模型参数

    # 每轮参数更新后，评估并记录训练集和验证集的指标
    train_eval_loss, train_accuracy = evaluate(model, X_train, y_train)
    val_loss, val_accuracy = evaluate(model, X_val, y_val)

    train_losses.append(train_eval_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)

    # 只根据验证集表现选择模型，不使用测试集
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), save_path)

    if (epoch + 1) % 50 == 0:
        print(
            f"Epoch {epoch + 1:3d}/{epochs} | "
            f"训练损失：{train_eval_loss:.4f} | "
            f"验证损失：{val_loss:.4f} | "
            f"验证准确率：{val_accuracy:.2%}"
        )

print(f"最佳模型参数已保存到：{save_path}")

# 创建一个结构相同的新模型
loaded_model = MLP().to(device)

# 从文件读取参数并装入模型
state_dict = torch.load(
    save_path,
    map_location=device,
    weights_only=True,
)
loaded_model.load_state_dict(state_dict)

# 用加载后的最佳模型评估测试集
test_loss, test_accuracy = evaluate(loaded_model, X_test, y_test)

print(f"测试损失：{test_loss:.4f}")
print(f"测试准确率：{test_accuracy:.2%}")

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 图片保存在当前工作目录的 images 文件夹中
output_dir = Path("images")
output_dir.mkdir(parents=True, exist_ok=True)

epoch_numbers = range(1, len(train_losses) + 1)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 左图：损失曲线
axes[0].plot(epoch_numbers, train_losses, label="Train")
axes[0].plot(epoch_numbers, val_losses, label="Validation")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_title("Loss Curve")
axes[0].legend()
axes[0].grid(alpha=0.3)

# 右图：准确率曲线
axes[1].plot(epoch_numbers, train_accuracies, label="Train")
axes[1].plot(epoch_numbers, val_accuracies, label="Validation")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].set_ylim(0, 1.05)
axes[1].set_title("Accuracy Curve")
axes[1].legend()
axes[1].grid(alpha=0.3)

fig.tight_layout()
fig.savefig(output_dir / "training_curves.png", dpi=150)
plt.show()

# 转回 CPU 上的 NumPy 数组，供绘图使用
X_test_np = X_test.detach().cpu().numpy()
y_test_np = y_test.detach().cpu().numpy().ravel()

# 根据全部样本的位置确定绘图范围，仅用于显示
X_all_np = torch.cat(
    [X_train, X_val, X_test], dim=0
).detach().cpu().numpy()

x_min = X_all_np[:, 0].min() - 0.5
x_max = X_all_np[:, 0].max() + 0.5
y_min = X_all_np[:, 1].min() - 0.5
y_max = X_all_np[:, 1].max() + 0.5

# 生成 300×300 个网格点
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300),
)

# 将网格整理成 (90000, 2)，每一行是一个位置的坐标
grid = np.column_stack([xx.ravel(), yy.ravel()])

# 网格张量与模型放在同一个设备
model_device = next(loaded_model.parameters()).device
grid_tensor = torch.tensor(
    grid, dtype=torch.float32, device=model_device
)

loaded_model.eval()

with torch.no_grad():
    probabilities = torch.sigmoid(loaded_model(grid_tensor))
    probabilities = probabilities.cpu().numpy().reshape(xx.shape)

fig, ax = plt.subplots(figsize=(7, 5))

# 背景颜色：模型预测属于类别 1 的概率
background = ax.contourf(
    xx,
    yy,
    probabilities,
    levels=np.linspace(0, 1, 21),
    cmap="coolwarm",
    alpha=0.6,
    vmin=0,
    vmax=1,
)

fig.colorbar(background, ax=ax, label="Probability of class 1")

# 概率等于 0.5 的位置，就是分类边界
if probabilities.min() < 0.5 < probabilities.max():
    ax.contour(
        xx, yy, probabilities,
        levels=[0.5],
        colors="black",
        linewidths=2,
    )

# 散点颜色表示真实类别
for label, color in [(0, "blue"), (1, "red")]:
    mask = y_test_np == label
    ax.scatter(
        X_test_np[mask, 0],
        X_test_np[mask, 1],
        color=color,
        edgecolors="white",
        s=30,
        label=f"True class {label}",
    )

ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_title("MLP Decision Boundary — Test Set")
ax.legend()
fig.tight_layout()
fig.savefig(output_dir / "decision_boundary.png", dpi=150)
plt.show()
