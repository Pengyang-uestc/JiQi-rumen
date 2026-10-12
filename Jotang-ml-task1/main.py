"""三组对照实验：噪声和学习率分别改变，自动保存所有结果。"""
import random
import time
import json
from pathlib import Path
from datetime import datetime
import platform
import sklearn
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # 自动保存，不弹出阻塞窗口
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def analyze_errors(model, X_test, y_test, name, output_dir):
    """分析最佳模型的测试错误；编号是测试集内从0开始的位置。"""
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(X_test)).cpu().numpy().ravel()
    points = X_test.cpu().numpy()
    truth = y_test.cpu().numpy().ravel().astype(int)
    predicted = (probs >= 0.5).astype(int)
    cm = confusion_matrix(truth, predicted, labels=[0, 1])
    wrong = np.flatnonzero(truth != predicted)
    assert cm.sum() == len(truth)
    assert cm[0, 1] + cm[1, 0] == len(wrong)
    print(f"{name} confusion matrix (rows=true, columns=predicted):\n{cm}")
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=[0, 1]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(name)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    rows = [dict(test_index=int(i), x1=float(points[i, 0]), x2=float(points[i, 1]),
                 true_label=int(truth[i]), predicted_label=int(predicted[i]),
                 probability_class_1=float(probs[i])) for i in wrong]
    (output_dir / "wrong_samples.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (output_dir / "confusion_matrix.json").write_text(json.dumps(cm.tolist()), encoding="utf-8")
    # 同时选择最不确定的错误和高置信度错误，避免只看一种失败情况。
    ordered = wrong[np.argsort(np.abs(probs[wrong] - 0.5))]
    selected = list(dict.fromkeys([int(i) for i in list(ordered[:2]) + list(ordered[::-1][:3])]))
    xx, yy = np.meshgrid(np.linspace(points[:, 0].min() - .5, points[:, 0].max() + .5, 250),
                         np.linspace(points[:, 1].min() - .5, points[:, 1].max() + .5, 250))
    grid = torch.tensor(np.column_stack([xx.ravel(), yy.ravel()]), dtype=torch.float32,
                        device=X_test.device)
    with torch.no_grad():
        surface = torch.sigmoid(model(grid)).cpu().numpy().reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.contourf(xx, yy, surface, levels=np.linspace(0, 1, 21), cmap="coolwarm", alpha=.35)
    if surface.min() < .5 < surface.max():
        ax.contour(xx, yy, surface, levels=[.5], colors="black", linewidths=1)
    ax.scatter(points[:, 0], points[:, 1], c=truth, cmap="coolwarm", vmin=0, vmax=1, s=22)
    ax.scatter(points[wrong, 0], points[wrong, 1], facecolors="none", edgecolors="black",
               s=85, label="All errors")
    for j, i in enumerate(selected):
        ax.annotate(f"#{i}: {truth[i]}->{predicted[i]}", points[i],
                    xytext=(10, 15 + 12 * (j % 3)), textcoords="offset points",
                    fontsize=8, arrowprops=dict(arrowstyle="-", color="black"))
    ax.set(xlabel="x1", ylabel="x2", title=f"Test errors: {name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "wrong_samples.png", dpi=150)
    plt.close(fig)

    lines = [f"# {name}：错误样本分析", "",
             "混淆矩阵行是真实类别，列是预测类别；类别顺序为0、1。",
             f"矩阵：{cm.tolist()}。200个测试样本中分错{len(wrong)}个。",
             f"真实0错分为1：{cm[0, 1]}个；真实1错分为0：{cm[1, 0]}个。", "",
             "![混淆矩阵](confusion_matrix.png)", "", "![错误样本与决策边界](wrong_samples.png)", "",
             "编号为测试集内从0开始的索引。圈出所有错误，标注最多5个，包括概率最接近0.5和最远离0.5的错误。",
             "P(类别1)是模型输出，未经概率校准，不代表实际正确率。", ""]
    for i in selected:
        # 邻域统计只用于事后解释，不参与训练或选择模型。
        distance = np.linalg.norm(points - points[i], axis=1)
        distance[i] = np.inf
        neighbors = np.argsort(distance)[:10]
        opposite = int(np.sum(truth[neighbors] != truth[i]))
        assessment = ("输出接近分类阈值0.5，模型对分类不确定；这不是到边界的几何距离。"
                      if abs(probs[i] - .5) <= .15 else
                      "输出明显偏向错误类别，说明模型也会对错误判断给出较强倾向。")
        lines += [f"## 样本 #{i}",
                  f"坐标({points[i, 0]:.4f}, {points[i, 1]:.4f})；真实{truth[i]}，预测{predicted[i]}；P(类别1)={probs[i]:.4f}。",
                  assessment,
                  f"最近的10个其他测试点中，有{opposite}个属于另一真实类别。" +
                  ("局部标签混杂，支持类别重叠可能参与误判的解释。" if opposite > 0 else
                   "这些邻居标签一致，仍需结合图中的模型边界判断。"),
                  "上述统计只提供线索，不能断定该点一定由噪声造成，也不能据此认定标签错误。", ""]
    if not selected:
        lines.append("本次测试没有错误样本。")
    (output_dir / "error_analysis.md").write_text("\n".join(lines), encoding="utf-8")
    return cm.tolist(), len(wrong)

def run_experiment(name, noise, learning_rate, output_dir):
    output_dir.mkdir(parents=True)
    SEED = 42
    
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    from sklearn.datasets import make_moons
    
    X, y = make_moons(
        n_samples=1000,
        noise=noise,
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
    plt.savefig(output_dir / "data.png", dpi=150)
    plt.close('all')
    
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
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
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
    save_path = output_dir / "best_mlp.pth"
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
    start_time = time.perf_counter()
    
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
            best_epoch = epoch + 1
            torch.save(model.state_dict(), save_path)
    
        if (epoch + 1) % 50 == 0:
            print(
                f"Epoch {epoch + 1:3d}/{epochs} | "
                f"训练损失：{train_eval_loss:.4f} | "
                f"验证损失：{val_loss:.4f} | "
                f"验证准确率：{val_accuracy:.2%}"
            )
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time
    peak_memory = torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else None
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
    confusion, wrong_count = analyze_errors(loaded_model, X_test, y_test, name, output_dir)
    
    import matplotlib.pyplot as plt
    from pathlib import Path
    
    # 所有图片和指标保存在当前实验的独立目录中。
    
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
    plt.close('all')
    
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
    plt.close('all')
    
    
    history = dict(train_loss=train_losses, val_loss=val_losses,
                   train_accuracy=train_accuracies, val_accuracy=val_accuracies)
    result = dict(name=name, noise=noise, learning_rate=learning_rate,
                  best_epoch=best_epoch, best_val_loss=best_val_loss,
                  test_loss=test_loss, test_accuracy=test_accuracy,
                  training_seconds=elapsed, ms_per_epoch=elapsed / epochs * 1000,
                  peak_gpu_tensor_mib=peak_memory, device=str(device),
                  parameters=sum(p.numel() for p in model.parameters()),
                  confusion_matrix=confusion, wrong_count=wrong_count)
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    assert all(len(values) == epochs for values in history.values())
    return result, history
    

def main():
    # 新建时间戳目录，避免覆盖此前的结果。
    root = Path(__file__).resolve().parent / "experiments" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    root.mkdir(parents=True)
    environment = dict(python=platform.python_version(), torch=torch.__version__,
                       sklearn=sklearn.__version__, numpy=np.__version__,
                       matplotlib=matplotlib.__version__, device="cpu",
                       seed=42, epochs=300, batch_size=600,
                       architecture="2 -> 16 -> 1, ReLU", optimizer="Adam")
    (root / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    configs = [("baseline", 0.2, 0.01), ("noise_0.5", 0.5, 0.01), ("lr_0.05", 0.2, 0.05)]
    results, histories = [], []
    for name, noise, lr in configs:
        print(f"\nExperiment: {name}, noise={noise}, lr={lr}")
        result, history = run_experiment(name, noise, lr, root / name)
        results.append(result)
        histories.append(history)
    (root / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for result, history in zip(results, histories):
        for ax, metric in zip(axes, ["loss", "accuracy"]):
            ax.plot(range(1, 301), history["val_" + metric], label=result["name"])
            ax.set(xlabel="Epoch", ylabel="Validation " + metric)
            ax.legend()
            ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(root / "comparison.png", dpi=150)
    plt.close(fig)
    lines = ["# 三组单变量对照实验", "",
             "共同条件：种子42，1000个样本，600/200/200划分，2→16→1，ReLU，Adam，全批次，300轮，CPU。",
             "每组重新初始化，使用最低验证损失的模型进行最终测试。", "",
             "| 实验 | 噪声 | 学习率 | 最佳轮数 | 验证loss | 测试准确率 | 耗时(s) | 每轮(ms) | GPU显存 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in results:
        lines.append(f'| {r["name"]} | {r["noise"]} | {r["learning_rate"]} | {r["best_epoch"]} | {r["best_val_loss"]:.4f} | {r["test_accuracy"]:.2%} | {r["training_seconds"]:.3f} | {r["ms_per_epoch"]:.3f} | 不适用(CPU) |')
    lines += ["", "![验证曲线对比](comparison.png)", "",
              "计时包含训练、验证、训练集评估、检查点保存和循环内打印；不包含数据生成、最终测试和绘图。",
              "单次计时受首次运行开销、磁盘和系统负载影响，不能把微小速度差异直接归因于实验因素。",
              "噪声组改变了训练、验证和测试的数据难度，准确率降低不等于算法本身变差。",
              "CPU实验不使用GPU显存。固定种子不能保证跨软件版本和设备的结果完全相同。",
              "各组目录保存模型、完整300轮指标、数据图、训练曲线和决策边界。"]
    lines += ["", "## 混淆矩阵与错误样本"]
    for r in results:
        lines += [f'- [{r["name"]}：错分{r["wrong_count"]}个，查看具体分析]({r["name"]}/error_analysis.md)']
    (root / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("Results:", root)

if __name__ == "__main__":
    main()
