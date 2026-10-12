"""thinking.md 的配套实验。CPU运行，结果保存到 thinking_results。"""
import json
import platform
from pathlib import Path
import numpy as np
import sklearn
import torch
from torch import nn
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, recall_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent / 'thinking_results'
OUT.mkdir(exist_ok=True)
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)

def tensor(x):
    return torch.tensor(x, dtype=torch.float32)

def fit(X, y, V, v, widths, epochs, weight=None):
    torch.manual_seed(42)
    layers = []
    for a, b in zip([2] + widths, widths):
        layers += [nn.Linear(a, b), nn.ReLU()]
    model = nn.Sequential(*layers, nn.Linear(widths[-1], 1))
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    objective = nn.BCEWithLogitsLoss(pos_weight=None if weight is None else tensor([weight]))
    metric = nn.BCEWithLogitsLoss()
    tx, ty, vx, vy = tensor(X), tensor(y).reshape(-1, 1), tensor(V), tensor(v).reshape(-1, 1)
    history = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        objective(model(tx), ty).backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            a, b = model(tx), model(vx)
            history.append([metric(a, ty).item(), metric(b, vy).item(),
                            ((a >= 0) == ty.bool()).float().mean().item(),
                            ((b >= 0) == vy.bool()).float().mean().item()])
    return model, np.array(history)

def metrics(model, X, y):
    with torch.no_grad():
        logits = model(tensor(X)).ravel()
        predicted = (logits >= 0).numpy().astype(int)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, tensor(y)).item()
    return dict(loss=loss, accuracy=float(np.mean(predicted == y)),
                balanced_accuracy=float(balanced_accuracy_score(y, predicted)),
                recall=recall_score(y, predicted, labels=[0, 1], average=None).tolist(),
                confusion_matrix=confusion_matrix(y, predicted, labels=[0, 1]).tolist())

def plot_case(name, model, X, y, V, v, history):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for j, label in enumerate(['Train', 'Validation']):
        axes[0].plot(np.arange(len(history)) + 1, history[:, j], label=label)
        axes[1].plot(np.arange(len(history)) + 1, history[:, j + 2], label=label)
    for ax, title in zip(axes[:2], ['Loss (unweighted)', 'Accuracy']):
        ax.set(title=title, xlabel='Epoch')
        ax.legend()
    axes[1].set_ylim(0, 1.05)
    xx, yy = np.meshgrid(np.linspace(-2, 3, 180), np.linspace(-1.5, 2, 180))
    with torch.no_grad():
        z = torch.sigmoid(model(tensor(np.column_stack([xx.ravel(), yy.ravel()])))).numpy().reshape(xx.shape)
    axes[2].contourf(xx, yy, z, levels=np.linspace(0, 1, 21), cmap='coolwarm', alpha=.5)
    axes[2].scatter(V[:, 0], V[:, 1], c=v, cmap='coolwarm', s=8, alpha=.25, vmin=0, vmax=1)
    axes[2].scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=20, edgecolors='black', linewidths=.4, vmin=0, vmax=1)
    axes[2].set(title='Black edge: training; faint: validation', xlabel='x1', ylabel='x2')
    fig.suptitle(name)
    fig.tight_layout()
    fig.savefig(OUT / f'{name}.png', dpi=150)
    plt.close(fig)
    np.save(OUT / f'{name}_history.npy', history)

def main():
    results = dict(environment=dict(python=platform.python_version(), torch=torch.__version__,
                   numpy=np.__version__, sklearn=sklearn.__version__, seed=42, device='cpu', threads=1))
    # 95:5不均衡；分层划分后训练570:30、验证190:10、测试190:10。
    X, y = make_moons(n_samples=(950, 50), noise=.3, random_state=42)
    X, rest, y, yr = train_test_split(X, y, test_size=.4, random_state=42, stratify=y)
    V, T, v, t = train_test_split(rest, yr, test_size=.5, random_state=42, stratify=yr)
    for name, weight in [('imbalanced', None), ('weighted', 19.)]:
        model, history = fit(X, y, V, v, [16], 300, weight)
        results[name] = metrics(model, T, t)
        plot_case(name, model, X, y, V, v, history)
    # 只污染小训练集的标签，验证/测试保持原始标签，演示记忆训练噪声。
    X, y = make_moons(n_samples=1000, noise=.2, random_state=42)
    X, rest, y, yr = train_test_split(X, y, train_size=60, random_state=42, stratify=y)
    V, T, v, t = train_test_split(rest, yr, test_size=.5, random_state=42, stratify=yr)
    noisy = y.copy()
    flip = np.random.default_rng(42).choice(len(y), size=15, replace=False)
    noisy[flip] = 1 - noisy[flip]
    np.savez(OUT / 'overfit_data.npz', X_train=X, original_y=y, noisy_y=noisy,
             flipped_indices=flip, X_val=V, y_val=v, X_test=T, y_test=t)
    for name, labels, widths, epochs in [('clean_small', y, [16], 300),
                                         ('overfit', noisy, [128,128,128], 3000)]:
        model, history = fit(X, labels, V, v, widths, epochs)
        results[name] = dict(train=metrics(model, X, labels), validation=metrics(model, V, v),
                             test=metrics(model, T, t), best_val_epoch=int(history[:,1].argmin()+1),
                             best_val_loss=float(history[:,1].min()), final_val_loss=float(history[-1,1]))
        plot_case(name, model, X, labels, V, v, history)
    (OUT / 'results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()
