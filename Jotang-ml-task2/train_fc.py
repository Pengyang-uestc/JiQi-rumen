"""相同输入/划分/轮数/优化器的全连接对照：python train_fc.py"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import csv
import json
import time

import torch
from torch import nn
from sklearn.metrics import confusion_matrix

from catdog_model import SEED, seed_everything
from train_cnn import OUT, make_loader, evaluate, plot_history


class FullyConnectedCatDog(nn.Module):
    """把 3x128x128 个像素直接展平，完全不使用卷积或池化。"""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(), nn.Linear(3 * 128 * 128, 256), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(256, 2))

    def forward(self, x):
        return self.layers(x)


def main():
    seed_everything()
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = json.loads((OUT / "splits.json").read_text(encoding="utf-8"))
    dest = OUT / "fully_connected"
    dest.mkdir(exist_ok=True)
    train_loader = make_loader(splits["train"], False, True, 128, 4)
    val_loader = make_loader(splits["val"], False, False, 128, 4)
    model = FullyConnectedCatDog().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started, best_loss, history = time.perf_counter(), float("inf"), []
    for epoch in range(1, 11):
        model.train()
        begin = time.perf_counter()
        count = correct = 0
        loss_total = 0.0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(x)
                loss = nn.functional.cross_entropy(logits, y)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss: try CPU or reduce learning rate")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            count += len(y)
            correct += (logits.argmax(1) == y).sum().item()
            loss_total += loss.item() * len(y)
        if device.type == "cuda":
            torch.cuda.synchronize()
        train_seconds = time.perf_counter() - begin
        validation = evaluate(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": loss_total/count,
               "train_accuracy": correct/count, "val_loss": validation["loss"],
               "val_accuracy": validation["accuracy"], "train_seconds": train_seconds,
               "epoch_seconds": time.perf_counter()-begin,
               "peak_allocated_mib": torch.cuda.max_memory_allocated()/2**20 if device.type=="cuda" else 0,
               "peak_reserved_mib": torch.cuda.max_memory_reserved()/2**20 if device.type=="cuda" else 0}
        history.append(row)
        if validation["loss"] < best_loss:
            best_loss = validation["loss"]
            torch.save({"model_state": model.state_dict(), "epoch": epoch,
                        "val_loss": best_loss, "val_accuracy": validation["accuracy"],
                        "architecture": "FullyConnectedCatDog-v1", "image_size": 128,
                        "classes": ["Cat", "Dog"], "seed": SEED}, dest / "best.pt")
        print(f'FC epoch {epoch}/10: train acc {row["train_accuracy"]:.2%}, val acc {row["val_accuracy"]:.2%}, val loss {row["val_loss"]:.4f}', flush=True)
    with (dest / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader(); writer.writerows(history)
    plot_history(history, dest / "curves.png")
    checkpoint = torch.load(dest / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    # 测试集直到验证 loss 最优检查点确定后才读取。
    # Windows 下训练/验证 worker 仍驻留；测试改为主进程读取，避免一次启动 12 个 worker。
    test_loader = make_loader(splits["test"], False, False, 128, 0)
    test = evaluate(model, test_loader, device)
    predictions = torch.tensor(test["probabilities"]).argmax(1).tolist()
    matrix = confusion_matrix(test["targets"], predictions, labels=[0,1]).tolist()
    summary = {"architecture": "Flatten(3*128*128) -> Linear(49152,256) -> ReLU -> Dropout(0.3) -> Linear(256,2)",
               "parameters": sum(p.numel() for p in model.parameters()),
               "best_epoch": checkpoint["epoch"], "val_loss": checkpoint["val_loss"],
               "val_accuracy": checkpoint["val_accuracy"],
               "test_loss": test["loss"], "test_accuracy": test["accuracy"],
               "confusion_matrix": matrix,
               "train_seconds_total": sum(row["train_seconds"] for row in history),
               "run_seconds_total": time.perf_counter()-started,
               "peak_allocated_mib": max(row["peak_allocated_mib"] for row in history),
               "peak_reserved_mib": max(row["peak_reserved_mib"] for row in history),
               "device": str(device), "optimizer": "Adam", "learning_rate": 0.001,
               "weight_decay": 1e-4, "batch_size": 128, "epochs": 10, "seed": SEED,
               "augmentation": False}
    (dest / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
