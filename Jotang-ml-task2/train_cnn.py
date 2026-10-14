"""python train_cnn.py --experiment baseline/augmented/all --epochs 10"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import csv
import json
import time
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from catdog_model import CatDogCNN, CatDogDataset, CLASSES, IMAGE_SIZE, SEED, seed_everything

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "cnn_results"


def make_loader(rows, augment, shuffle, batch_size, workers):
    generator = torch.Generator().manual_seed(SEED)
    return DataLoader(CatDogDataset(OUT / "images_128.npy", rows, augment),
                      batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      pin_memory=torch.cuda.is_available(), generator=generator,
                      persistent_workers=workers > 0)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_sum, correct, count = 0.0, 0, 0
    probabilities, targets = [], []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        loss_sum += nn.functional.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(1) == y).sum().item()
        count += len(y)
        probabilities.extend(logits.softmax(1).cpu().tolist())
        targets.extend(y.cpu().tolist())
    return {"loss": loss_sum / count, "accuracy": correct / count,
            "probabilities": probabilities, "targets": targets}


def plot_history(history, target):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    epochs = [r["epoch"] for r in history]
    for ax, metric in zip(axes, ["loss", "accuracy"]):
        for split in ["train", "val"]:
            ax.plot(epochs, [r[f"{split}_{metric}"] for r in history], marker="o", label=split)
        ax.set(xlabel="Epoch", ylabel=metric, title=metric.title())
        ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(target, dpi=140); plt.close(fig)


def error_grid(rows, result, target, title):
    import numpy as np
    from PIL import Image, ImageOps
    errors = []
    for row, probability in zip(rows, result["probabilities"]):
        pred = int(np.argmax(probability))
        if pred != row["label"]:
            errors.append({**row, "prediction": pred, "confidence": max(probability),
                           "probabilities": probability})
    errors.sort(key=lambda x: x["confidence"], reverse=True)
    target.with_suffix(".json").write_text(json.dumps(errors, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(2, 4, figsize=(13, 7))
    for ax in axes.flat:
        ax.axis("off")
    for ax, row in zip(axes.flat, errors[:8]):
        with Image.open(ROOT / "data" / "PetImages" / row["path"]) as image:
            ax.imshow(ImageOps.exif_transpose(image).convert("RGB"))
        ax.set_title(f'{row["path"]}\nTrue {CLASSES[row["label"]]}, pred {CLASSES[row["prediction"]]} ({row["confidence"]:.2%})')
    fig.suptitle(title); fig.tight_layout(); fig.savefig(target, dpi=140); plt.close(fig)
    return errors


def train(name, args, splits):
    seed_everything()
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    directory = OUT / name
    directory.mkdir(exist_ok=True)
    train_loader = make_loader(splits["train"], name == "augmented", True, args.batch_size, args.workers)
    val_loader = make_loader(splits["val"], False, False, args.batch_size, args.workers)
    model = CatDogCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history, best_loss = [], float("inf")
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, args.epochs + 1):
        model.train()
        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_start = time.perf_counter()
        total, correct, loss_sum = 0, 0, 0.0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(x)
                loss = nn.functional.cross_entropy(logits, y)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            total += len(y)
        if device.type == "cuda":
            torch.cuda.synchronize()
        training_seconds = time.perf_counter() - epoch_start
        val = evaluate(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": loss_sum / total,
               "train_accuracy": correct / total, "val_loss": val["loss"],
               "val_accuracy": val["accuracy"], "train_seconds": training_seconds,
               "images_per_second": total / training_seconds,
               "epoch_seconds": time.perf_counter() - epoch_start,
               "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0,
               "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20 if device.type == "cuda" else 0}
        history.append(row)
        if val["loss"] < best_loss:
            best_loss = val["loss"]
            torch.save({"model_state": model.state_dict(), "epoch": epoch,
                        "val_loss": best_loss, "val_accuracy": val["accuracy"],
                        "classes": CLASSES, "image_size": IMAGE_SIZE,
                        "normalization_mean": [0.5]*3, "normalization_std": [0.5]*3,
                        "resize": "keep_ratio_gray_padding", "seed": SEED,
                        "experiment": name, "architecture": "CatDogCNN-v1"}, directory / "best.pt")
        with (directory / "history.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row)); writer.writeheader(); writer.writerows(history)
        print(f'{name} epoch {epoch}/{args.epochs}: loss {row["train_loss"]:.4f}/{row["val_loss"]:.4f}, acc {row["train_accuracy"]:.3%}/{row["val_accuracy"]:.3%}, {row["epoch_seconds"]:.1f}s', flush=True)
    checkpoint = torch.load(directory / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    result = evaluate(model, val_loader, device)
    error_grid(splits["val"], result, directory / "validation_errors.png", f"{name}: validation errors (highest confidence)")
    plot_history(history, directory / "curves.png")
    summary = {"experiment": name, "best_epoch": checkpoint["epoch"],
               "val_loss": result["loss"], "val_accuracy": result["accuracy"],
               "epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": 0.001,
               "weight_decay": 1e-4, "optimizer": "Adam", "seed": SEED,
               "seconds": time.perf_counter()-started, "device": str(device),
               "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
               "torch": str(torch.__version__), "num_parameters": sum(p.numel() for p in model.parameters()),
               "peak_allocated_mib": max(r["peak_allocated_mib"] for r in history),
               "peak_reserved_mib": max(r["peak_reserved_mib"] for r in history)}
    (directory / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["baseline", "augmented", "all"], default="all")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("epochs must be positive")
    torch.set_num_threads(2)
    splits = json.loads((OUT / "splits.json").read_text(encoding="utf-8"))
    for name in (["baseline", "augmented"] if args.experiment == "all" else [args.experiment]):
        train(name, args, splits)


if __name__ == "__main__":
    main()
