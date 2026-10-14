"""训练已完成但测试阶段中断时，直接评估保存的最佳全连接权重。"""
import csv
import json
from pathlib import Path
import torch
from sklearn.metrics import confusion_matrix
from catdog_model import seed_everything
from train_cnn import OUT, make_loader, evaluate
from train_fc import FullyConnectedCatDog


def main():
    seed_everything()
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    directory = OUT / "fully_connected"
    checkpoint = torch.load(directory / "best.pt", map_location=device, weights_only=True)
    model = FullyConnectedCatDog().to(device)
    model.load_state_dict(checkpoint["model_state"])
    splits = json.loads((OUT / "splits.json").read_text(encoding="utf-8"))
    test_loader = make_loader(splits["test"], False, False, 128, 0)
    result = evaluate(model, test_loader, device)
    predictions = torch.tensor(result["probabilities"]).argmax(1).tolist()
    rows = list(csv.DictReader((directory / "history.csv").open(encoding="utf-8")))
    summary = {"architecture": "Flatten(3*128*128) -> Linear(49152,256) -> ReLU -> Dropout(0.3) -> Linear(256,2)",
               "parameters": sum(p.numel() for p in model.parameters()),
               "best_epoch": checkpoint["epoch"], "val_loss": checkpoint["val_loss"],
               "val_accuracy": checkpoint["val_accuracy"],
               "test_loss": result["loss"], "test_accuracy": result["accuracy"],
               "confusion_matrix": confusion_matrix(result["targets"], predictions, labels=[0,1]).tolist(),
               "train_seconds_total": sum(float(r["train_seconds"]) for r in rows),
               "peak_allocated_mib": max(float(r["peak_allocated_mib"]) for r in rows),
               "peak_reserved_mib": max(float(r["peak_reserved_mib"]) for r in rows),
               "device": str(device), "optimizer": "Adam", "learning_rate": 0.001,
               "weight_decay": 1e-4, "batch_size": 128, "epochs": 10, "seed": 42,
               "augmentation": False}
    (directory / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
