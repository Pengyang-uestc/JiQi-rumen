"""基线错误与验证曲线分析后的单独改进实验：降低学习率短程训练。"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import csv
import json
import time
from pathlib import Path
import torch
from torch import nn
from catdog_model import CatDogCNN, SEED, seed_everything
from train_cnn import OUT, make_loader, evaluate, error_grid, plot_history


def main():
    seed_everything()
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = json.loads((OUT / "splits.json").read_text(encoding="utf-8"))
    baseline = torch.load(OUT / "baseline" / "best.pt", map_location=device, weights_only=True)
    model = CatDogCNN().to(device)
    model.load_state_dict(baseline["model_state"])
    loader = make_loader(splits["train"], False, True, 128, 4)
    validation = make_loader(splits["val"], False, False, 128, 4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    directory = OUT / "improved"
    directory.mkdir(exist_ok=True)
    before = evaluate(model, validation, device)
    best_loss = before["loss"]
    # 起点也保存：若降低学习率未改善，则不能宣称改进成功。
    torch.save({**baseline, "experiment": "improved", "fine_tune_epochs": 0,
                "fine_tune_lr": 0.0002, "parent": "baseline"}, directory / "best.pt")
    start = time.perf_counter()
    history = []
    for epoch in range(1, 6):
        model.train()
        epoch_start = time.perf_counter()
        loss_sum = correct = total = 0
        for x,y in loader:
            x,y = x.to(device,non_blocking=True),y.to(device,non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=="cuda"):
                logits = model(x)
                loss = nn.functional.cross_entropy(logits,y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item()*len(y)
            correct += (logits.argmax(1)==y).sum().item()
            total += len(y)
        val = evaluate(model, validation, device)
        row = {"epoch": epoch, "train_loss": loss_sum/total,
               "train_accuracy": correct/total, "val_loss": val["loss"],
               "val_accuracy": val["accuracy"], "epoch_seconds": time.perf_counter()-epoch_start}
        history.append(row)
        if val["loss"] < best_loss:
            best_loss = val["loss"]
            torch.save({**baseline,"model_state":model.state_dict(),"experiment":"improved",
                        "fine_tune_epochs":epoch,"fine_tune_lr":0.0002,
                        "val_loss":best_loss,"val_accuracy":val["accuracy"],
                        "parent":"baseline"},directory / "best.pt")
        print(f'fine-tune {epoch}/5: train loss {row["train_loss"]:.4f}, val loss {row["val_loss"]:.4f}, val acc {row["val_accuracy"]:.3%}',flush=True)
    with (directory / "history.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(history[0]));writer.writeheader();writer.writerows(history)
    plot_history(history,directory / "curves.png")
    best=torch.load(directory / "best.pt",map_location=device,weights_only=True)
    model.load_state_dict(best["model_state"])
    result=evaluate(model,validation,device)
    error_grid(splits["val"],result,directory / "validation_errors.png","Fine-tuned: validation errors")
    summary={"experiment":"improved","method":"continue baseline checkpoint for up to 5 epochs at lr=0.0002",
             "before_val_loss":before["loss"],"before_val_accuracy":before["accuracy"],
             "fine_tune_epochs_selected":best["fine_tune_epochs"],
             "val_loss":result["loss"],"val_accuracy":result["accuracy"],
             "seconds":time.perf_counter()-start,"device":str(device),"learning_rate":0.0002}
    (directory / "summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary),flush=True)


if __name__=="__main__":
    main()
