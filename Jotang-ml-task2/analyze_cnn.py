"""冻结验证集选出的模型，再评估测试集，生成特征图、结构图和结果。"""
import json
import shutil
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from catdog_model import CatDogCNN, CatDogDataset, CLASSES, SEED, seed_everything
from train_cnn import OUT, make_loader, evaluate, error_grid


def shapes_and_diagram(model, device, sample_row):
    shapes, handles = [], []
    for name, module in model.named_modules():
        if name and not list(module.children()):
            def hook(m, args, out, name=name):
                shapes.append({"layer": name, "type": type(m).__name__,
                               "input": list(args[0].shape), "output": list(out.shape)})
            handles.append(module.register_forward_hook(hook))
    sample, _ = CatDogDataset(OUT / "images_128.npy", [sample_row])[0]
    with torch.no_grad():
        model(sample.unsqueeze(0).to(device))
    for handle in handles:
        handle.remove()
    (OUT / "layer_shapes.json").write_text(json.dumps({"sample":sample_row["path"], "layers":shapes}, indent=2), encoding="utf-8")
    labels = ["RGB input\nN x 3 x 128 x 128", "Conv 3x3: 3 -> 16, BN, ReLU\nMaxPool 2x2 -> N x 16 x 64 x 64",
              "Conv 3x3: 16 -> 32, BN, ReLU\nMaxPool 2x2 -> N x 32 x 32 x 32",
              "Conv 3x3: 32 -> 64, BN, ReLU\nMaxPool 2x2 -> N x 64 x 16 x 16",
              "Conv 3x3: 64 -> 128, BN, ReLU\nMaxPool 2x2 -> N x 128 x 8 x 8",
              "AvgPool 2x2 -> 4x4 + Flatten\nN x 2048",
              "Linear 2048 -> 128, ReLU, Dropout(0.3)",
              "Linear 128 -> 2 logits\nSoftmax (inference): P(Cat), P(Dog)"]
    fig, ax = plt.subplots(figsize=(8, 11))
    ax.set(xlim=(0, 1), ylim=(0, len(labels))); ax.axis("off")
    for i, label in enumerate(labels):
        y = len(labels) - i - 0.5
        ax.text(0.5, y, label, ha="center", va="center", fontsize=11,
                bbox=dict(boxstyle="round,pad=0.6", facecolor="#e8f1fb", edgecolor="#2764a0"))
        if i < len(labels)-1:
            ax.annotate("", xy=(0.5, y-0.67), xytext=(0.5,y-0.34), arrowprops=dict(arrowstyle="->"))
    fig.tight_layout(); fig.savefig(OUT / "model_structure.png", dpi=140); plt.close(fig)


def feature_maps(model, rows, device):
    rng = np.random.default_rng(SEED)
    selected = []
    for label in [0, 1]:
        candidates = [r for r in rows if r["label"] == label]
        selected.extend(candidates[int(i)] for i in rng.choice(len(candidates), 2, replace=False))
    dataset = CatDogDataset(OUT / "images_128.npy", selected)
    outputs = {}
    handles = []
    for name, module in [("first_conv", model.features[0]), ("last_conv", model.features[12])]:
        def hook(m, args, out, name=name):
            outputs[name] = out.detach().cpu()[0]
        handles.append(module.register_forward_hook(hook))
    for i, row in enumerate(selected):
        x, _ = dataset[i]
        with torch.no_grad():
            model(x.unsqueeze(0).to(device))
        fig, axes = plt.subplots(2, 9, figsize=(16, 4))
        for axes_row, (layer, maps) in zip(axes, outputs.items()):
            axes_row[0].imshow((x.permute(1,2,0).numpy()*0.5+0.5).clip(0,1))
            axes_row[0].set_title(row["path"], fontsize=9)
            indices = maps.flatten(1).std(1).argsort(descending=True)[:8]
            for ax, channel in zip(axes_row[1:], indices):
                values = maps[channel].numpy()
                # 每张特征图单独缩放，只比较空间响应位置，不能比较绝对强弱。
                ax.imshow(values, cmap="viridis", vmin=float(values.min()), vmax=float(values.max()))
                ax.set_title(f"{layer}\nchannel {int(channel)}", fontsize=8)
            for ax in axes_row:
                ax.axis("off")
        fig.suptitle("Raw convolution outputs; per-channel color scale; channels selected by spatial std", fontsize=10)
        fig.tight_layout(); fig.savefig(OUT / f"features_{i+1}.png", dpi=140); plt.close(fig)
    for handle in handles:
        handle.remove()
    (OUT / "feature_samples.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")


def main():
    torch.set_num_threads(2)
    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = json.loads((OUT / "splits.json").read_text(encoding="utf-8"))
    names = ["baseline", "augmented"]
    if (OUT / "improved" / "summary.json").exists():
        names.append("improved")
    summaries = {name: json.loads((OUT / name / "summary.json").read_text()) for name in names}
    selected = min(summaries, key=lambda name: summaries[name]["val_loss"])
    # 选择发生在第一次测试集评估之前；不根据测试结果改选。
    shutil.copy2(OUT / selected / "best.pt", OUT / "best_model.pt")
    (OUT / "selection.json").write_text(json.dumps({"selected": selected, "criterion": "minimum validation cross entropy", "summaries": summaries}, indent=2), encoding="utf-8")
    test_loader = make_loader(splits["test"], False, False, 128, 4)
    model = CatDogCNN().to(device)
    comparison = {}
    for name in names:
        checkpoint = torch.load(OUT / name / "best.pt", map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state"])
        result = evaluate(model, test_loader, device)
        pred = np.argmax(result["probabilities"], axis=1)
        matrix = confusion_matrix(result["targets"], pred, labels=[0,1])
        metrics = {"loss": result["loss"], "accuracy": result["accuracy"],
                   "confusion_matrix": matrix.tolist(),
                   "classification_report": classification_report(result["targets"], pred, target_names=CLASSES, output_dict=True, zero_division=0)}
        comparison[name] = {**summaries[name], "test": metrics}
        (OUT / name / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        fig, ax = plt.subplots(figsize=(5,4))
        ax.imshow(matrix, cmap="Blues")
        for (i,j), count in np.ndenumerate(matrix):
            ax.text(j,i,str(count),ha="center",va="center",color="white" if count > matrix.max()/2 else "black")
        ax.set(xticks=[0,1],yticks=[0,1],xticklabels=CLASSES,yticklabels=CLASSES,
               xlabel="Prediction",ylabel="True label",title=f"{name}: test confusion matrix")
        fig.tight_layout(); fig.savefig(OUT / name / "confusion_matrix.png", dpi=140); plt.close(fig)
        error_grid(splits["test"], result, OUT / name / "test_errors.png", f"{name}: test errors (highest confidence)")
        # 保存逐个测试样本的输出，便于检查错误是否来自同一张图。
        (OUT / name / "test_predictions.json").write_text(json.dumps([
            {"path": row["path"], "label": row["label"], "probabilities": p}
            for row,p in zip(splits["test"],result["probabilities"])], indent=2), encoding="utf-8")
    (OUT / "comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    checkpoint = torch.load(OUT / "best_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"]); model.eval()
    shapes_and_diagram(model, device, splits["test"][0])
    feature_maps(model, splits["test"], device)
    fig, axes = plt.subplots(1,2,figsize=(10,4))
    axes[0].bar(names, [comparison[n]["test"]["accuracy"] for n in names])
    axes[0].set(ylabel="Test accuracy", ylim=(0,1),title="Final comparison (selection used validation)")
    axes[1].bar(names, [comparison[n]["seconds"] for n in names])
    axes[1].set(ylabel="Seconds", title="Stage runtime (fine-tuning excludes baseline)")
    fig.tight_layout(); fig.savefig(OUT / "comparison.png", dpi=140); plt.close(fig)
    print(json.dumps({"selected": selected, "test_accuracy": {n:comparison[n]["test"]["accuracy"] for n in comparison}}, indent=2))


if __name__ == "__main__":
    main()
