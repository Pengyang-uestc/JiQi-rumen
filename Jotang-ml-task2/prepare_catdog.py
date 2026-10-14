"""检查全部图片、排除损坏/精确重复文件、固定划分并生成观察图。"""
import hashlib
import json
import warnings
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from catdog_model import IMAGE_SIZE, SEED, CLASSES, letterbox, seed_everything, tensor_transform

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "cnn_results"
DATA = ROOT / "data" / "PetImages"


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    seed_everything()
    OUT.mkdir(exist_ok=True)
    records, invalid, duplicates, conflicts, seen = [], [], [], [], {}
    counts = {}
    for label, name in enumerate(CLASSES):
        files = sorted((DATA / name).glob("*.jpg"))
        counts[name] = len(files)
        for i, path in enumerate(files):
            relative = path.relative_to(DATA).as_posix()
            try:
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    with Image.open(path) as original:
                        image = ImageOps.exif_transpose(original).convert("RGB")
                        image.load()  # 实际解码，不仅检查文件头
                        width, height = image.size
                        if width < 1 or height < 1:
                            raise ValueError("Empty image")
                if digest in seen:
                    previous = seen[digest]
                    duplicates.append({"path": relative, "same_as": previous["path"]})
                    if previous["label"] != label:
                        conflicts.append(previous["path"])
                    continue
                row = {"path": relative, "label": label, "width": width,
                       "height": height, "sha256": digest}
                seen[digest] = row
                records.append(row)
            except (OSError, ValueError, Image.DecompressionBombError) as error:
                invalid.append({"path": relative, "reason": str(error)})
            if (i + 1) % 2500 == 0:
                print(f"Checked {name}: {i+1}/{len(files)}", flush=True)
    records = [r for r in records if r["path"] not in set(conflicts)]
    cache = np.lib.format.open_memmap(OUT / "images_128.npy", mode="w+", dtype=np.uint8,
                                     shape=(len(records), IMAGE_SIZE, IMAGE_SIZE, 3))
    for index, row in enumerate(records):
        row["cache_index"] = index
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with Image.open(DATA / row["path"]) as image:
                cache[index] = np.asarray(letterbox(image))
        if (index + 1) % 5000 == 0:
            print(f"Cached {index+1}/{len(records)}", flush=True)
    cache.flush()
    train, rest = train_test_split(records, test_size=0.3, random_state=SEED,
                                   stratify=[r["label"] for r in records])
    val, test = train_test_split(rest, test_size=0.5, random_state=SEED,
                                stratify=[r["label"] for r in rest])
    splits = {"train": train, "val": val, "test": test}
    save_json(OUT / "splits.json", splits)
    stats = {"source": "https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset",
             "seed": SEED, "original_jpg": counts, "invalid": invalid,
             "duplicates": duplicates, "conflicting_originals_excluded": conflicts,
             "usable": dict(Counter(CLASSES[r["label"]] for r in records)),
             "split_counts": {k: dict(Counter(CLASSES[r["label"]] for r in v)) for k,v in splits.items()},
             "width_min_max": [min(r["width"] for r in records), max(r["width"] for r in records)],
             "height_min_max": [min(r["height"] for r in records), max(r["height"] for r in records)],
             "unique_sizes": len({(r["width"],r["height"]) for r in records})}
    save_json(OUT / "dataset_audit.json", stats)
    rng = np.random.default_rng(SEED)
    samples = [train[int(i)] for i in rng.choice(len(train), 8, replace=False)]
    fig, axes = plt.subplots(2, 4, figsize=(13, 7))
    for ax, row in zip(axes.flat, samples):
        with Image.open(DATA / row["path"]) as image:
            ax.imshow(ImageOps.exif_transpose(image).convert("RGB"))
        ax.set_title(f'{row["path"]}\n{row["width"]} x {row["height"]}')
        ax.axis("off")
    fig.tight_layout(); fig.savefig(OUT / "raw_samples.png", dpi=140); plt.close(fig)
    fig, axes = plt.subplots(4, 3, figsize=(10, 12))
    aug = tensor_transform(True)
    for axes_row, row in zip(axes, samples[:4]):
        with Image.open(DATA / row["path"]) as image:
            raw = ImageOps.exif_transpose(image).convert("RGB")
            processed = letterbox(raw)
            augmented = (aug(processed).permute(1, 2, 0).numpy() * 0.5 + 0.5).clip(0, 1)
            for ax, im, title in zip(axes_row, [raw, processed, augmented],
                                      ["Original", "Keep ratio + pad: 128x128", "Flip + mild color jitter"]):
                ax.imshow(im); ax.set_title(title); ax.axis("off")
    fig.tight_layout(); fig.savefig(OUT / "preprocessing_augmentation.png", dpi=140); plt.close(fig)
    print(json.dumps({k:v for k,v in stats.items() if k not in ("invalid","duplicates")}, ensure_ascii=False), flush=True)
    print(f"Invalid={len(invalid)}, duplicates={len(duplicates)}", flush=True)


if __name__ == "__main__":
    main()
