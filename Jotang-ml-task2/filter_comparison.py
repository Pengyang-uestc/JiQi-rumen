"""Pillow仅负责读写图片；所有滤波均调用手写的NumPy卷积。"""
from pathlib import Path
import json
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numpy_convolution import conv2d


def gaussian_kernel(size=5, sigma=1.0):
    """G(x,y)=exp(-(x²+y²)/(2σ²))，最后除以元素总和。"""
    axis = np.arange(size) - (size - 1) / 2
    xx, yy = np.meshgrid(axis, axis)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / kernel.sum()


def main():
    root = Path(__file__).resolve().parent
    out = root / "filter_results"
    out.mkdir(exist_ok=True)
    with Image.open(root / "image" / "img.png") as image:
        rgb = np.array(image.convert("RGB"))
    # 浮点灰度亮度近似；Sobel在灰度图上计算。
    gray = rgb.astype(np.float64) @ np.array([0.299, 0.587, 0.114])
    kernels = {
        "mean_3x3": np.ones((3, 3)) / 9,
        "gaussian_5x5": gaussian_kernel(),
        "sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
        "sobel_x": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
        "sobel_y": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),
    }
    results = {}
    for name, kernel in kernels.items():
        print("Computing:", name, flush=True)
        # 不翻转核，采用常见的Sobel方向约定；对称的前三种核不受影响。
        results[name] = conv2d(gray if name.startswith("sobel") else rgb,
                               kernel, padding="same", flip_kernel=False)
    gx, gy = results["sobel_x"], results["sobel_y"]
    results["sobel_magnitude"] = np.hypot(gx, gy)
    assert all(np.isfinite(x).all() for x in results.values())
    assert results["mean_3x3"].shape == rgb.shape
    assert results["sobel_magnitude"].shape == rgb.shape[:2]
    assert np.isclose(kernels["gaussian_5x5"].sum(), 1)
    # 原始浮点结果含负梯度/大于255的值，完整保存以便后续分析。
    np.savez_compressed(out / "raw_results.npz", **results)
    (out / "kernels.json").write_text(json.dumps({k: v.tolist() for k, v in kernels.items()}, indent=2), encoding="utf-8")
    stats = {k: dict(shape=list(v.shape), dtype=str(v.dtype), minimum=float(v.min()), maximum=float(v.max()))
             for k, v in results.items()}
    (out / "statistics.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    def uint8(values):
        return np.rint(np.clip(values, 0, 255)).astype(np.uint8)

    # 三张边缘显示图共用同一缩放系数，便于对比强度。
    edge_scale = max(float(results["sobel_magnitude"].max()), 1e-12)
    displays = {"original": rgb, "grayscale": uint8(gray)}
    displays.update({name: uint8(results[name]) for name in ["mean_3x3", "gaussian_5x5", "sharpen"]})
    displays.update({name: uint8(np.abs(results[name]) / edge_scale * 255)
                     for name in ["sobel_x", "sobel_y", "sobel_magnitude"]})
    titles = ["Original RGB", "Mean blur 3 x 3", "Gaussian 5 x 5 (sigma=1)", "Sharpen",
              "Grayscale input", "Sobel X: |Gx|", "Sobel Y: |Gy|", "Sobel: sqrt(Gx^2 + Gy^2)"]
    names = ["original", "mean_3x3", "gaussian_5x5", "sharpen", "grayscale", "sobel_x", "sobel_y", "sobel_magnitude"]
    fig, axes = plt.subplots(2, 4, figsize=(20, 7))
    for ax, name, title in zip(axes.flat, names, titles):
        Image.fromarray(displays[name]).save(out / f"{name}.png")
        ax.imshow(displays[name], cmap="gray", vmin=0, vmax=255)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "comparison.png", dpi=160)
    plt.close(fig)
    # 固定局部放大，不改变滤波输入，便于看清道路标线的差别。
    h, w = rgb.shape[:2]
    region = np.s_[h // 2: h * 5 // 6, w // 2: w * 5 // 6]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, name, title in zip(axes, names[:4], titles[:4]):
        ax.imshow(displays[name][region])
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "detail_comparison.png", dpi=160)
    plt.close(fig)
    print("Input shape:", rgb.shape, "Output:", out)


if __name__ == "__main__":
    main()
