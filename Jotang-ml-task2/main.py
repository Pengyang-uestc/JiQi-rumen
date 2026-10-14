from PIL import Image
import numpy as np
import torch

# 1. 用 Pillow 读取图片，统一为 RGB 三通道
img = Image.open(
    "image\img.png"
).convert("RGB")

# 2. Pillow → NumPy，排列为 HWC
arr = np.array(img)

# 3. NumPy → PyTorch，并把 HWC 调整为 CHW
tensor = torch.from_numpy(arr).permute(2, 0, 1)

print("NumPy:")
print("shape:", arr.shape)
print("dtype:", arr.dtype)
print("最大值:", arr.max())
print("最小值:", arr.min())

print("\nPyTorch:")
print("shape:", tensor.shape)
print("dtype:", tensor.dtype)
print("最大值:", tensor.max().item())
print("最小值:", tensor.min().item())

# 4. 取图片中心附近的一个像素，索引从 0 开始
y, x = 270, 480
print("\nNumPy 像素 RGB:", arr[y, x].tolist())
print("PyTorch 像素 RGB:", tensor[:, y, x].tolist())