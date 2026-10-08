import sys
import torch

print("Python 版本：", sys.version)
print("PyTorch 版本：", torch.__version__)

# 创建两个 2×2 张量
a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6], [7, 8]])

# 矩阵乘法
result = a @ b

print("张量 a：\n", a)
print("张量 b：\n", b)
print("矩阵乘法结果：\n", result)
print("结果形状：", result.shape)