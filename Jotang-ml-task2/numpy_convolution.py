import numpy as np


def conv2d(image, kernel, padding="same", stride=1, flip_kernel=True):
    """对灰度(H,W)或RGB(H,W,3)数组做逐通道二维卷积。

    kernel: 二维卷积核，各颜色通道使用同一个核，不混合通道。
    padding: 'valid'不补零，'same'使输出尺寸为ceil(输入尺寸/stride)。
    stride: 正整数，滑动步长。
    flip_kernel: True为数学卷积；False为互相关（深度学习常用形式）。
    返回float64数组，不裁剪、不转uint8，避免负数/大于255的结果丢失。
    """
    image = np.asarray(image)
    kernel = np.asarray(kernel)
    if image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] != 3):
        raise ValueError("image必须是(H,W)灰度图或(H,W,3) RGB图，不能是CHW")
    if image.size == 0 or kernel.ndim != 2 or kernel.size == 0:
        raise ValueError("图片和二维卷积核均不能为空")
    for array in (image, kernel):
        if array.dtype.kind not in "buif" or not np.all(np.isfinite(array)):
            raise ValueError("图片和卷积核必须包含有限实数")
    if isinstance(stride, (bool, np.bool_)) or not isinstance(stride, (int, np.integer)) or stride <= 0:
        raise ValueError("stride必须是正整数")
    if padding not in ("same", "valid"):
        raise ValueError("padding必须是'same'或'valid'")
    if not isinstance(flip_kernel, (bool, np.bool_)):
        raise ValueError("flip_kernel必须是布尔值")

    # 先转浮点，防止uint8相乘/相加时溢出。
    data = image.astype(np.float64)
    weights = kernel.astype(np.float64)
    grayscale = data.ndim == 2
    if grayscale:
        data = data[:, :, None]  # 暂时增加一个通道，统一处理
    height, width, channels = data.shape
    kh, kw = weights.shape
    if flip_kernel:
        weights = weights[::-1, ::-1]  # 数学卷积需要上下、左右翻转

    if padding == "same":
        out_h = (height + stride - 1) // stride
        out_w = (width + stride - 1) // stride
        pad_h = max((out_h - 1) * stride + kh - height, 0)
        pad_w = max((out_w - 1) * stride + kw - width, 0)
        # 奇数个补零时，额外的一行/列放在下侧/右侧。
        top, left = pad_h // 2, pad_w // 2
        data = np.pad(data, ((top, pad_h - top), (left, pad_w - left), (0, 0)),
                      mode="constant", constant_values=0)
    else:
        if kh > height or kw > width:
            raise ValueError("valid模式下卷积核不能大于图片")
        out_h = (height - kh) // stride + 1
        out_w = (width - kw) // stride + 1

    output = np.zeros((out_h, out_w, channels), dtype=np.float64)
    # 核心：遍历输出位置，取窗口，与核逐元素相乘后求和。
    for row in range(out_h):
        for col in range(out_w):
            y, x = row * stride, col * stride
            for channel in range(channels):
                window = data[y:y + kh, x:x + kw, channel]
                output[row, col, channel] = np.sum(window * weights)
    return output[:, :, 0] if grayscale else output


if __name__ == "__main__":
    gray = np.arange(1, 10).reshape(3, 3)
    kernel = np.array([[1, 2], [3, 4]])
    print("输入灰度图：\n", gray)
    print("卷积核：\n", kernel)
    print("数学卷积(valid)：\n", conv2d(gray, kernel, padding="valid"))
    print("互相关(valid)：\n", conv2d(gray, kernel, padding="valid", flip_kernel=False))
    rgb = np.stack([gray, gray * 2, gray * 3], axis=-1)
    result = conv2d(rgb, np.ones((3, 3)) / 9)
    print("RGB均值滤波：shape =", result.shape, "dtype =", result.dtype)
    print("中心像素RGB：", result[1, 1])
