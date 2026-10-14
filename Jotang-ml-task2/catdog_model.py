"""猫狗项目共用的模型、预处理和数据集；训练与独立推理复用这里。"""
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from PIL import Image, ImageOps
from torchvision import transforms

IMAGE_SIZE = 128
CLASSES = ["Cat", "Dog"]
SEED = 42


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def letterbox(image, size=IMAGE_SIZE):
    """保持比例缩放并居中补灰边，不拉伸、不裁掉主体。"""
    image = ImageOps.exif_transpose(image).convert("RGB")
    return ImageOps.pad(image, (size, size), method=Image.Resampling.BILINEAR,
                        color=(128, 128, 128), centering=(0.5, 0.5))


def tensor_transform(augment=False):
    ops = []
    if augment:
        ops += [transforms.RandomHorizontalFlip(0.5),
                transforms.ColorJitter(brightness=0.15, contrast=0.15,
                                       saturation=0.15, hue=0.03)]
    return transforms.Compose(ops + [transforms.ToTensor(),
                                     transforms.Normalize((0.5,) * 3, (0.5,) * 3)])


def preprocess(image, size=IMAGE_SIZE):
    return tensor_transform(False)(letterbox(image, size))


class CatDogDataset(Dataset):
    """缓存保留 uint8 RGB；每次访问训练样本时重新随机增强。"""
    def __init__(self, cache_path, records, augment=False):
        self.cache_path = str(cache_path)
        self.records = records
        self.transform = tensor_transform(augment)
        self._images = None

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        if self._images is None:
            self._images = np.load(self.cache_path, mmap_mode="r")
        row = self.records[index]
        image = Image.fromarray(np.array(self._images[row["cache_index"]], copy=True))
        return self.transform(image), row["label"]

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_images"] = None  # Windows worker 各自打开内存映射
        return state


class CatDogCNN(nn.Module):
    def __init__(self):
        super().__init__()
        blocks = []
        channels = [3, 16, 32, 64, 128]
        for cin, cout in zip(channels[:-1], channels[1:]):
            blocks += [nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                       nn.BatchNorm2d(cout), nn.ReLU(), nn.MaxPool2d(2)]
        self.features = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(nn.AvgPool2d(2), nn.Flatten(),
                                        nn.Linear(128 * 4 * 4, 128), nn.ReLU(),
                                        nn.Dropout(0.3), nn.Linear(128, 2))

    def forward(self, x):
        return self.classifier(self.features(x))  # 原始 logits 交给 CrossEntropyLoss
