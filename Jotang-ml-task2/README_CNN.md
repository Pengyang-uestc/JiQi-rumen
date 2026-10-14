# 猫狗 CNN 实验：运行说明

本部分使用你选择的 [Microsoft Cats-vs-Dogs 数据集](https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset)。原来的 `main.py` 和手写卷积实验仍可单独运行。

## 文件各自负责什么

| 文件 | 作用 |
|---|---|
| `prepare_catdog.py` | 检查并解码所有图片，排除损坏、精确重复和跨类别重复，划分数据集，生成缓存与观察图 |
| `catdog_model.py` | 共用的图片预处理、Dataset 和 CNN 定义 |
| `train_cnn.py` | 分别训练无增强/有增强两组模型，记录每轮指标，保存验证集最优模型 |
| `finetune_cnn.py` | 根据基线验证曲线，降低学习率做短程改进实验 |
| `train_fc.py` | 训练同数据条件下的全连接网络，完成卷积与全连接对照 |
| `evaluate_fc.py` | 全连接权重已训练好时单独测试，无需重复训练 |
| `analyze_cnn.py` | 根据验证集选择模型，然后测试、画混淆矩阵、错误样本、特征图及结构图 |
| `predict_catdog.py` | 对一张本地图片独立预测，保存处理后的图片和概率 |
| `test_catdog.py` | 检查数据泄漏、预处理一致性、batch/梯度以及模型保存加载 |
| `cnn_report.md` | 对照题目的中文分析报告 |
| `cnn_results/` | 图片、指标、划分清单、缓存和训练权重 |

## 运行顺序

在 PowerShell 或 VS Code 的终端中运行：

```powershell
conda activate yolo
cd "F:\焦糖工作室招新\Jotang-ml-task2"

# 第一步：准备数据。已经生成缓存时不必每次重跑。
python prepare_catdog.py

# 第二步：验证数据读取和模型。
python -m unittest test_catdog -v

# 第三步：先做基线实验，再观察验证集错误。
python train_cnn.py --experiment baseline --epochs 10 --workers 4

# 第四步：只改变增强策略，进行第二组实验。
python train_cnn.py --experiment augmented --epochs 10 --workers 4

# 第五步：尝试降低学习率改进；再根据验证集选择模型并生成测试图表。
python finetune_cnn.py
python analyze_cnn.py

# 可选的结构对照：把卷积网络换成全连接网络。
python train_fc.py

# 第六步：用训练之外的本地图片预测（把路径换成你的图片）。
python predict_catdog.py --image "F:\我的图片\cat.jpg"
```

若终端没有配置 conda，可以把上面的 `python` 换成 `D:\envs\yolo\python.exe`。
两组实验也可以一次运行：`python train_cnn.py --experiment all --epochs 10 --workers 4`。
Windows 上必须保留脚本末尾的 `if __name__ == "__main__":`，否则多进程加载可能反复启动。
显存不足时，两组都使用相同的较小 batch，例如 `--batch-size 64`；进程加载出问题可用 `--workers 0`。

## 实验产物

- `dataset_audit.json`：原始数量、损坏图片、重复图片、图片尺寸范围、类别与划分统计。
- `splits.json`：每张图片的文件名、类别、SHA256、尺寸、缓存索引及所属集合。
- `images_128.npy`：uint8 RGB 图片缓存，约 1.14 GiB，可由准备脚本重建。
- `baseline/`、`augmented/`、`improved/`：各自包含 `history.csv`、`summary.json`、`best.pt`、`curves.png`、验证集和测试集错误样本、混淆矩阵。
- `fully_connected/`：全连接对照的最佳权重、曲线、训练记录和测试统计；问答与比较见 `answers.md` 第七节。
- `selection.json`：模型选择依据；`comparison.json`：对照结果。
- `best_model.pt`：最终模型，依据验证集最小 loss 选择，包含权重和预处理说明。
- `layer_shapes.json`、`model_structure.png`：真实前向传播记录和结构图。
- `features_1.png`～`features_4.png`：测试集猫狗各两张的卷积特征图。
- `prediction/processed.png`、`prediction/prediction.json`：独立推理结果。

推理程序输出两类 softmax 概率，它们加起来约为 1，但这些分数没有做概率校准，不能理解成现实世界的可靠度保证。输入并非猫或狗时，二分类模型仍会被迫在两类中选一个。

## 可复现设置及环境

固定随机种子 42；分层划分 70%/15%/15%；两组使用相同模型初始化、样本顺序种子、batch size、优化器、学习率和训练轮数。
启用确定性算法，关闭 cuDNN 自动 benchmark，设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`。
GPU 训练使用自动混合精度；验证和测试使用 float32。不同硬件、驱动及 PyTorch 版本仍可能带来数值差别。

本机验证环境：Python 3.11.16；PyTorch 2.13.0+cu130；torchvision 0.28.0+cu130；NumPy 1.23.5；Pillow 12.3.0；Matplotlib 3.7.2；scikit-learn 1.5.2；RTX 5060 Laptop GPU。
可以继续使用已有 `yolo` 环境，无需重复安装 CUDA。精确环境清单另见 `requirements_cnn.txt`。

重复运行训练会更新对应实验目录的输出；要保留多轮实验请先备份 `cnn_results`。`data`、ZIP、缓存和模型权重已列入本项目 `.gitignore`，适合留在本地，报告与图表可提交。
