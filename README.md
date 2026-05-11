# MNIST CNN 手写数字识别

基于 PyTorch 实现的手写数字识别项目，使用卷积神经网络（CNN）在 MNIST 数据集上训练模型，并提供图片预测和可视化手写画板两种使用方式。

## 技术栈

- Python
- PyTorch
- torchvision
- Pillow
- Tkinter
- MNIST 数据集

## 功能

- 使用 `torchvision.datasets.MNIST` 自动下载并加载 MNIST 数据集
- 基于 CNN 完成 `0-9` 手写数字分类
- 使用 `CrossEntropyLoss` 进行多分类训练
- 使用 `Adam` 优化器更新模型参数
- 支持 CPU / CUDA 自动选择运行设备
- 训练完成后保存模型权重到 `mnist_cnn.pth`
- 支持命令行图片预测，并输出每个数字的概率
- 支持 Tkinter 手写画板，绘制数字后实时识别

## 项目结构

```text
.
├── mnist_cnn.py      # 模型定义、数据加载、训练与测试
├── predict.py        # 加载模型并预测本地图片
├── draw.py           # Tkinter 手写数字识别画板
├── requirements.txt  # Python 依赖
├── .gitignore        # Git 忽略规则
└── README.md
```

## 本地运行

建议使用 Python 3.10 或更新版本。

```bash
pip install -r requirements.txt
```

训练模型：

```bash
python mnist_cnn.py
```

训练脚本会自动下载 MNIST 数据集，并在训练完成后生成 `mnist_cnn.pth`。

预测本地图片：

```bash
python predict.py my_digit.png
```

启动手写画板：

```bash
python draw.py
```

## 模型结构

本项目使用一个轻量级 CNN 网络：

- 输入：`1 x 28 x 28` 灰度图像
- 卷积层 1：`1 -> 32`，卷积核大小 `3x3`
- 最大池化：将特征图尺寸减半
- 卷积层 2：`32 -> 64`，卷积核大小 `3x3`
- 最大池化：继续压缩空间尺寸
- 全连接层：`64 * 7 * 7 -> 128`
- Dropout：降低过拟合风险
- 输出层：`128 -> 10`，对应数字 `0-9`

整体流程是：先通过卷积层提取笔画、边缘、局部形状等特征，再通过全连接层完成数字分类。

## 数据预处理

训练阶段对 MNIST 图像进行了两步处理：

- `ToTensor()`：将图像转换为张量
- `Normalize((0.1307,), (0.3081,))`：使用 MNIST 的均值和标准差进行归一化

在预测自定义图片和手写画板内容时，项目会将图像转换为灰度图，并缩放到 `28x28`。画板预测还会对手写内容进行反色、裁剪和居中处理，让输入形式尽量接近 MNIST 数据集。

## 我学到了什么

通过这个项目，我完整走了一遍深度学习图像分类任务的基本流程：数据加载、数据预处理、模型搭建、损失计算、反向传播、参数更新、模型保存和模型推理。

我更加直观地理解了 CNN 为什么适合图像任务。卷积层可以从局部区域中学习笔画和边缘特征，池化层可以压缩特征并增强一定的平移鲁棒性，全连接层则负责把提取到的图像特征映射到具体类别。

这个项目也让我认识到，模型效果不仅取决于网络结构，推理阶段的输入处理同样重要。如果自己绘制或导入的图片和训练数据分布差异过大，即使模型在测试集上表现不错，也可能出现识别不稳定的问题。因此，保持训练和预测阶段的数据格式一致，是机器学习项目中非常关键的一步。

## 参考链接

- [PyTorch 官方文档](https://docs.pytorch.org/docs/stable/index.html)
- [torchvision MNIST 数据集](https://docs.pytorch.org/vision/main/generated/torchvision.datasets.MNIST.html)
- [MNIST 官方数据集](https://yann.lecun.com/exdb/mnist/)
- [Pillow 文档](https://pillow.readthedocs.io/)
- [Tkinter 文档](https://docs.python.org/3/library/tkinter.html)
