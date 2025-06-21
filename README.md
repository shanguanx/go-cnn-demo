# 数字识别器 (Digit Recognizer)

这是一个完整的数字识别项目，包含Python和Go语言两种实现方式。项目使用MNIST数据集训练卷积神经网络(CNN)来识别手写数字，并提供了详细的代码对比验证。

## 🎯 项目特色

- **双语言实现**: Python (TensorFlow/Keras) + Go (纯Go实现)
- **代码验证**: Python验证脚本确保Go实现的正确性
- **完整流程**: 从数据处理到模型训练再到推理预测
- **可视化工具**: CSV数据转图像可视化
- **详细文档**: 包含API参考手册和开发路线图

## 📁 项目结构

```
digit-recognizer/
├── 📊 数据处理与训练
│   ├── train_model.py              # Python CNN模型训练
│   ├── visualize_digits.py         # CSV数据可视化工具
│   ├── predict_test.py             # 测试数据预测
│   ├── train.csv                   # 训练数据集
│   ├── test.csv                    # 测试数据集
│   └── sample_submission.csv       # 提交样例
│
├── 🤖 模型文件
│   ├── digit_recognition_model.h5  # Python训练模型
│   ├── best_model.h5               # 最佳模型
│   ├── training_history.png        # 训练历史图表
│   └── digit_preview.png           # 数字预览图
│
├── 🐍 Python验证模块 (/python_verify)
│   ├── pytorch_cnn_comparison.py           # PyTorch CNN对比
│   ├── pytorch_verify_dense.py             # 全连接层验证
│   ├── verify_convolutional_layer_pytorch.py # 卷积层验证(PyTorch)
│   ├── verify_convolutional_layer.py       # 卷积层验证
│   ├── convolution_backward_verify.py      # 卷积反向传播验证
│   ├── verify_convolution_python.py        # 卷积操作验证
│   └── verify_convolution.py               # 基础卷积验证
│
├── 🐹 Go语言CNN实现 (/go-cnn)
│   ├── main.go                     # 主程序入口
│   ├── train.go                    # 模型训练
│   ├── inference.go                # 模型推理
│   ├── mnist_model.json            # 训练好的Go模型
│   ├── mnist-cnn                   # 编译后的可执行文件
│   ├── go.mod                      # Go模块文件
│   ├── API-参考手册.md             # API参考文档
│   ├── foundation-status.md        # 基础功能状态
│   └── 📁 核心模块
│       ├── models/                 # 模型定义
│       ├── layers/                 # 网络层实现
│       ├── activations/            # 激活函数
│       ├── losses/                 # 损失函数
│       ├── optimizers/             # 优化器
│       ├── matrix/                 # 矩阵运算
│       ├── graph/                  # 计算图
│       ├── storage/                # 数据存储
│       ├── inference/              # 推理引擎
│       ├── examples/               # 示例代码
│       └── tests/                  # 测试代码
│
└── 📝 文档
    └── roadmap-go-cnn.md           # Go CNN开发路线图
```

## 🚀 快速开始

### 环境要求

#### Python环境
```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn pillow
```

#### Go环境
```bash
go version  # 需要Go 1.16+
```

### 1. 数据可视化

将CSV数据转换为图像文件：
```bash
python visualize_digits.py
```
- 生成 `digit_images/` 目录
- 创建 `digit_preview.png` 预览图

### 2. Python模型训练

训练CNN模型：
```bash
python train_model.py
```
- 训练卷积神经网络
- 保存模型到 `digit_recognition_model.h5`
- 生成训练历史图表 `training_history.png`

### 3. Go语言CNN实现

#### 训练模型
```bash
cd go-cnn
go run main.go train.go
```

#### 推理预测
```bash
cd go-cnn
go run main.go inference.go
```

#### 编译可执行文件
```bash
cd go-cnn
go build -o mnist-cnn main.go train.go inference.go
./mnist-cnn
```

### 4. 代码验证

运行Python验证脚本确保Go实现的正确性：
```bash
cd python_verify
python pytorch_cnn_comparison.py
python verify_convolutional_layer.py
python convolution_backward_verify.py
```

## 🔍 核心功能

### Python实现 (TensorFlow/Keras)

- **数据处理**: CSV文件读取、数据归一化、训练/验证集分割
- **模型架构**: 3层卷积 + 池化 + 全连接层
- **训练优化**: 早停、模型检查点、学习率调度
- **可视化**: 训练历史、数字图像预览

### Go语言实现

- **纯Go实现**: 不依赖外部深度学习库
- **模块化设计**: 分离的层、激活函数、损失函数、优化器
- **矩阵运算**: 自定义矩阵运算库
- **计算图**: 支持前向和反向传播
- **模型序列化**: JSON格式保存/加载模型

### 验证系统

- **PyTorch对比**: 使用PyTorch验证Go实现的正确性
- **逐层验证**: 卷积层、全连接层、激活函数验证
- **反向传播验证**: 梯度计算正确性检查
- **数值精度**: 确保浮点数计算的准确性

## 📊 模型性能

### CNN架构
```
输入层: 28x28x1 (灰度图像)
├── 卷积层1: 32个3x3滤波器 + ReLU
├── 池化层1: 2x2最大池化
├── 卷积层2: 64个3x3滤波器 + ReLU  
├── 池化层2: 2x2最大池化
├── 卷积层3: 64个3x3滤波器 + ReLU
├── 展平层
├── 全连接层: 128个神经元 + ReLU + Dropout(0.5)
└── 输出层: 10个神经元 + Softmax
```

### 训练参数
- **优化器**: Adam
- **损失函数**: Sparse Categorical Crossentropy
- **批次大小**: 128
- **训练轮数**: 20 (带早停)
- **验证集比例**: 20%

## 🛠️ 技术栈

### Python
- **TensorFlow/Keras**: 深度学习框架
- **NumPy**: 数值计算
- **Pandas**: 数据处理
- **Matplotlib/Seaborn**: 数据可视化
- **Scikit-learn**: 机器学习工具

### Go
- **标准库**: 纯Go实现，无外部依赖
- **JSON**: 模型序列化
- **并发**: Goroutine并行计算
- **接口**: 模块化设计

## 📚 文档

- **[API参考手册](go-cnn/API-参考手册.md)**: Go CNN库的详细API文档
- **[开发路线图](roadmap-go-cnn.md)**: Go CNN实现的发展计划
- **[基础功能状态](go-cnn/foundation-status.md)**: 各模块实现状态

## 🔧 开发指南

### 添加新的网络层
1. 在 `go-cnn/layers/` 中实现新层
2. 在 `python_verify/` 中添加验证脚本
3. 更新API文档

### 扩展激活函数
1. 在 `go-cnn/activations/` 中实现新函数
2. 添加对应的Python验证
3. 更新测试用例

### 优化器改进
1. 在 `go-cnn/optimizers/` 中实现新优化器
2. 与PyTorch/TensorFlow对比验证
3. 性能基准测试

## 🤝 贡献

欢迎提交Issue和Pull Request！

### 贡献指南
1. Fork项目
2. 创建功能分支
3. 提交更改
4. 添加测试
5. 提交Pull Request

## 📄 许可证

MIT License

## 🙏 致谢

- MNIST数据集提供者
- TensorFlow/Keras开发团队
- Go语言社区

---

**注意**: 大文件（如train.csv、test.csv）已上传到GitHub，但建议使用Git LFS管理这些文件以获得更好的性能。 