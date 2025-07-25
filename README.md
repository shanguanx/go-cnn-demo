# Go语言CNN数字识别器

这是一个使用纯Go语言实现的卷积神经网络(CNN)项目，用于识别手写数字。项目包含完整的CNN实现，从底层矩阵运算到高层网络架构，并提供了详细的验证系统确保实现的正确性。

## 🎯 项目特色

- **纯Go实现**: 不依赖任何外部深度学习库，完全使用Go标准库
- **模块化设计**: 清晰的模块分离，易于理解和扩展
- **完整验证**: 提供Python验证脚本确保Go实现的正确性
- **详细文档**: 包含API参考手册和开发路线图
- **生产就绪**: 支持模型训练、推理和序列化

## 📁 项目结构

```
digit-recognizer/
├── 🐹 Go语言CNN实现 (/go-cnn) - 核心实现
│   ├── main.go                     # 主程序入口
│   ├── train.go                    # 模型训练
│   ├── inference.go                # 模型推理
│   ├── mnist_model.json            # 训练好的Go模型
│   ├── mnist-cnn                   # 编译后的可执行文件
│   ├── go.mod                      # Go模块文件
│   ├── API-参考手册.md             # API参考文档
│   ├── foundation-status.md        # 基础功能状态
│   └── 📁 核心模块
│       ├── models/                 # LeNet-5模型定义
│       ├── activations/            # 激活函数(ReLU/Sigmoid/Softmax)
│       ├── losses/                 # 损失函数(交叉熵/MSE)
│       ├── optimizers/             # 优化器(SGD)
│       ├── matrix/                 # 矩阵运算(卷积/广播/统计)
│       ├── graph/                  # 计算图(自动微分/层定义)
│       ├── storage/                # 模型序列化(JSON格式)
│       ├── inference/              # 推理引擎(批量预测/性能评估)
│       ├── data/                   # 数据加载(MNIST CSV/批处理)
│       ├── examples/               # 示例代码
│       └── tests/                  # 测试代码
│
├── 🐍 验证模块 (/python_verify) - 确保Go实现正确性
│   ├── pytorch_cnn_comparison.py           # PyTorch CNN对比
│   ├── pytorch_verify_dense.py             # 全连接层验证
│   ├── verify_convolutional_layer_pytorch.py # 卷积层验证(PyTorch)
│   ├── verify_convolutional_layer.py       # 卷积层验证
│   ├── convolution_backward_verify.py      # 卷积反向传播验证
│   ├── verify_convolution_python.py        # 卷积操作验证
│   └── verify_convolution.py               # 基础卷积验证
│
├── 📊 数据文件
│   ├── train.csv                   # 训练数据集
│   ├── test.csv                    # 测试数据集
│   └── sample_submission.csv       # 提交样例
│
└── 📝 文档
    └── roadmap-go-cnn.md           # Go CNN开发路线图
```

## 🚀 快速开始

### 环境要求

```bash
go version  # 需要Go 1.16+
```

### 1. 克隆项目

```bash
git clone <your-repository-url>
cd digit-recognizer
```

### 2. 训练Go CNN模型

```bash
cd go-cnn
go run main.go train
```

### 3. 推理预测

```bash
cd go-cnn
go run main.go inference
```

### 4. 编译可执行文件

```bash
cd go-cnn
go build -o mnist-cnn main.go train.go inference.go
./mnist-cnn train    # 训练模式
./mnist-cnn inference # 推理模式
```

### 5. 验证实现正确性

运行Python验证脚本确保Go实现的正确性：
```bash
cd python_verify
python pytorch_cnn_comparison.py
python verify_convolutional_layer.py
python convolution_backward_verify.py
```

## 🔍 核心功能

### Go语言CNN实现

- **纯Go实现**: 不依赖外部深度学习库，完全使用Go标准库
- **LeNet-5架构**: 完整的CNN实现，包含卷积层、池化层、全连接层
- **自动微分**: 基于计算图的自动梯度计算系统
- **高效卷积**: 使用Im2Col算法实现高效的卷积运算
- **模型序列化**: JSON格式保存/加载模型参数和元数据
- **生产级特性**: 数值稳定性、内存优化、错误处理

### 核心模块

#### 计算图系统 (graph/)
- **自动微分**: 完整的前向和反向传播
- **卷积层**: 2D卷积，支持多通道、He初始化
- **池化层**: 最大池化和平均池化，支持梯度回传
- **全连接层**: 矩阵乘法+偏置，He权重初始化
- **参数管理**: 自动参数收集、梯度清零、参数更新

#### 激活函数 (activations/)
- **ReLU**: 修正线性单元，支持原地计算
- **Sigmoid**: S型激活函数，数值稳定实现
- **Softmax**: 多分类输出，数值稳定的批处理版本

#### 损失函数 (losses/)
- **交叉熵损失**: 支持one-hot和标量标签
- **Softmax交叉熵**: 数值稳定的组合实现
- **均方误差**: 回归任务损失函数
- **二元交叉熵**: 二分类损失函数

#### 优化器 (optimizers/)
- **SGD**: 随机梯度下降，可配置学习率

#### 矩阵运算 (matrix/)
- **基础运算**: 加减乘除、转置、reshape
- **卷积运算**: Im2Col/Col2Im高效实现
- **广播运算**: NumPy风格的张量广播
- **统计函数**: 按轴求和、平均、最大最小值
- **矩阵创建**: 零矩阵、随机矩阵(正态/均匀分布)

#### 数据处理 (data/)
- **MNIST加载器**: CSV格式数据读取
- **数据预处理**: 像素归一化(0-255 → 0-1)
- **批处理器**: 配置批大小的数据迭代器
- **数据集接口**: 通用数据集抽象

#### 推理引擎 (inference/)
- **批量推理**: 支持单样本和批量预测
- **性能评估**: 准确率、混淆矩阵、分类报告
- **置信度分析**: 预测置信度统计

## 📊 模型架构

### LeNet-5 CNN架构
```
输入层: 784 (28×28 MNIST图像展平)
├── 卷积层1: 1→6通道, 5×5卷积核 → 24×24×6
├── 最大池化1: 2×2池化 → 12×12×6
├── 卷积层2: 6→16通道, 5×5卷积核 → 8×8×16  
├── 最大池化2: 2×2池化 → 4×4×16
├── 展平层: → 256特征
├── 全连接层1: 256→120 + ReLU
├── 全连接层2: 120→84 + ReLU
└── 输出层: 84→10 + Softmax交叉熵
```

### 训练配置
- **批大小**: 32
- **学习率**: 0.017
- **优化器**: SGD
- **权重初始化**: He初始化
- **激活函数**: ReLU + Softmax


## 📚 文档

- **[API参考手册](go-cnn/API-参考手册.md)**: Go CNN库的详细API文档
- **[开发路线图](roadmap-go-cnn.md)**: Go CNN实现的发展计划
- **[基础功能状态](go-cnn/foundation-status.md)**: 各模块实现状态

## 🔧 开发指南

### 添加新的网络层
1. 在 `go-cnn/graph/` 中实现新的操作函数
2. 实现前向传播和反向传播逻辑
3. 在 `python_verify/` 中添加验证脚本
4. 更新API文档和测试用例

### 扩展激活函数
1. 在 `go-cnn/activations/` 中实现新函数
2. 实现前向和反向传播方法
3. 添加对应的Python验证
4. 更新测试用例

### 优化器改进
1. 在 `go-cnn/optimizers/` 中实现新优化器
2. 实现 `Optimizer` 接口的 `Update` 方法
3. 与PyTorch/TensorFlow对比验证
4. 性能基准测试

### 模型架构扩展
1. 在 `go-cnn/models/` 中定义新模型
2. 使用 `graph` 包构建计算图
3. 实现模型构建器模式
4. 添加对应的训练和推理脚本

