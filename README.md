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
git clone https://github.com/shanguanx/go-cnn-demo.git
cd go-cnn-demo
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
./mnist-cnn
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
- **模块化设计**: 分离的层、激活函数、损失函数、优化器
- **矩阵运算**: 自定义矩阵运算库，支持高效的数值计算
- **计算图**: 支持前向和反向传播的完整计算图
- **模型序列化**: JSON格式保存/加载模型
- **并发优化**: 利用Go的Goroutine进行并行计算

### 核心模块

#### 网络层 (layers/)
- **卷积层**: 支持多种卷积操作和填充策略
- **池化层**: 最大池化和平均池化
- **全连接层**: 标准的全连接神经网络层
- **展平层**: 将多维张量展平为一维向量

#### 激活函数 (activations/)
- **ReLU**: 修正线性单元
- **Sigmoid**: S型激活函数
- **Tanh**: 双曲正切函数
- **Softmax**: 多分类输出激活函数

#### 损失函数 (losses/)
- **交叉熵**: 分类问题的标准损失函数
- **均方误差**: 回归问题的损失函数

#### 优化器 (optimizers/)
- **SGD**: 随机梯度下降
- **Adam**: 自适应矩估计优化器
- **RMSprop**: 均方根传播优化器

#### 矩阵运算 (matrix/)
- **基础运算**: 加法、减法、乘法、除法
- **高级运算**: 矩阵乘法、转置、求逆
- **广播**: 支持不同维度张量的广播操作

## 📊 模型架构

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


## 📚 文档

- **[API参考手册](go-cnn/API-参考手册.md)**: Go CNN库的详细API文档
- **[开发路线图](roadmap-go-cnn.md)**: Go CNN实现的发展计划
- **[基础功能状态](go-cnn/foundation-status.md)**: 各模块实现状态

## 🔧 开发指南

### 添加新的网络层
1. 在 `go-cnn/layers/` 中实现新层
2. 实现 `Layer` 接口
3. 在 `python_verify/` 中添加验证脚本
4. 更新API文档

### 扩展激活函数
1. 在 `go-cnn/activations/` 中实现新函数
2. 实现 `Activation` 接口
3. 添加对应的Python验证
4. 更新测试用例

### 优化器改进
1. 在 `go-cnn/optimizers/` 中实现新优化器
2. 实现 `Optimizer` 接口
3. 与PyTorch/TensorFlow对比验证
4. 性能基准测试

