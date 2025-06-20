# Go-CNN API 参考手册

本文档整理了Go-CNN项目中已实现的重要函数和操作，用于在编写代码时快速查找，避免与Python常用库的参数混乱。

## 1. 矩阵操作 (matrix包)

### 1.1 矩阵创建
```go
// 基础创建
m := matrix.NewMatrix(rows, cols int) *Matrix                    // 创建零矩阵
m := matrix.NewMatrixFromData(data []float64, rows, cols int)    // 从数据创建
m := matrix.NewMatrixFrom2D(data [][]float64)                   // 从二维切片创建

// 特殊矩阵
matrix.Zeros(rows, cols int) *Matrix                            // 全零矩阵
matrix.Ones(rows, cols int) *Matrix                             // 全一矩阵
matrix.Eye(size int) *Matrix                                    // 单位矩阵
matrix.Random(rows, cols int, min, max float64) *Matrix         // 均匀随机矩阵
matrix.Randn(rows, cols int, mean, stddev float64) *Matrix      // 正态随机矩阵
```

### 1.2 矩阵访问和操作
```go
// 访问和设置
value := m.At(i, j int) float64                    // 获取元素 (边界检查)
m.Set(i, j int, val float64)                      // 设置元素 (边界检查)

// 形状操作
newM := m.Copy() *Matrix                          // 深拷贝
newM := m.Reshape(rows, cols int) *Matrix         // 重塑形状
newM := m.T() *Matrix                             // 转置
```

### 1.3 矩阵运算
```go
// 算术运算 (返回新矩阵)
result := m.Add(other *Matrix) *Matrix            // 矩阵加法
result := m.Sub(other *Matrix) *Matrix            // 矩阵减法  
result := m.Mul(other *Matrix) *Matrix            // 矩阵乘法
result := m.HadamardProduct(other *Matrix) *Matrix // 逐元素乘法

// 就地运算 (修改原矩阵)
m.AddInPlace(other *Matrix)                       // 就地加法
m.SubInPlace(other *Matrix)                       // 就地减法
m.HadamardProductInPlace(other *Matrix)           // 就地逐元素乘法

// 标量运算
result := m.Scale(scalar float64) *Matrix         // 标量乘法
result := m.AddScalar(scalar float64) *Matrix     // 标量加法
m.ScaleInPlace(scalar float64)                    // 就地标量乘法
m.AddScalarInPlace(scalar float64)                // 就地标量加法
```

### 1.4 统计和工具函数
```go
// 统计
sum := m.Sum() float64                            // 元素和
mean := m.Mean() float64                          // 平均值
max := m.Max() float64                            // 最大值
min := m.Min() float64                            // 最小值

// 函数应用
newM := m.Apply(fn func(float64) float64) *Matrix // 应用函数返回新矩阵
m.ApplyInPlace(fn func(float64) float64)          // 就地应用函数

// 比较和显示
equal := m.Equals(other *Matrix, tolerance float64) bool  // 矩阵比较
str := m.String() string                                  // 字符串表示
```

### 1.5 卷积相关操作
```go
// Im2Col 和 Col2Im 操作
cols := matrix.Im2ColWithChannels(input *Matrix, channels, kh, kw, strideH, strideW, padH, padW int) *Matrix
output := matrix.Col2ImWithChannels(cols *Matrix, channels, inputH, inputW, kh, kw, strideH, strideW, padH, padW int) *Matrix

// 广播操作
result := m.BroadcastAdd(other *Matrix) *Matrix   // 广播加法
result := m.SumAxis(axis int, keepDims bool) *Matrix  // 按轴求和
```

## 2. 激活函数 (activations包)

### 2.1 ReLU激活
```go
// ReLU: f(x) = max(0, x)
output := activations.ReLU(input *Matrix) *Matrix
activations.ReLUInPlace(input *Matrix)            // 就地版本

// ReLU导数: f'(x) = 1 if x > 0, else 0
grad := activations.ReLUDerivative(input *Matrix) *Matrix
activations.ReLUDerivativeInPlace(input *Matrix)  // 就地版本
```

### 2.2 Sigmoid激活
```go
// Sigmoid: f(x) = 1 / (1 + exp(-x))
output := activations.Sigmoid(input *Matrix) *Matrix
activations.SigmoidInPlace(input *Matrix)         // 就地版本

// Sigmoid导数: f'(x) = f(x) * (1 - f(x))
// 注意：输入应该是已经过Sigmoid的输出
grad := activations.SigmoidDerivative(sigmoidOutput *Matrix) *Matrix
activations.SigmoidDerivativeInPlace(sigmoidOutput *Matrix)  // 就地版本
```

### 2.3 Softmax激活
```go
// Softmax: 用于多分类输出层
// 支持批处理: (batch_size, num_classes)
output := activations.Softmax(input *Matrix) *Matrix
activations.SoftmaxInPlace(input *Matrix)         // 就地版本

// Softmax+交叉熵联合导数: predicted - true
grad := activations.SoftmaxCrossEntropyDerivative(predicted, trueLabels *Matrix) *Matrix
```

### 2.4 基础数学函数
```go
result := activations.Exp(x float64) float64      // e^x
result := activations.Tanh(x float64) float64     // tanh(x)
```

## 3. 损失函数 (losses包)

### 3.1 交叉熵损失
```go
// 标准交叉熵
loss, err := losses.CrossEntropyLoss(predictions, targets *Matrix) (float64, error)
grad, err := losses.CrossEntropyLossDerivative(predictions, targets *Matrix) (*Matrix, error)

// Softmax+交叉熵联合优化 (推荐用于多分类)
loss, probs, err := losses.SoftmaxCrossEntropyLoss(logits, targets *Matrix) (float64, *Matrix, error)
grad, err := losses.SoftmaxCrossEntropyLossDerivative(softmaxProbs, targets *Matrix) (*Matrix, error)
```

### 3.2 二分类交叉熵
```go
// 二分类交叉熵
loss, err := losses.BinaryCrossEntropyLoss(predictions, targets *Matrix) (float64, error)
grad, err := losses.BinaryCrossEntropyLossDerivative(predictions, targets *Matrix) (*Matrix, error)
```

### 3.3 均方误差损失
```go
// MSE损失
loss, err := losses.MeanSquaredErrorLoss(predictions, targets *Matrix) (float64, error)
grad, err := losses.MeanSquaredErrorLossDerivative(predictions, targets *Matrix) (*Matrix, error)
```

## 4. 计算图API (graph包)

### 4.1 基础操作节点
```go
// 创建输入和参数节点
input := graph.Input(data *Matrix, name string) *Node     // 输入数据节点
param := graph.Parameter(data *Matrix, name string) *Node // 参数节点(需要梯度)

// 基础运算
result := graph.Add(a, b *Node) *Node                     // 加法节点(用于偏置)
result := graph.MatMul(a, b *Node) *Node                  // 矩阵乘法节点
result := graph.Reshape(a *Node, rows, cols int) *Node    // 重塑节点
```

### 4.2 激活函数节点
```go
// 激活函数
result := graph.ReLU(input *Node) *Node                   // ReLU激活
result := graph.Softmax(input *Node) *Node                // Softmax激活
```

### 4.3 神经网络层
```go
// 全连接层
dense := graph.Dense(input *Node, outputSize int) *Node
// 测试用固定权重版本
dense := graph.DenseWithFixedWeights(input *Node, outputSize int) *Node

// 卷积层
conv := graph.Conv2d(input *Node, outChannels, kernelSize, stride, padding int, 
                    inputHeight, inputWidth, inChannels int) *Node
// 测试用固定权重版本
conv := graph.Conv2dWithFixedWeights(input *Node, outChannels, kernelSize, stride, padding int,
                                    inputHeight, inputWidth, inChannels int) *Node

// 池化层
pool := graph.MaxPool2d(input *Node, poolSize, stride int, 
                       inputHeight, inputWidth, channels int) *Node
```

### 4.4 损失函数节点
```go
// Softmax+交叉熵联合损失 (推荐用于多分类)
loss := graph.SoftmaxCrossEntropyLoss(logits, targets *Node, useScalarLabels bool) *Node
```

### 4.5 完整CNN架构
```go
// 构建手写数字识别CNN (LeNet风格)
// 输入: 28x28x1 -> Conv1(6@5x5) -> MaxPool -> Conv2(16@5x5) -> MaxPool -> FC(120) -> FC(84) -> FC(10)
output := graph.BuildDigitCNN(input *Node) *Node
```

## 5. 神经网络层操作详细参数

### 5.1 卷积层 (ConvOp)
```go
op := graph.NewConvOp(inChannels, outChannels, kernelSize, stride, padding int)

// 必须设置输入尺寸
err := op.SetInputSize(height, width int) error

// 前向传播
output := op.Forward(input *Matrix) *Matrix

// 获取参数和梯度
weights := op.GetWeights() *Matrix
biases := op.GetBiases() *Matrix  
weightGrads := op.GetWeightGradients() *Matrix
biasGrads := op.GetBiasGradients() *Matrix

// 梯度管理
op.ZeroGradients()                    // 清零梯度
op.SetFixedWeights()                  // 设置固定权重(测试用)
```

### 5.2 全连接层 (DenseOp)
```go
op := graph.NewDenseOp(inputFeatures, outputFeatures int)

// 前向传播: output = input * weights + biases
output := op.Forward(input *Matrix) *Matrix

// 获取参数和梯度 (同卷积层)
weights := op.GetWeights() *Matrix
biases := op.GetBiases() *Matrix
// ... 其他方法同卷积层
```

## 6. 重要数据格式说明

### 6.1 矩阵形状约定
```go
// 批处理数据格式
input: (batch_size, features)                    // 全连接层输入
conv_input: (batch_size, channels*height*width)  // 卷积层输入(flatten格式)
conv_output: (batch_size, out_channels*out_height*out_width) // 卷积层输出

// 权重格式
dense_weights: (input_features, output_features)     // 全连接层权重
conv_weights: (out_channels, in_channels*kh*kw)     // 卷积层权重
biases: (1, output_features) 或 (output_channels, 1) // 偏置
```

### 6.2 目标标签格式
```go
// 多分类目标
one_hot: (batch_size, num_classes)     // one-hot编码
scalar: (batch_size, 1)                // 标量索引 (0, 1, 2, ...)

// 二分类目标  
binary: (batch_size, 1)                // 0或1标签
```

## 7. 常见使用模式

### 7.1 构建简单CNN
```go
// 输入节点
input := graph.Input(inputData, "input")

// 卷积层 + ReLU + 池化
conv1 := graph.Conv2d(input, 6, 5, 1, 0, 28, 28, 1)
relu1 := graph.ReLU(conv1)
pool1 := graph.MaxPool2d(relu1, 2, 2, 24, 24, 6)

// 展平 + 全连接
flatten := graph.Reshape(pool1, 1, 6*12*12)
dense1 := graph.Dense(flatten, 120)
output := graph.ReLU(dense1)
```

### 7.2 计算损失和梯度
```go
// 创建目标节点
targets := graph.Input(targetData, "targets")

// 计算损失
loss := graph.SoftmaxCrossEntropyLoss(output, targets, true)

// 反向传播
loss.Backward()

// 获取梯度
for _, node := range trainableNodes {
    grad := node.Gradient
    // 更新参数...
}
```

## 8. 注意事项

1. **矩阵索引**: 使用`At(i, j)`和`Set(i, j, val)`进行安全的边界检查访问
2. **内存管理**: 优先使用就地操作(`*InPlace`)减少内存分配
3. **数值稳定性**: Softmax和损失函数已内置数值稳定性处理
4. **批处理**: 所有操作都支持批处理，第一维为batch维度
5. **梯度管理**: 训练前记得调用`ZeroGradients()`清零梯度
6. **权重初始化**: 卷积和全连接层默认使用He初始化
7. **测试**: 使用`*WithFixedWeights`版本进行确定性测试

## 9. 与Python库对比

| 功能 | Go-CNN | PyTorch/NumPy |
|------|--------|---------------|
| 矩阵创建 | `matrix.NewMatrix(r,c)` | `torch.zeros(r,c)` |
| 矩阵乘法 | `a.Mul(b)` | `torch.mm(a,b)` |
| 逐元素乘 | `a.HadamardProduct(b)` | `a * b` |
| 卷积层 | `Conv2d(in,out,k,s,p,h,w,c)` | `nn.Conv2d(c,out,k,s,p)` |
| 全连接 | `Dense(in, out)` | `nn.Linear(in,out)` |
| 损失函数 | `SoftmaxCrossEntropyLoss` | `nn.CrossEntropyLoss` |

这个参考手册涵盖了Go-CNN项目的主要API，可以作为快速查找和避免参数混乱的工具。