package graph

import (
	"github.com/user/go-cnn/matrix"
)

// 基础操作API - 用于构建计算图

// Add 创建加法节点 - 用于偏置相加
func Add(a, b *Node) *Node {
	requiresGrad := a.RequiresGrad || b.RequiresGrad
	value := a.Value.BroadcastAdd(b.Value)
	return NewNode(value, requiresGrad, &AddOp{}, a, b)
}

// MatMul 创建矩阵乘法节点 - 用于全连接层
func MatMul(a, b *Node) *Node {
	requiresGrad := a.RequiresGrad || b.RequiresGrad
	value := a.Value.Mul(b.Value)
	return NewNode(value, requiresGrad, &MulOp{}, a, b)
}

// Reshape 创建重塑节点 - 用于卷积层到全连接层的形状变换
func Reshape(a *Node, rows, cols int) *Node {
	op := &ReshapeOp{NewShape: []int{rows, cols}}
	value := op.Forward(a.Value)
	return NewNode(value, a.RequiresGrad, op, a)
}

// 激活函数API

// ReLU 创建ReLU激活节点 - CNN隐藏层的标准激活
func ReLU(a *Node) *Node {
	op := &ReLUOp{}
	value := op.Forward(a.Value)
	return NewNode(value, a.RequiresGrad, op, a)
}

// Softmax 创建Softmax激活节点 - 多分类输出层的标准激活
func Softmax(a *Node) *Node {
	op := &SoftmaxOp{}
	value := op.Forward(a.Value)
	return NewNode(value, a.RequiresGrad, op, a)
}

// CNN层API - 创建深度学习层节点

// Dense 创建全连接层节点
func Dense(input *Node, outputSize int) *Node {
	inputSize := input.Value.Cols
	op := NewDenseOp(inputSize, outputSize)
	output := op.Forward(input.Value)
	// 全连接层总是需要梯度，因为它有可训练的权重
	return NewNode(output, true, op, input)
}

// DenseWithFixedWeights 创建带固定权重的全连接层节点（用于测试）
func DenseWithFixedWeights(input *Node, outputSize int) *Node {
	inputSize := input.Value.Cols
	op := NewDenseOp(inputSize, outputSize)

	// 在前向传播之前设置固定权重
	op.SetFixedWeights()

	output := op.Forward(input.Value)
	// 全连接层总是需要梯度，因为它有可训练的权重
	return NewNode(output, true, op, input)
}

// Conv2d 创建卷积层节点
func Conv2d(input *Node, outChannels, kernelSize, stride, padding int, inputHeight, inputWidth, inChannels int) *Node {
	op := NewConvOp(inChannels, outChannels, kernelSize, stride, padding)

	// 设置输入尺寸 - 卷积层只需要高度和宽度
	err := op.SetInputSize(inputHeight, inputWidth)
	if err != nil {
		panic("Failed to set input size for Conv2d: " + err.Error())
	}

	output := op.Forward(input.Value)
	// 卷积层总是需要梯度，因为它有可训练的权重
	return NewNode(output, true, op, input)
}

// Conv2dWithFixedWeights 创建带固定权重的卷积层节点（用于测试）
func Conv2dWithFixedWeights(input *Node, outChannels, kernelSize, stride, padding int, inputHeight, inputWidth, inChannels int) *Node {
	op := NewConvOp(inChannels, outChannels, kernelSize, stride, padding)

	// 在前向传播之前设置固定权重
	op.SetFixedWeights()

	// 设置输入尺寸 - 卷积层只需要高度和宽度
	err := op.SetInputSize(inputHeight, inputWidth)
	if err != nil {
		panic("Failed to set input size for Conv2dWithFixedWeights: " + err.Error())
	}

	output := op.Forward(input.Value)
	// 卷积层总是需要梯度，因为它有可训练的权重
	return NewNode(output, true, op, input)
}

// MaxPool2d 创建最大池化层节点
func MaxPool2d(input *Node, poolSize, stride int, inputHeight, inputWidth, channels int) *Node {
	op := NewMaxPoolOp(poolSize, poolSize, stride, stride)

	// 设置输入尺寸
	err := op.SetInputSize(inputHeight, inputWidth, channels)
	if err != nil {
		panic("Failed to set input size for MaxPool2d: " + err.Error())
	}

	output := op.Forward(input.Value)
	return NewNode(output, input.RequiresGrad, op, input)
}

// 损失函数API

// SoftmaxCrossEntropyLoss 创建Softmax和交叉熵联合损失节点
// 这是手写数字识别等多分类问题的标准损失函数
func SoftmaxCrossEntropyLoss(logits, targets *Node, useScalarLabels bool) *Node {
	op := NewSoftmaxCrossEntropyLossOp(useScalarLabels)
	value := op.Forward(logits.Value, targets.Value)
	requiresGrad := logits.RequiresGrad
	return NewNode(value, requiresGrad, op, logits, targets)
}

// 便利函数 - 用于创建输入和参数节点

// Input 创建输入数据节点
func Input(data *matrix.Matrix, name string) *Node {
	return NewConstant(data, name)
}
