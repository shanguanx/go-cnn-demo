package graph

import (
	"github.com/user/go-cnn/activations"
	"github.com/user/go-cnn/matrix"
)

// ReLUOp ReLU激活函数操作 - CNN隐藏层的标准激活
type ReLUOp struct {
	input *matrix.Matrix // 存储前向传播时的输入，用于反向传播
}

func (op *ReLUOp) Forward(inputs ...*matrix.Matrix) *matrix.Matrix {
	if len(inputs) != 1 {
		panic("ReLUOp requires exactly 1 input")
	}

	op.input = inputs[0]
	return activations.ReLU(inputs[0])
}

func (op *ReLUOp) Backward(gradOutput *matrix.Matrix, inputs ...*Node) []*matrix.Matrix {
	// ReLU的反向传播：只传递正值位置的梯度
	mask := activations.ReLUDerivative(op.input)
	grad := gradOutput.HadamardProduct(mask)
	return []*matrix.Matrix{grad}
}

func (op *ReLUOp) Name() string {
	return "ReLU"
}

// SoftmaxOp Softmax激活函数操作 - 多分类输出层的标准激活
type SoftmaxOp struct {
	output *matrix.Matrix // 存储前向传播的输出
}

func (op *SoftmaxOp) Forward(inputs ...*matrix.Matrix) *matrix.Matrix {
	if len(inputs) != 1 {
		panic("SoftmaxOp requires exactly 1 input")
	}

	op.output = activations.Softmax(inputs[0])
	return op.output
}

func (op *SoftmaxOp) Backward(gradOutput *matrix.Matrix, inputs ...*Node) []*matrix.Matrix {
	// 使用activations包中的SoftmaxDerivative函数
	grad := activations.SoftmaxDerivative(op.output, gradOutput)
	return []*matrix.Matrix{grad}
}

func (op *SoftmaxOp) Name() string {
	return "Softmax"
}
