package graph

import (
	"github.com/user/go-cnn/activations"
	"github.com/user/go-cnn/matrix"
)

// ReLUOp ReLU激活函数操作 - CNN隐藏层的标准激活
type ReLUOp struct {
	mask *matrix.Matrix // 存储前向传播时的掩码，用于反向传播
}

func (op *ReLUOp) Forward(inputs ...*matrix.Matrix) *matrix.Matrix {
	if len(inputs) != 1 {
		panic("ReLUOp requires exactly 1 input")
	}

	input := inputs[0]
	output := input.Copy()

	// 创建掩码记录哪些位置被激活
	op.mask = matrix.Zeros(input.Rows, input.Cols)
	for i := 0; i < input.Rows; i++ {
		for j := 0; j < input.Cols; j++ {
			if input.At(i, j) > 0 {
				op.mask.Set(i, j, 1.0)
			} else {
				output.Set(i, j, 0.0)
			}
		}
	}

	return output
}

func (op *ReLUOp) Backward(gradOutput *matrix.Matrix, inputs ...*Node) []*matrix.Matrix {
	// ReLU的反向传播：只传递正值位置的梯度
	grad := gradOutput.HadamardProduct(op.mask)
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
	// Softmax的反向传播比较复杂
	// 对于批量输入，每个样本独立计算
	batchSize := op.output.Rows
	numClasses := op.output.Cols
	grad := matrix.Zeros(batchSize, numClasses)

	for b := 0; b < batchSize; b++ {
		// 获取当前样本的softmax输出
		softmaxOut := make([]float64, numClasses)
		gradOut := make([]float64, numClasses)

		for j := 0; j < numClasses; j++ {
			softmaxOut[j] = op.output.At(b, j)
			gradOut[j] = gradOutput.At(b, j)
		}

		// 计算雅可比矩阵与梯度的乘积
		for i := 0; i < numClasses; i++ {
			sum := 0.0
			for j := 0; j < numClasses; j++ {
				if i == j {
					sum += gradOut[j] * softmaxOut[i] * (1 - softmaxOut[j])
				} else {
					sum += gradOut[j] * (-softmaxOut[i] * softmaxOut[j])
				}
			}
			grad.Set(b, i, sum)
		}
	}

	return []*matrix.Matrix{grad}
}

func (op *SoftmaxOp) Name() string {
	return "Softmax"
}
