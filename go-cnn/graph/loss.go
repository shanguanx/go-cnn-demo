package graph

import (
	"fmt"
	"github.com/user/go-cnn/losses"
	"github.com/user/go-cnn/matrix"
)

// SoftmaxCrossEntropyLossOp Softmax和交叉熵的联合操作
// 这是多分类问题（如手写数字识别）的标准损失函数
type SoftmaxCrossEntropyLossOp struct {
	softmaxOutput   *matrix.Matrix
	useScalarLabels bool
}

func NewSoftmaxCrossEntropyLossOp(useScalarLabels bool) *SoftmaxCrossEntropyLossOp {
	return &SoftmaxCrossEntropyLossOp{useScalarLabels: useScalarLabels}
}

func (op *SoftmaxCrossEntropyLossOp) Forward(inputs ...*matrix.Matrix) *matrix.Matrix {
	if len(inputs) != 2 {
		panic("SoftmaxCrossEntropyLossOp requires exactly 2 inputs (logits and targets)")
	}

	logits := inputs[0]
	targets := inputs[1]

	loss, softmaxOut, err := losses.SoftmaxCrossEntropyLoss(logits, targets)
	if err != nil {
		panic(fmt.Sprintf("SoftmaxCrossEntropy loss error: %v", err))
	}
	op.softmaxOutput = softmaxOut

	return matrix.NewMatrixFromData([]float64{loss}, 1, 1)
}

func (op *SoftmaxCrossEntropyLossOp) Backward(gradOutput *matrix.Matrix, inputs ...*Node) []*matrix.Matrix {
	targets := inputs[1].Value

	// 使用高效的联合梯度计算
	gradLogits, err := losses.SoftmaxCrossEntropyLossDerivative(op.softmaxOutput, targets)
	if err != nil {
		panic(fmt.Sprintf("SoftmaxCrossEntropy loss derivative error: %v", err))
	}

	// 乘以来自上游的梯度
	if gradOutput.At(0, 0) != 1.0 {
		gradLogits.ScaleInPlace(gradOutput.At(0, 0))
	}

	return []*matrix.Matrix{gradLogits, nil}
}

func (op *SoftmaxCrossEntropyLossOp) Name() string {
	return "SoftmaxCrossEntropyLoss"
}
