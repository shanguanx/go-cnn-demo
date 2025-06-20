package graph

import (
	"github.com/user/go-cnn/matrix"
)

// Operation 定义计算图中的操作接口
type Operation interface {
	// Forward 执行前向传播
	Forward(inputs ...*matrix.Matrix) *matrix.Matrix

	// Backward 执行反向传播，返回对每个输入的梯度
	Backward(gradOutput *matrix.Matrix, inputs ...*Node) []*matrix.Matrix

	// Name 返回操作的名称
	Name() string
}

// AddOp 加法操作 - 用于偏置相加
type AddOp struct{}

func (op *AddOp) Forward(inputs ...*matrix.Matrix) *matrix.Matrix {
	if len(inputs) != 2 {
		panic("AddOp requires exactly 2 inputs")
	}
	return inputs[0].BroadcastAdd(inputs[1])
}

func (op *AddOp) Backward(gradOutput *matrix.Matrix, inputs ...*Node) []*matrix.Matrix {
	// 加法的导数都是1，但需要考虑广播
	b := inputs[1].Value

	gradA := gradOutput.Copy()
	gradB := gradOutput.Copy()

	// 如果形状不同，需要将梯度求和到原始形状
	if b.Rows != gradOutput.Rows || b.Cols != gradOutput.Cols {
		// b可能被广播了，需要将梯度求和回原始形状
		if b.Rows == 1 && b.Cols == gradOutput.Cols {
			// 行广播：(1, n) -> (m, n)，沿行求和
			gradB = gradOutput.SumAxis(0, true)
		} else if b.Cols == 1 && b.Rows == gradOutput.Rows {
			// 列广播：(m, 1) -> (m, n)，沿列求和
			gradB = gradOutput.SumAxis(1, true)
		} else if b.Rows == 1 && b.Cols == 1 {
			// 标量广播：(1, 1) -> (m, n)，全部求和
			sum := gradOutput.Sum()
			gradB = matrix.NewMatrixFromData([]float64{sum}, 1, 1)
		}
	}

	return []*matrix.Matrix{gradA, gradB}
}

func (op *AddOp) Name() string {
	return "Add"
}

// MulOp 矩阵乘法操作 - 用于全连接层
type MulOp struct{}

func (op *MulOp) Forward(inputs ...*matrix.Matrix) *matrix.Matrix {
	if len(inputs) != 2 {
		panic("MulOp requires exactly 2 inputs")
	}
	return inputs[0].Mul(inputs[1])
}

func (op *MulOp) Backward(gradOutput *matrix.Matrix, inputs ...*Node) []*matrix.Matrix {
	// 矩阵乘法的反向传播
	// 对A的梯度: grad_output @ B^T
	// 对B的梯度: A^T @ grad_output
	A := inputs[0].Value
	B := inputs[1].Value

	gradA := gradOutput.Mul(B.T())
	gradB := A.T().Mul(gradOutput)

	return []*matrix.Matrix{gradA, gradB}
}

func (op *MulOp) Name() string {
	return "MatMul"
}

// ReshapeOp 重塑操作 - 用于卷积层到全连接层的形状变换
type ReshapeOp struct {
	OriginalShape []int
	NewShape      []int
}

func (op *ReshapeOp) Forward(inputs ...*matrix.Matrix) *matrix.Matrix {
	if len(inputs) != 1 {
		panic("ReshapeOp requires exactly 1 input")
	}

	input := inputs[0]
	op.OriginalShape = []int{input.Rows, input.Cols}

	if len(op.NewShape) != 2 {
		panic("ReshapeOp only supports 2D shapes")
	}

	return input.Reshape(op.NewShape[0], op.NewShape[1])
}

func (op *ReshapeOp) Backward(gradOutput *matrix.Matrix, inputs ...*Node) []*matrix.Matrix {
	// 重塑的反向传播：将梯度重塑回原始形状
	grad := gradOutput.Reshape(op.OriginalShape[0], op.OriginalShape[1])
	return []*matrix.Matrix{grad}
}

func (op *ReshapeOp) Name() string {
	return "Reshape"
}
