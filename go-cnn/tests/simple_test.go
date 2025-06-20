package tests

import (
	"fmt"
	"github.com/user/go-cnn/graph"
	"github.com/user/go-cnn/matrix"
	"math"
	"testing"
)

// TestSimpleAddition 测试简单的加法操作
func TestSimpleAddition(t *testing.T) {
	// 创建两个2x1的矩阵
	a := graph.NewParameter(matrix.NewMatrixFromData([]float64{2.0, 3.0}, 2, 1), "a")
	b := graph.NewParameter(matrix.NewMatrixFromData([]float64{1.0, 1.0}, 2, 1), "b")

	// c = a + b
	c := graph.Add(a, b)

	// 期望结果应该是 [3.0, 4.0]
	fmt.Printf("前向传播结果: %v\n", c.Value.Data)

	// 设置输出梯度为 [1.0, 1.0]
	c.Gradient = matrix.NewMatrixFromData([]float64{1.0, 1.0}, 2, 1)
	c.Backward()

	// 检查梯度
	fmt.Printf("a的梯度: %v\n", a.Gradient.Data)
	fmt.Printf("b的梯度: %v\n", b.Gradient.Data)

	// 对于加法，梯度应该直接传递，所以a和b的梯度都应该是[1.0, 1.0]
	for i := 0; i < 2; i++ {
		if math.Abs(a.Gradient.Data[i]-1.0) > 1e-6 {
			t.Errorf("a的梯度错误: 位置%d, 期望1.0, 得到%v", i, a.Gradient.Data[i])
		}
		if math.Abs(b.Gradient.Data[i]-1.0) > 1e-6 {
			t.Errorf("b的梯度错误: 位置%d, 期望1.0, 得到%v", i, b.Gradient.Data[i])
		}
	}
}

// TestSimpleMatMul 测试简单的矩阵乘法
func TestSimpleMatMul(t *testing.T) {
	// 创建简单的矩阵：A = [2], B = [3]
	// C = A * B = [6]
	a := graph.NewParameter(matrix.NewMatrixFromData([]float64{2.0}, 1, 1), "a")
	b := graph.NewParameter(matrix.NewMatrixFromData([]float64{3.0}, 1, 1), "b")

	c := graph.MatMul(a, b)

	fmt.Printf("前向传播结果: %v (期望: 6.0)\n", c.Value.Data[0])

	// 设置输出梯度为1.0
	c.Gradient = matrix.NewMatrixFromData([]float64{1.0}, 1, 1)
	c.Backward()

	fmt.Printf("a的梯度: %v (期望: 3.0)\n", a.Gradient.Data[0])
	fmt.Printf("b的梯度: %v (期望: 2.0)\n", b.Gradient.Data[0])

	// 对于 C = A * B：
	// dC/dA = B = 3.0
	// dC/dB = A = 2.0
	if math.Abs(a.Gradient.Data[0]-3.0) > 1e-6 {
		t.Errorf("a的梯度错误: 期望3.0, 得到%v", a.Gradient.Data[0])
	}
	if math.Abs(b.Gradient.Data[0]-2.0) > 1e-6 {
		t.Errorf("b的梯度错误: 期望2.0, 得到%v", b.Gradient.Data[0])
	}
}

// TestMatMul2x2 测试2x2矩阵乘法
func TestMatMul2x2(t *testing.T) {
	// A = [[1, 2]], B = [[3], [4]]
	// C = A * B = [[11]] (1*3 + 2*4 = 11)
	a := graph.NewParameter(matrix.NewMatrixFromData([]float64{1.0, 2.0}, 1, 2), "a")
	b := graph.NewParameter(matrix.NewMatrixFromData([]float64{3.0, 4.0}, 2, 1), "b")

	c := graph.MatMul(a, b)

	fmt.Printf("前向传播结果: %v (期望: 11.0)\n", c.Value.Data[0])

	// 设置输出梯度为1.0
	c.Gradient = matrix.NewMatrixFromData([]float64{1.0}, 1, 1)
	c.Backward()

	fmt.Printf("a的梯度: %v (期望: [3.0, 4.0])\n", a.Gradient.Data)
	fmt.Printf("b的梯度: %v (期望: [1.0, 2.0])\n", b.Gradient.Data)

	// 对于 C = A * B，其中 A 是 (1,2)，B 是 (2,1)：
	// dC/dA = B^T = [3, 4] (1x2)
	// dC/dB = A^T = [[1], [2]] (2x1)
	expectedGradA := []float64{3.0, 4.0}
	expectedGradB := []float64{1.0, 2.0}

	for i := 0; i < 2; i++ {
		if math.Abs(a.Gradient.Data[i]-expectedGradA[i]) > 1e-6 {
			t.Errorf("a的梯度错误: 位置%d, 期望%v, 得到%v", i, expectedGradA[i], a.Gradient.Data[i])
		}
		if math.Abs(b.Gradient.Data[i]-expectedGradB[i]) > 1e-6 {
			t.Errorf("b的梯度错误: 位置%d, 期望%v, 得到%v", i, expectedGradB[i], b.Gradient.Data[i])
		}
	}
}
