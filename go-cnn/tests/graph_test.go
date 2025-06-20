package tests

import (
	"math"
	"testing"

	"github.com/user/go-cnn/graph"
	"github.com/user/go-cnn/matrix"
)

// TestBasicOperations 测试基本操作的前向和反向传播
func TestBasicOperations(t *testing.T) {
	t.Run("Addition", func(t *testing.T) {
		// 创建输入
		a := graph.NewParameter(matrix.NewMatrixFromData([]float64{1, 2, 3, 4}, 2, 2), "a")
		b := graph.NewParameter(matrix.NewMatrixFromData([]float64{5, 6, 7, 8}, 2, 2), "b")

		// 前向传播
		c := graph.Add(a, b)

		// 检查前向传播结果
		expected := []float64{6, 8, 10, 12}
		for i := 0; i < 4; i++ {
			if math.Abs(c.Value.Data[i]-expected[i]) > 1e-6 {
				t.Errorf("前向传播错误: 期望 %v, 得到 %v", expected[i], c.Value.Data[i])
			}
		}
		t.Log("前向传播结果:")
		t.Log(c.Value)

		// 初始化输出梯度并执行反向传播
		c.Gradient = matrix.Ones(c.Value.Rows, c.Value.Cols)
		t.Log("c输出梯度:")
		t.Log(c.Gradient.String())
		c.Backward()
		t.Log("a输出梯度:")
		t.Log(a.Gradient.String())
		t.Log("b输出梯度:")
		t.Log(b.Gradient.String())

		// 检查梯度
		for i := 0; i < 4; i++ {
			if math.Abs(a.Gradient.Data[i]-1.0) > 1e-6 {
				t.Errorf("梯度错误: a的梯度应该全是1")
			}
			if math.Abs(b.Gradient.Data[i]-1.0) > 1e-6 {
				t.Errorf("梯度错误: b的梯度应该全是1")
			}
		}
	})

	t.Run("MatrixMultiplication", func(t *testing.T) {
		// 创建输入 - 使用更简单的形状
		a := graph.NewParameter(matrix.NewMatrixFromData([]float64{1, 2}, 1, 2), "a")
		b := graph.NewParameter(matrix.NewMatrixFromData([]float64{3, 4}, 2, 1), "b")

		// 前向传播: (1x2) * (2x1) = (1x1)
		c := graph.MatMul(a, b)

		// 检查前向传播结果
		expected := 1*3 + 2*4 // = 11
		if math.Abs(c.Value.At(0, 0)-float64(expected)) > 1e-6 {
			t.Errorf("矩阵乘法前向传播错误: 期望 %v, 得到 %v", expected, c.Value.At(0, 0))
		}

		// 初始化输出梯度并执行反向传播
		c.Gradient = matrix.Ones(c.Value.Rows, c.Value.Cols)
		c.Backward()

		// 手动检查梯度
		// dC/dA = B^T = [3, 4]
		expectedGradA := []float64{3, 4}
		for i := 0; i < 2; i++ {
			if math.Abs(a.Gradient.Data[i]-expectedGradA[i]) > 1e-6 {
				t.Errorf("a的梯度错误: 位置%d, 期望%v, 得到%v", i, expectedGradA[i], a.Gradient.Data[i])
			}
		}

		// dC/dB = A^T = [1, 2]^T
		expectedGradB := []float64{1, 2}
		for i := 0; i < 2; i++ {
			if math.Abs(b.Gradient.Data[i]-expectedGradB[i]) > 1e-6 {
				t.Errorf("b的梯度错误: 位置%d, 期望%v, 得到%v", i, expectedGradB[i], b.Gradient.Data[i])
			}
		}
	})

	t.Run("Reshape", func(t *testing.T) {
		// 创建输入
		x := graph.NewParameter(matrix.NewMatrixFromData([]float64{1, 2, 3, 4, 5, 6}, 2, 3), "x")

		// 前向传播: (2x3) -> (3x2)
		y := graph.Reshape(x, 3, 2)

		// 检查前向传播结果
		if y.Value.Rows != 3 || y.Value.Cols != 2 {
			t.Errorf("Reshape形状错误: 期望(3,2), 得到(%d,%d)", y.Value.Rows, y.Value.Cols)
		}

		// 数据应该保持不变，只是形状改变
		expectedData := []float64{1, 2, 3, 4, 5, 6}
		for i := 0; i < 6; i++ {
			if math.Abs(y.Value.Data[i]-expectedData[i]) > 1e-6 {
				t.Errorf("Reshape数据错误: 位置%d, 期望%v, 得到%v", i, expectedData[i], y.Value.Data[i])
			}
		}

		// 反向传播测试
		y.Gradient = matrix.Ones(y.Value.Rows, y.Value.Cols)
		y.Backward()

		// 梯度应该被重塑回原始形状
		if x.Gradient.Rows != 2 || x.Gradient.Cols != 3 {
			t.Errorf("Reshape梯度形状错误: 期望(2,3), 得到(%d,%d)", x.Gradient.Rows, x.Gradient.Cols)
		}
	})
}

// TestActivationFunctions 测试激活函数的前向和反向传播
func TestActivationFunctions(t *testing.T) {
	t.Run("ReLU", func(t *testing.T) {
		// 创建输入（包含正值和负值）
		x := graph.NewParameter(matrix.NewMatrixFromData([]float64{-2, -1, 0, 1, 2, 3}, 2, 3), "x")

		// 前向传播
		y := graph.ReLU(x)

		// 检查前向传播结果
		expected := []float64{0, 0, 0, 1, 2, 3}
		for i := 0; i < 6; i++ {
			if math.Abs(y.Value.Data[i]-expected[i]) > 1e-6 {
				t.Errorf("ReLU前向传播错误: 位置%d, 期望 %v, 得到 %v",
					i, expected[i], y.Value.Data[i])
			}
		}

		// 初始化输出梯度并执行反向传播
		y.Gradient = matrix.Ones(y.Value.Rows, y.Value.Cols)
		y.Backward()

		// 检查梯度（负值位置梯度为0，正值位置梯度为1）
		expectedGrad := []float64{0, 0, 0, 1, 1, 1}
		for i := 0; i < 6; i++ {
			if math.Abs(x.Gradient.Data[i]-expectedGrad[i]) > 1e-6 {
				t.Errorf("ReLU梯度错误: 位置%d, 期望 %v, 得到 %v",
					i, expectedGrad[i], x.Gradient.Data[i])
			}
		}
	})

	t.Run("Softmax", func(t *testing.T) {
		// 创建输入 - 简单的2类分类
		x := graph.NewParameter(matrix.NewMatrixFromData([]float64{2.0, 1.0, 1.0, 3.0}, 2, 2), "x")

		// 前向传播
		softmaxOutput := graph.Softmax(x)

		// 验证Softmax前向传播结果 - 每行应该和为1
		for i := 0; i < 2; i++ {
			sum := 0.0
			for j := 0; j < 2; j++ {
				sum += softmaxOutput.Value.At(i, j)
			}
			if math.Abs(sum-1.0) > 1e-6 {
				t.Errorf("Softmax前向传播错误: 第%d行和应该为1, 得到 %f", i, sum)
			}
		}

		// 第一行: [2, 1] -> softmax([2,1]) = [e^2/(e^2+e^1), e^1/(e^2+e^1)]
		exp2, exp1 := math.Exp(2), math.Exp(1)
		expected00 := exp2 / (exp2 + exp1)
		expected01 := exp1 / (exp2 + exp1)

		if math.Abs(softmaxOutput.Value.At(0, 0)-expected00) > 1e-6 {
			t.Errorf("Softmax[0,0]错误: 期望 %f, 得到 %f", expected00, softmaxOutput.Value.At(0, 0))
		}
		if math.Abs(softmaxOutput.Value.At(0, 1)-expected01) > 1e-6 {
			t.Errorf("Softmax[0,1]错误: 期望 %f, 得到 %f", expected01, softmaxOutput.Value.At(0, 1))
		}

		t.Logf("Softmax输出第一行: [%f, %f]", softmaxOutput.Value.At(0, 0), softmaxOutput.Value.At(0, 1))
	})
}

// TestLossFunctions 测试损失函数的前向和反向传播
func TestLossFunctions(t *testing.T) {
	t.Run("SoftmaxCrossEntropy", func(t *testing.T) {
		// 创建logits和标签
		logits := graph.NewParameter(matrix.NewMatrixFromData(
			[]float64{2.0, 1.0, 0.1, 1.0, 3.0, 0.1}, 2, 3), "logits")

		// 使用标量标签
		labels := graph.NewConstant(matrix.NewMatrixFromData([]float64{0, 1}, 2, 1), "labels")

		// 计算损失
		loss := graph.SoftmaxCrossEntropyLoss(logits, labels, true)

		// 检查损失是否为正数
		if loss.Value.At(0, 0) <= 0 {
			t.Errorf("损失应该为正数, 得到 %v", loss.Value.At(0, 0))
		}

		// 反向传播测试
		loss.Backward()

		// 检查损失合理性
		if loss.Value.At(0, 0) <= 0 {
			t.Errorf("损失应该为正数, 得到 %v", loss.Value.At(0, 0))
		}

		t.Logf("损失值: %f", loss.Value.At(0, 0))
		t.Logf("logits梯度形状: (%d, %d)", logits.Gradient.Rows, logits.Gradient.Cols)
	})
}

// TestCNNLayers 测试CNN层的基本功能
func TestCNNLayers(t *testing.T) {
	t.Run("DenseLayer", func(t *testing.T) {
		// 创建输入 (batch_size=2, input_size=3)
		input := graph.NewConstant(matrix.NewMatrixFromData(
			[]float64{1, 2, 3, 4, 5, 6}, 2, 3), "input")

		// 创建全连接层节点 (3 -> 2)
		dense := graph.Dense(input, 2)

		// 检查输出形状
		if dense.Value.Rows != 2 || dense.Value.Cols != 2 {
			t.Errorf("Dense层输出形状错误: 期望(2,2), 得到(%d,%d)",
				dense.Value.Rows, dense.Value.Cols)
		}

		// 反向传播测试
		dense.Gradient = matrix.Ones(dense.Value.Rows, dense.Value.Cols)
		dense.Backward()

		t.Logf("Dense层输出形状: (%d, %d)", dense.Value.Rows, dense.Value.Cols)
	})
}

// TestComplexComputation 测试复杂计算图 - 简单的手写数字识别网络
func TestComplexComputation(t *testing.T) {
	// 创建一个简单的两层神经网络
	// 输入 -> 全连接层 -> ReLU -> 全连接层 -> SoftmaxCrossEntropy损失

	// 输入数据 (2个样本, 3个特征)
	x := graph.NewConstant(matrix.NewMatrixFromData(
		[]float64{1, 2, 3, 4, 5, 6}, 2, 3), "input")

	// 第一层参数
	w1 := graph.NewParameter(matrix.Randn(3, 4, 0.0, 0.1), "w1")
	b1 := graph.NewParameter(matrix.Zeros(1, 4), "b1")

	// 第二层参数
	w2 := graph.NewParameter(matrix.Randn(4, 2, 0.0, 0.1), "w2")
	b2 := graph.NewParameter(matrix.Zeros(1, 2), "b2")

	// 标签 (2个样本, 类别0和1)
	labels := graph.NewConstant(matrix.NewMatrixFromData([]float64{0, 1}, 2, 1), "labels")

	// 构建计算图
	computeNetwork := func() *graph.Node {
		// 第一层: X @ W1 + B1
		z1 := graph.Add(graph.MatMul(x, w1), b1)
		a1 := graph.ReLU(z1)

		// 第二层: A1 @ W2 + B2
		z2 := graph.Add(graph.MatMul(a1, w2), b2)

		// 损失
		loss := graph.SoftmaxCrossEntropyLoss(z2, labels, true)
		return loss
	}

	// 前向传播
	loss := computeNetwork()

	// 检查损失合理性
	if loss.Value.At(0, 0) <= 0 {
		t.Errorf("损失应该为正数, 得到 %v", loss.Value.At(0, 0))
	}

	// 反向传播
	loss.Backward()

	// 检查损失合理性
	if loss.Value.At(0, 0) <= 0 {
		t.Errorf("损失应该为正数, 得到 %v", loss.Value.At(0, 0))
	}

	t.Logf("最终损失: %f", loss.Value.At(0, 0))
}

// TestGradientAccumulation 测试梯度累积
func TestGradientAccumulation(t *testing.T) {
	// 创建参数
	x := graph.NewParameter(matrix.NewMatrixFromData([]float64{1, 2}, 1, 2), "x")

	// 第一次前向+反向传播
	y1 := graph.MatMul(x, graph.NewConstant(matrix.NewMatrixFromData([]float64{2, 0, 0, 2}, 2, 2), "w1"))
	y1.Gradient = matrix.Ones(y1.Value.Rows, y1.Value.Cols)
	y1.Backward()

	// 保存第一次梯度
	firstGrad := x.Gradient.Copy()

	// 第二次前向+反向传播（不清零梯度）
	y2 := graph.MatMul(x, graph.NewConstant(matrix.NewMatrixFromData([]float64{1, 0, 0, 1}, 2, 2), "w2"))
	y2.Gradient = matrix.Ones(y2.Value.Rows, y2.Value.Cols)
	y2.Backward()

	// 检查梯度累积
	for i := 0; i < 2; i++ {
		expected := firstGrad.Data[i] + 1.0 // 第二次的梯度应该是1
		if math.Abs(x.Gradient.Data[i]-expected) > 1e-6 {
			t.Errorf("梯度累积错误: 位置%d, 期望%v, 得到%v",
				i, expected, x.Gradient.Data[i])
		}
	}

	// 清零梯度后再次计算
	x.ZeroGrad()
	if x.Gradient != nil {
		t.Errorf("清零后梯度应该为nil")
	}
}
