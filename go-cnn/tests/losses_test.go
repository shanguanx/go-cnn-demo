package tests

import (
	"math"
	"testing"

	"github.com/user/go-cnn/losses"
	"github.com/user/go-cnn/matrix"
)

func TestCrossEntropyLoss(t *testing.T) {
	tests := []struct {
		name        string
		predictions [][]float64
		targets     [][]float64
		expected    float64
		tolerance   float64
	}{
		{
			name: "完美预测one-hot",
			predictions: [][]float64{
				{0.7, 0.2, 0.1},
				{0.1, 0.6, 0.3},
			},
			targets: [][]float64{
				{1, 0, 0},
				{0, 1, 0},
			},

			expected:  0.4337502838523616, // 约 -ln(0.9+0.8)/2
			tolerance: 0.01,
		},
		{
			name: "标量索引目标",
			predictions: [][]float64{
				{0.9, 0.05, 0.05},
				{0.1, 0.8, 0.1},
			},
			targets: [][]float64{
				{0},
				{1},
			},
			expected:  0.164,
			tolerance: 0.01,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			predMatrix := matrix.NewMatrixFrom2D(tt.predictions)
			targetMatrix := matrix.NewMatrixFrom2D(tt.targets)

			t.Log(predMatrix.String())
			t.Log(targetMatrix.String())
			loss, err := losses.CrossEntropyLoss(predMatrix, targetMatrix)
			t.Log(loss)
			if err != nil {
				t.Fatalf("计算交叉熵损失时出错: %v", err)
			}

			if math.Abs(loss-tt.expected) > tt.tolerance {
				t.Errorf("交叉熵损失 = %v, 期望值 %v, 容差 %v", loss, tt.expected, tt.tolerance)
			}
		})
	}
}

func TestCrossEntropyLossDerivative(t *testing.T) {
	// 测试导数的形状和基本数值
	predictions := [][]float64{
		{0.7, 0.2, 0.1},
		{0.1, 0.6, 0.3},
	}
	targets := [][]float64{
		{1, 0, 0},
		{0, 1, 0},
	}

	predMatrix := matrix.NewMatrixFrom2D(predictions)
	targetMatrix := matrix.NewMatrixFrom2D(targets)

	grad, err := losses.CrossEntropyLossDerivative(predMatrix, targetMatrix)
	if err != nil {
		t.Fatalf("计算交叉熵导数时出错: %v", err)
	}

	// 检查形状
	if grad.Rows != predMatrix.Rows || grad.Cols != predMatrix.Cols {
		t.Errorf("梯度形状不正确: 得到 (%d, %d), 期望 (%d, %d)",
			grad.Rows, grad.Cols, predMatrix.Rows, predMatrix.Cols)
	}

	t.Log(grad.String())

	// 检查梯度的符号和大致范围
	// 对于正确类别，梯度应该是 (pred - 1) / batch_size，为负值
	// 对于错误类别，梯度应该是 pred / batch_size，为正值
	if grad.At(0, 0) >= 0 {
		t.Errorf("正确类别的梯度应该为负值，得到 %v", grad.At(0, 0))
	}
	if grad.At(0, 1) <= 0 {
		t.Errorf("错误类别的梯度应该为正值，得到 %v", grad.At(0, 1))
	}
}

func TestSoftmaxCrossEntropyLoss(t *testing.T) {
	// 测试Softmax+交叉熵的联合计算
	logits := [][]float64{
		{2.0, 1.0, 0.1},
		{0.5, 2.1, 0.3},
	}
	targets := [][]float64{
		{0}, // 第一个样本的正确类别是0
		{1}, // 第二个样本的正确类别是1
	}

	logitsMatrix := matrix.NewMatrixFrom2D(logits)
	targetMatrix := matrix.NewMatrixFrom2D(targets)

	t.Log(logitsMatrix.String())
	t.Log(targetMatrix.String())

	loss, softmaxProbs, err := losses.SoftmaxCrossEntropyLoss(logitsMatrix, targetMatrix)
	if err != nil {
		t.Fatalf("计算Softmax交叉熵损失时出错: %v", err)
	}

	// 检查loss合理性（应该是正值）
	if loss <= 0 {
		t.Errorf("损失值应该为正，得到 %v", loss)
	}

	// 检查softmax概率的和为1
	for i := 0; i < softmaxProbs.Rows; i++ {
		sum := 0.0
		for j := 0; j < softmaxProbs.Cols; j++ {
			sum += softmaxProbs.At(i, j)
		}
		if math.Abs(sum-1.0) > 1e-6 {
			t.Errorf("第%d行softmax概率和不为1: %v", i, sum)
		}
	}
	t.Log(softmaxProbs.String())
	t.Log(loss)

	// 检查softmax概率都为正值
	for i := 0; i < softmaxProbs.Rows; i++ {
		for j := 0; j < softmaxProbs.Cols; j++ {
			if softmaxProbs.At(i, j) <= 0 {
				t.Errorf("softmax概率应该为正值，在位置(%d,%d)得到 %v", i, j, softmaxProbs.At(i, j))
			}
		}
	}
}

func TestSoftmaxCrossEntropyLossDerivative(t *testing.T) {
	// 创建测试数据
	softmaxProbs := [][]float64{
		{0.7, 0.2, 0.1},
		{0.1, 0.8, 0.1},
	}
	targets := [][]float64{
		{0}, // 第一个样本的正确类别是0
		{1}, // 第二个样本的正确类别是1
	}

	probsMatrix := matrix.NewMatrixFrom2D(softmaxProbs)
	targetMatrix := matrix.NewMatrixFrom2D(targets)

	grad, err := losses.SoftmaxCrossEntropyLossDerivative(probsMatrix, targetMatrix)
	if err != nil {
		t.Fatalf("计算Softmax交叉熵导数时出错: %v", err)
	}

	// 检查形状
	if grad.Rows != probsMatrix.Rows || grad.Cols != probsMatrix.Cols {
		t.Errorf("梯度形状不正确")
	}

	t.Log(grad.String())

	// 检查数值：对于正确类别，梯度 = (prob - 1) / batch_size
	// 对于错误类别，梯度 = prob / batch_size
	batchSize := 2.0
	expectedGrad00 := (0.7 - 1.0) / batchSize // -0.15
	expectedGrad01 := 0.2 / batchSize         // 0.1
	expectedGrad11 := (0.8 - 1.0) / batchSize // -0.1

	tolerance := 1e-6
	if math.Abs(grad.At(0, 0)-expectedGrad00) > tolerance {
		t.Errorf("梯度(0,0)不正确: 得到 %v, 期望 %v", grad.At(0, 0), expectedGrad00)
	}
	if math.Abs(grad.At(0, 1)-expectedGrad01) > tolerance {
		t.Errorf("梯度(0,1)不正确: 得到 %v, 期望 %v", grad.At(0, 1), expectedGrad01)
	}
	if math.Abs(grad.At(1, 1)-expectedGrad11) > tolerance {
		t.Errorf("梯度(1,1)不正确: 得到 %v, 期望 %v", grad.At(1, 1), expectedGrad11)
	}
}

func TestMeanSquaredErrorLoss(t *testing.T) {
	tests := []struct {
		name        string
		predictions [][]float64
		targets     [][]float64
		expected    float64
		tolerance   float64
	}{
		{
			name: "完美预测",
			predictions: [][]float64{
				{1.0, 2.0},
				{3.0, 4.0},
			},
			targets: [][]float64{
				{1.0, 2.0},
				{3.0, 4.0},
			},
			expected:  0.0,
			tolerance: 1e-10,
		},
		{
			name: "简单MSE计算",
			predictions: [][]float64{
				{1.0, 2.0},
				{3.0, 4.0},
			},
			targets: [][]float64{
				{2.0, 3.0},
				{4.0, 5.0},
			},
			expected:  1.0, // ((1-2)^2 + (2-3)^2 + (3-4)^2 + (4-5)^2) / 4 = 4/4 = 1
			tolerance: 1e-10,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			predMatrix := matrix.NewMatrixFrom2D(tt.predictions)
			targetMatrix := matrix.NewMatrixFrom2D(tt.targets)

			loss, err := losses.MeanSquaredErrorLoss(predMatrix, targetMatrix)
			if err != nil {
				t.Fatalf("计算MSE损失时出错: %v", err)
			}

			if math.Abs(loss-tt.expected) > tt.tolerance {
				t.Errorf("MSE损失 = %v, 期望值 %v", loss, tt.expected)
			}
		})
	}
}

func TestMeanSquaredErrorLossDerivative(t *testing.T) {
	predictions := [][]float64{
		{1.0, 2.0},
		{3.0, 4.0},
	}
	targets := [][]float64{
		{2.0, 3.0},
		{4.0, 5.0},
	}

	predMatrix := matrix.NewMatrixFrom2D(predictions)
	targetMatrix := matrix.NewMatrixFrom2D(targets)

	grad, err := losses.MeanSquaredErrorLossDerivative(predMatrix, targetMatrix)
	if err != nil {
		t.Fatalf("计算MSE导数时出错: %v", err)
	}

	// 检查形状
	if grad.Rows != predMatrix.Rows || grad.Cols != predMatrix.Cols {
		t.Errorf("梯度形状不正确")
	}

	// 检查具体数值：MSE导数 = 2 * (pred - target) / n
	// n = 4, 所以系数是 2/4 = 0.5
	expected := [][]float64{
		{-0.5, -0.5}, // 2*(1-2)/4, 2*(2-3)/4
		{-0.5, -0.5}, // 2*(3-4)/4, 2*(4-5)/4
	}

	tolerance := 1e-10
	for i := 0; i < grad.Rows; i++ {
		for j := 0; j < grad.Cols; j++ {
			if math.Abs(grad.At(i, j)-expected[i][j]) > tolerance {
				t.Errorf("梯度(%d,%d)不正确: 得到 %v, 期望 %v", i, j, grad.At(i, j), expected[i][j])
			}
		}
	}
}

func TestBinaryCrossEntropyLoss(t *testing.T) {
	tests := []struct {
		name        string
		predictions [][]float64
		targets     [][]float64
		expected    float64
		tolerance   float64
	}{
		{
			name: "完美预测",
			predictions: [][]float64{
				{0.99999},
				{0.00001},
			},
			targets: [][]float64{
				{1.0},
				{0.0},
			},
			expected:  0.00001, // 接近0
			tolerance: 0.0001,
		},
		{
			name: "完全错误预测",
			predictions: [][]float64{
				{0.00001},
				{0.99999},
			},
			targets: [][]float64{
				{1.0},
				{0.0},
			},
			expected:  11.51, // 接近 -ln(0.00001)
			tolerance: 0.1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			predMatrix := matrix.NewMatrixFrom2D(tt.predictions)
			targetMatrix := matrix.NewMatrixFrom2D(tt.targets)

			loss, err := losses.BinaryCrossEntropyLoss(predMatrix, targetMatrix)
			if err != nil {
				t.Fatalf("计算二分类交叉熵损失时出错: %v", err)
			}

			if math.Abs(loss-tt.expected) > tt.tolerance {
				t.Errorf("二分类交叉熵损失 = %v, 期望值 %v, 容差 %v", loss, tt.expected, tt.tolerance)
			}
		})
	}
}

func TestBinaryCrossEntropyLossDerivative(t *testing.T) {
	predictions := [][]float64{
		{0.8},
		{0.3},
	}
	targets := [][]float64{
		{1.0},
		{0.0},
	}

	predMatrix := matrix.NewMatrixFrom2D(predictions)
	targetMatrix := matrix.NewMatrixFrom2D(targets)

	grad, err := losses.BinaryCrossEntropyLossDerivative(predMatrix, targetMatrix)
	if err != nil {
		t.Fatalf("计算二分类交叉熵导数时出错: %v", err)
	}

	// 检查形状
	if grad.Rows != predMatrix.Rows || grad.Cols != predMatrix.Cols {
		t.Errorf("梯度形状不正确")
	}

	// 检查导数符号：
	// 当target=1时，grad = (pred-1)/(pred*(1-pred))/batch_size，应该为负
	// 当target=0时，grad = pred/(pred*(1-pred))/batch_size，应该为正
	if grad.At(0, 0) >= 0 {
		t.Errorf("target=1时梯度应该为负，得到 %v", grad.At(0, 0))
	}
	if grad.At(1, 0) <= 0 {
		t.Errorf("target=0时梯度应该为正，得到 %v", grad.At(1, 0))
	}
}

// 测试错误情况
func TestLossErrorHandling(t *testing.T) {
	// 测试维度不匹配的错误
	pred1 := matrix.NewMatrix(2, 3)
	target1 := matrix.NewMatrix(3, 3) // 不同的batch size

	_, err := losses.CrossEntropyLoss(pred1, target1)
	if err == nil {
		t.Error("应该返回批量大小不匹配的错误")
	}

	// 测试目标索引超出范围
	pred2 := matrix.NewMatrix(1, 3)
	target2 := matrix.NewMatrixFrom2D([][]float64{{5}}) // 索引5超出范围[0,2]

	_, err = losses.CrossEntropyLoss(pred2, target2)
	if err == nil {
		t.Error("应该返回目标索引超出范围的错误")
	}
}

func TestNumericalStability(t *testing.T) {
	// 测试数值稳定性：极小概率值
	predictions := [][]float64{
		{1e-20, 1.0 - 1e-20},
		{1.0 - 1e-20, 1e-20},
	}
	targets := [][]float64{
		{1.0, 0.0},
		{0.0, 1.0},
	}

	predMatrix := matrix.NewMatrixFrom2D(predictions)
	targetMatrix := matrix.NewMatrixFrom2D(targets)

	// 交叉熵损失应该处理极小值而不返回无穷大
	loss, err := losses.CrossEntropyLoss(predMatrix, targetMatrix)
	if err != nil {
		t.Fatalf("数值稳定性测试失败: %v", err)
	}

	if math.IsInf(loss, 0) || math.IsNaN(loss) {
		t.Errorf("损失值应该是有限数值，得到 %v", loss)
	}

	// 测试梯度的数值稳定性
	grad, err := losses.CrossEntropyLossDerivative(predMatrix, targetMatrix)
	if err != nil {
		t.Fatalf("梯度数值稳定性测试失败: %v", err)
	}

	for i := 0; i < grad.Rows; i++ {
		for j := 0; j < grad.Cols; j++ {
			val := grad.At(i, j)
			if math.IsInf(val, 0) || math.IsNaN(val) {
				t.Errorf("梯度值应该是有限数值，在位置(%d,%d)得到 %v", i, j, val)
			}
		}
	}
}

// 基准测试
func BenchmarkCrossEntropyLoss(b *testing.B) {
	predictions := matrix.Random(100, 10, 0.0, 1.0) // 100个样本，10个类别
	targets := matrix.Zeros(100, 1)                 // 标量索引目标

	// 设置合理的目标值
	for i := 0; i < 100; i++ {
		targets.Set(i, 0, float64(i%10))
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := losses.CrossEntropyLoss(predictions, targets)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkSoftmaxCrossEntropyLoss(b *testing.B) {
	logits := matrix.Randn(100, 10, 0.0, 1.0) // 100个样本，10个类别
	targets := matrix.Zeros(100, 1)

	for i := 0; i < 100; i++ {
		targets.Set(i, 0, float64(i%10))
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _, err := losses.SoftmaxCrossEntropyLoss(logits, targets)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMSELoss(b *testing.B) {
	predictions := matrix.Random(100, 50, 0.0, 1.0)
	targets := matrix.Random(100, 50, 0.0, 1.0)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := losses.MeanSquaredErrorLoss(predictions, targets)
		if err != nil {
			b.Fatal(err)
		}
	}
}
