package tests

import (
	"math"
	"testing"

	"github.com/user/go-cnn/activations"
	"github.com/user/go-cnn/matrix"
)

//const tolerance = 1e-10

func TestReLU(t *testing.T) {
	// 测试基本ReLU功能
	input := matrix.NewMatrixFromData([]float64{-2, -1, 0, 1, 2}, 1, 5)
	expected := matrix.NewMatrixFromData([]float64{0, 0, 0, 1, 2}, 1, 5)

	result := activations.ReLU(input)
	t.Log(input.String())
	t.Log(result.String())

	if !result.Equals(expected, tolerance) {
		t.Errorf("ReLU测试失败: 期望 %v, 得到 %v", expected.Data, result.Data)
	}
}

func TestReLUInPlace(t *testing.T) {
	// 测试就地ReLU
	input := matrix.NewMatrixFromData([]float64{-2, -1, 0, 1, 2}, 1, 5)
	expected := matrix.NewMatrixFromData([]float64{0, 0, 0, 1, 2}, 1, 5)

	activations.ReLUInPlace(input)

	if !input.Equals(expected, tolerance) {
		t.Errorf("ReLUInPlace测试失败: 期望 %v, 得到 %v", expected.Data, input.Data)
	}
}

func TestReLUDerivative(t *testing.T) {
	// 测试ReLU导数
	input := matrix.NewMatrixFromData([]float64{-2, -1, 0, 1, 2}, 1, 5)
	expected := matrix.NewMatrixFromData([]float64{0, 0, 0, 1, 1}, 1, 5)

	result := activations.ReLUDerivative(input)
	t.Log(input.String())
	t.Log(result.String())

	if !result.Equals(expected, tolerance) {
		t.Errorf("ReLUDerivative测试失败: 期望 %v, 得到 %v", expected.Data, result.Data)
	}
}

func TestReLUDerivativeInPlace(t *testing.T) {
	// 测试就地ReLU导数
	input := matrix.NewMatrixFromData([]float64{-2, -1, 0, 1, 2}, 1, 5)
	expected := matrix.NewMatrixFromData([]float64{0, 0, 0, 1, 1}, 1, 5)

	activations.ReLUDerivativeInPlace(input)

	if !input.Equals(expected, tolerance) {
		t.Errorf("ReLUDerivativeInPlace测试失败: 期望 %v, 得到 %v", expected.Data, input.Data)
	}
}

func TestSigmoid(t *testing.T) {
	// 测试基本Sigmoid功能
	input := matrix.NewMatrixFromData([]float64{-2, -1, 0, 1, 2}, 1, 5)

	result := activations.Sigmoid(input)

	// 验证几个关键点
	if math.Abs(result.At(0, 2)-0.5) > tolerance {
		t.Errorf("Sigmoid(0)应该等于0.5, 得到 %v", result.At(0, 2))
	}

	t.Log(input.String())
	t.Log(result.String())

	// 验证sigmoid的值域在(0,1)
	for i := 0; i < len(result.Data); i++ {
		if result.Data[i] <= 0 || result.Data[i] >= 1 {
			t.Errorf("Sigmoid值应该在(0,1)范围内, 得到 %v", result.Data[i])
		}
	}

	// 验证单调性
	for i := 1; i < len(result.Data); i++ {
		if result.Data[i] <= result.Data[i-1] {
			t.Errorf("Sigmoid应该是单调递增的")
		}
	}
}

func TestSigmoidInPlace(t *testing.T) {
	// 测试就地Sigmoid
	input := matrix.NewMatrixFromData([]float64{0}, 1, 1)

	activations.SigmoidInPlace(input)

	if math.Abs(input.At(0, 0)-0.5) > tolerance {
		t.Errorf("Sigmoid(0)应该等于0.5, 得到 %v", input.At(0, 0))
	}
}

func TestSigmoidNumericalStability(t *testing.T) {
	// 测试数值稳定性
	input := matrix.NewMatrixFromData([]float64{-1000, 1000}, 1, 2)

	result := activations.Sigmoid(input)

	if result.At(0, 0) != 0.0 {
		t.Errorf("Sigmoid(-1000)应该约等于0, 得到 %v", result.At(0, 0))
	}

	if result.At(0, 1) != 1.0 {
		t.Errorf("Sigmoid(1000)应该约等于1, 得到 %v", result.At(0, 1))
	}
}

func TestSigmoidDerivative(t *testing.T) {
	// 测试Sigmoid导数
	// 对于sigmoid输出为0.5的情况，导数应该是0.25
	sigmoidOutput := matrix.NewMatrixFromData([]float64{0.5}, 1, 1)
	expected := matrix.NewMatrixFromData([]float64{0.25}, 1, 1)

	result := activations.SigmoidDerivative(sigmoidOutput)

	if !result.Equals(expected, tolerance) {
		t.Errorf("SigmoidDerivative测试失败: 期望 %v, 得到 %v", expected.Data, result.Data)
	}
}

func TestSigmoidDerivativeInPlace(t *testing.T) {
	// 测试就地Sigmoid导数
	sigmoidOutput := matrix.NewMatrixFromData([]float64{0.5}, 1, 1)
	expected := matrix.NewMatrixFromData([]float64{0.25}, 1, 1)

	activations.SigmoidDerivativeInPlace(sigmoidOutput)

	if !sigmoidOutput.Equals(expected, tolerance) {
		t.Errorf("SigmoidDerivativeInPlace测试失败: 期望 %v, 得到 %v", expected.Data, sigmoidOutput.Data)
	}
}

func TestSoftmax(t *testing.T) {
	// 测试基本Softmax功能
	input := matrix.NewMatrixFromData([]float64{1, 2, 3, 4, 5}, 1, 5)

	result := activations.Softmax(input)

	// 验证概率和为1
	sum := 0.0
	for j := 0; j < 5; j++ {
		sum += result.At(0, j)
	}

	if math.Abs(sum-1.0) > tolerance {
		t.Errorf("Softmax概率和应该等于1, 得到 %v", sum)
	}

	// 验证所有值都是正数
	for j := 0; j < 5; j++ {
		if result.At(0, j) <= 0 {
			t.Errorf("Softmax值应该大于0, 得到 %v", result.At(0, j))
		}
	}

	// 验证单调性（输入递增，输出也应该递增）
	for j := 1; j < 5; j++ {
		if result.At(0, j) <= result.At(0, j-1) {
			t.Errorf("Softmax应该保持单调性")
		}
	}
}

func TestSoftmaxInPlace(t *testing.T) {
	// 测试就地Softmax
	input := matrix.NewMatrixFromData([]float64{1, 2, 3}, 1, 3)

	activations.SoftmaxInPlace(input)

	// 验证概率和为1
	sum := 0.0
	for j := 0; j < 3; j++ {
		sum += input.At(0, j)
	}

	if math.Abs(sum-1.0) > tolerance {
		t.Errorf("SoftmaxInPlace概率和应该等于1, 得到 %v", sum)
	}
}

func TestSoftmaxBatch(t *testing.T) {
	// 测试批量Softmax
	input := matrix.NewMatrixFromData([]float64{
		1, 2, 3,
		4, 5, 6,
	}, 2, 3)

	result := activations.Softmax(input)

	// 验证每个样本的概率和都为1
	for batch := 0; batch < 2; batch++ {
		sum := 0.0
		for j := 0; j < 3; j++ {
			sum += result.At(batch, j)
		}

		if math.Abs(sum-1.0) > tolerance {
			t.Errorf("批次 %d 的Softmax概率和应该等于1, 得到 %v", batch, sum)
		}
	}
}

func TestSoftmaxNumericalStability(t *testing.T) {
	// 测试数值稳定性
	input := matrix.NewMatrixFromData([]float64{1000, 1001, 1002}, 1, 3)

	result := activations.Softmax(input)

	// 验证没有NaN或Inf
	for j := 0; j < 3; j++ {
		val := result.At(0, j)
		if math.IsNaN(val) || math.IsInf(val, 0) {
			t.Errorf("Softmax应该处理大数值输入, 得到 %v", val)
		}
	}

	// 验证概率和为1
	sum := 0.0
	for j := 0; j < 3; j++ {
		sum += result.At(0, j)
	}

	if math.Abs(sum-1.0) > tolerance {
		t.Errorf("大数值输入的Softmax概率和应该等于1, 得到 %v", sum)
	}
}

func TestSoftmaxPanic(t *testing.T) {
	// 跳过此测试，因为当前Matrix结构只支持2D
	t.Skip("当前Matrix结构只支持2D，跳过此测试")
}

func TestSoftmaxCrossEntropyDerivative(t *testing.T) {
	// 测试Softmax交叉熵导数
	predicted := matrix.NewMatrixFromData([]float64{0.1, 0.2, 0.7}, 1, 3)
	trueLabels := matrix.NewMatrixFromData([]float64{0, 0, 1}, 1, 3)
	expected := matrix.NewMatrixFromData([]float64{0.1, 0.2, -0.3}, 1, 3)

	result := activations.SoftmaxCrossEntropyDerivative(predicted, trueLabels)

	if !result.Equals(expected, tolerance) {
		t.Errorf("SoftmaxCrossEntropyDerivative测试失败: 期望 %v, 得到 %v", expected.Data, result.Data)
	}
}

func TestSoftmaxCrossEntropyDerivativePanic(t *testing.T) {
	// 测试形状不匹配应该panic
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("SoftmaxCrossEntropyDerivative应该对形状不匹配的输入panic")
		}
	}()

	predicted := matrix.NewMatrixFromData([]float64{0.1, 0.2, 0.7}, 1, 3)
	trueLabels := matrix.NewMatrixFromData([]float64{0, 1}, 1, 2)

	activations.SoftmaxCrossEntropyDerivative(predicted, trueLabels)
}

// 基准测试
func BenchmarkReLU(b *testing.B) {
	input := matrix.Random(100, 100, -1.0, 1.0)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		activations.ReLU(input)
	}
}

func BenchmarkReLUInPlace(b *testing.B) {
	for i := 0; i < b.N; i++ {
		input := matrix.Random(100, 100, -1.0, 1.0)
		activations.ReLUInPlace(input)
	}
}

func BenchmarkSigmoid(b *testing.B) {
	input := matrix.Random(100, 100, -1.0, 1.0)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		activations.Sigmoid(input)
	}
}

func BenchmarkSigmoidInPlace(b *testing.B) {
	for i := 0; i < b.N; i++ {
		input := matrix.Random(100, 100, -1.0, 1.0)
		activations.SigmoidInPlace(input)
	}
}

func BenchmarkSoftmax(b *testing.B) {
	input := matrix.Random(32, 10, -1.0, 1.0) // 批量大小32，类别数10

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		activations.Softmax(input)
	}
}

func BenchmarkSoftmaxInPlace(b *testing.B) {
	for i := 0; i < b.N; i++ {
		input := matrix.Random(32, 10, -1.0, 1.0)
		activations.SoftmaxInPlace(input)
	}
}
