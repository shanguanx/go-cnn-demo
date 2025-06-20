package activations

import (
	"math"

	"github.com/user/go-cnn/matrix"
)

// Exp 计算e的x次幂
func Exp(x float64) float64 {
	return math.Exp(x)
}

// Tanh 计算双曲正切
func Tanh(x float64) float64 {
	return math.Tanh(x)
}

// ReLU 激活函数：f(x) = max(0, x)
func ReLU(m *matrix.Matrix) *matrix.Matrix {
	result := matrix.Zeros(m.Rows, m.Cols)

	for i := 0; i < len(m.Data); i++ {
		if m.Data[i] > 0 {
			result.Data[i] = m.Data[i]
		}
		// 否则保持为0（已经初始化为0）
	}

	return result
}

// ReLUInPlace ReLU激活函数的就地版本
func ReLUInPlace(m *matrix.Matrix) {
	for i := 0; i < len(m.Data); i++ {
		if m.Data[i] <= 0 {
			m.Data[i] = 0
		}
	}
}

// ReLUDerivative ReLU的导数：f'(x) = 1 if x > 0, else 0
func ReLUDerivative(m *matrix.Matrix) *matrix.Matrix {
	result := matrix.Zeros(m.Rows, m.Cols)

	for i := 0; i < len(m.Data); i++ {
		if m.Data[i] > 0 {
			result.Data[i] = 1.0
		}
		// 否则保持为0
	}

	return result
}

// ReLUDerivativeInPlace ReLU导数的就地版本
func ReLUDerivativeInPlace(m *matrix.Matrix) {
	for i := 0; i < len(m.Data); i++ {
		if m.Data[i] > 0 {
			m.Data[i] = 1.0
		} else {
			m.Data[i] = 0.0
		}
	}
}

// Sigmoid 激活函数：f(x) = 1 / (1 + exp(-x))
func Sigmoid(m *matrix.Matrix) *matrix.Matrix {
	result := matrix.Zeros(m.Rows, m.Cols)

	for i := 0; i < len(m.Data); i++ {
		// 数值稳定性处理：避免exp溢出
		if m.Data[i] > 500 {
			result.Data[i] = 1.0
		} else if m.Data[i] < -500 {
			result.Data[i] = 0.0
		} else {
			result.Data[i] = 1.0 / (1.0 + math.Exp(-m.Data[i]))
		}
	}

	return result
}

// SigmoidInPlace Sigmoid激活函数的就地版本
func SigmoidInPlace(m *matrix.Matrix) {
	for i := 0; i < len(m.Data); i++ {
		if m.Data[i] > 500 {
			m.Data[i] = 1.0
		} else if m.Data[i] < -500 {
			m.Data[i] = 0.0
		} else {
			m.Data[i] = 1.0 / (1.0 + math.Exp(-m.Data[i]))
		}
	}
}

// SigmoidDerivative Sigmoid的导数：f'(x) = f(x) * (1 - f(x))
// 注意：输入应该是已经经过Sigmoid激活的值
func SigmoidDerivative(sigmoidOutput *matrix.Matrix) *matrix.Matrix {
	result := matrix.Zeros(sigmoidOutput.Rows, sigmoidOutput.Cols)

	for i := 0; i < len(sigmoidOutput.Data); i++ {
		s := sigmoidOutput.Data[i]
		result.Data[i] = s * (1.0 - s)
	}

	return result
}

// SigmoidDerivativeInPlace Sigmoid导数的就地版本
func SigmoidDerivativeInPlace(sigmoidOutput *matrix.Matrix) {
	for i := 0; i < len(sigmoidOutput.Data); i++ {
		s := sigmoidOutput.Data[i]
		sigmoidOutput.Data[i] = s * (1.0 - s)
	}
}

// Softmax 激活函数：用于多分类输出层
// 对于向量x，softmax(x_i) = exp(x_i) / sum(exp(x_j))
func Softmax(m *matrix.Matrix) *matrix.Matrix {
	if len(m.Shape) != 2 {
		panic("Softmax目前只支持2D矩阵 (batch_size, num_classes)")
	}

	batchSize := m.Shape[0]
	numClasses := m.Shape[1]
	result := matrix.Zeros(m.Rows, m.Cols)

	// 对每个样本分别计算softmax
	for batch := 0; batch < batchSize; batch++ {
		// 找到该样本的最大值，用于数值稳定性
		maxVal := m.At(batch, 0)
		for j := 1; j < numClasses; j++ {
			if val := m.At(batch, j); val > maxVal {
				maxVal = val
			}
		}

		// 计算exp(x_i - max)并求和
		var sum float64
		for j := 0; j < numClasses; j++ {
			expVal := math.Exp(m.At(batch, j) - maxVal)
			result.Set(batch, j, expVal)
			sum += expVal
		}

		// 归一化
		for j := 0; j < numClasses; j++ {
			result.Set(batch, j, result.At(batch, j)/sum)
		}
	}

	return result
}

// SoftmaxInPlace Softmax激活函数的就地版本
func SoftmaxInPlace(m *matrix.Matrix) {
	if len(m.Shape) != 2 {
		panic("Softmax目前只支持2D矩阵 (batch_size, num_classes)")
	}

	batchSize := m.Shape[0]
	numClasses := m.Shape[1]

	for batch := 0; batch < batchSize; batch++ {
		// 找到最大值
		maxVal := m.At(batch, 0)
		for j := 1; j < numClasses; j++ {
			if val := m.At(batch, j); val > maxVal {
				maxVal = val
			}
		}

		// 计算exp并求和
		var sum float64
		for j := 0; j < numClasses; j++ {
			expVal := math.Exp(m.At(batch, j) - maxVal)
			m.Set(batch, j, expVal)
			sum += expVal
		}

		// 归一化
		for j := 0; j < numClasses; j++ {
			m.Set(batch, j, m.At(batch, j)/sum)
		}
	}
}

// SoftmaxCrossEntropyDerivative Softmax与交叉熵损失结合的导数
// 当Softmax作为输出层，交叉熵作为损失函数时，导数简化为：predicted - true
func SoftmaxCrossEntropyDerivative(predicted, trueLabels *matrix.Matrix) *matrix.Matrix {
	if predicted.Rows != trueLabels.Rows || predicted.Cols != trueLabels.Cols {
		panic("预测值和真实标签的形状必须相同")
	}

	result := predicted.Copy()

	// 计算 predicted - true
	for i := 0; i < len(result.Data); i++ {
		result.Data[i] -= trueLabels.Data[i]
	}

	return result
}
