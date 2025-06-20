package layers

import (
	"github.com/user/go-cnn/matrix"
)

// Layer 定义神经网络层的通用接口
type Layer interface {
	// Forward 执行前向传播
	Forward(input *matrix.Matrix) (*matrix.Matrix, error)

	// Backward 执行反向传播，返回输入梯度
	Backward(gradOutput *matrix.Matrix) (*matrix.Matrix, error)

	// UpdateWeights 更新层的权重参数
	UpdateWeights(learningRate float64)

	// ZeroGradients 清零梯度
	ZeroGradients()

	// GetName 获取层的名称
	GetName() string

	// SetInputSize 设置输入大小（如果需要）
	SetInputSize(inputSize int) error

	// ClearCache 清理缓存
	ClearCache()
}

// ParameterLayer 有参数的层的接口
type ParameterLayer interface {
	Layer

	// GetWeights 获取权重
	GetWeights() *matrix.Matrix

	// GetBiases 获取偏置
	GetBiases() *matrix.Matrix

	// GetWeightGradients 获取权重梯度
	GetWeightGradients() *matrix.Matrix

	// GetBiasGradients 获取偏置梯度
	GetBiasGradients() *matrix.Matrix
}
