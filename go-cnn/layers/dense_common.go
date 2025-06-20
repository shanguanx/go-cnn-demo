package layers

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/user/go-cnn/matrix"
)

// DenseLayer 全连接层结构体 - 基于2D矩阵实现
type DenseLayer struct {
	// 网络参数
	Weights *matrix.Matrix // 权重矩阵：(input_features, output_features)
	Biases  *matrix.Matrix // 偏置向量：(1, output_features)

	// 层配置
	InputFeatures  int // 输入特征数
	OutputFeatures int // 输出特征数

	// 用于反向传播的缓存
	LastInput  *matrix.Matrix // 缓存的输入 (batch_size, input_features)
	LastOutput *matrix.Matrix // 缓存的输出 (batch_size, output_features)

	// 梯度缓存
	WeightGradients *matrix.Matrix // 权重梯度 (input_features, output_features)
	BiasGradients   *matrix.Matrix // 偏置梯度 (1, output_features)
}

// NewDenseLayer 创建新的全连接层
func NewDenseLayer(inputFeatures, outputFeatures int) *DenseLayer {
	if inputFeatures <= 0 || outputFeatures <= 0 {
		panic("全连接层的输入和输出特征数必须大于0")
	}

	// 权重矩阵：(input_features, output_features)
	weights := matrix.NewMatrix(inputFeatures, outputFeatures)

	// 偏置向量：(1, output_features)
	biases := matrix.NewMatrix(1, outputFeatures)

	// 梯度矩阵：与权重和偏置相同尺寸
	weightGradients := matrix.NewMatrix(inputFeatures, outputFeatures)
	biasGradients := matrix.NewMatrix(1, outputFeatures)

	layer := &DenseLayer{
		Weights:         weights,
		Biases:          biases,
		InputFeatures:   inputFeatures,
		OutputFeatures:  outputFeatures,
		WeightGradients: weightGradients,
		BiasGradients:   biasGradients,
	}

	// 初始化权重
	layer.initializeWeights()

	return layer
}

// initializeWeights 使用He初始化方法初始化权重
// 对于ReLU激活函数，使用He初始化：权重 ~ N(0, √(2/fan_in))
func (layer *DenseLayer) initializeWeights() {
	rand.Seed(time.Now().UnixNano())

	// He初始化标准差
	fanIn := float64(layer.InputFeatures)
	stddev := math.Sqrt(2.0 / fanIn)

	// 初始化权重
	rows, cols := layer.Weights.Rows, layer.Weights.Cols
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			// 生成正态分布随机数
			val := rand.NormFloat64() * stddev
			layer.Weights.Set(i, j, val)
		}
	}

	// 偏置初始化为0
	for j := 0; j < layer.OutputFeatures; j++ {
		layer.Biases.Set(0, j, 0.0)
	}
}

// initializeWeightsXavier 使用Xavier初始化方法初始化权重
// 对于Sigmoid/Tanh激活函数，使用Xavier初始化：权重 ~ N(0, √(1/fan_in))
func (layer *DenseLayer) initializeWeightsXavier() {
	rand.Seed(time.Now().UnixNano())

	// Xavier初始化标准差
	fanIn := float64(layer.InputFeatures)
	stddev := math.Sqrt(1.0 / fanIn)

	// 初始化权重
	rows, cols := layer.Weights.Rows, layer.Weights.Cols
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			// 生成正态分布随机数
			val := rand.NormFloat64() * stddev
			layer.Weights.Set(i, j, val)
		}
	}

	// 偏置初始化为0
	for j := 0; j < layer.OutputFeatures; j++ {
		layer.Biases.Set(0, j, 0.0)
	}
}

// GetWeights 获取权重矩阵
func (layer *DenseLayer) GetWeights() *matrix.Matrix {
	return layer.Weights
}

// GetBiases 获取偏置向量
func (layer *DenseLayer) GetBiases() *matrix.Matrix {
	return layer.Biases
}

// GetWeightGradients 获取权重梯度
func (layer *DenseLayer) GetWeightGradients() *matrix.Matrix {
	return layer.WeightGradients
}

// GetBiasGradients 获取偏置梯度
func (layer *DenseLayer) GetBiasGradients() *matrix.Matrix {
	return layer.BiasGradients
}

// GetInputFeatures 获取输入特征数
func (layer *DenseLayer) GetInputFeatures() int {
	return layer.InputFeatures
}

// GetOutputFeatures 获取输出特征数
func (layer *DenseLayer) GetOutputFeatures() int {
	return layer.OutputFeatures
}

// ZeroGradients 清零梯度
func (layer *DenseLayer) ZeroGradients() {
	// 清零权重梯度
	rows, cols := layer.WeightGradients.Rows, layer.WeightGradients.Cols
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			layer.WeightGradients.Set(i, j, 0.0)
		}
	}

	// 清零偏置梯度
	for j := 0; j < layer.OutputFeatures; j++ {
		layer.BiasGradients.Set(0, j, 0.0)
	}
}

// UpdateWeights 更新权重和偏置（使用简单的SGD）
func (layer *DenseLayer) UpdateWeights(learningRate float64) error {
	if learningRate <= 0 {
		return errors.New("学习率必须大于0")
	}

	// 更新权重：W = W - lr * dW
	rows, cols := layer.Weights.Rows, layer.Weights.Cols
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			oldWeight := layer.Weights.At(i, j)
			gradient := layer.WeightGradients.At(i, j)
			newWeight := oldWeight - learningRate*gradient
			layer.Weights.Set(i, j, newWeight)
		}
	}

	// 更新偏置：b = b - lr * db
	for j := 0; j < layer.OutputFeatures; j++ {
		oldBias := layer.Biases.At(0, j)
		gradient := layer.BiasGradients.At(0, j)
		newBias := oldBias - learningRate*gradient
		layer.Biases.Set(0, j, newBias)
	}

	return nil
}

// String 返回层的字符串表示
func (layer *DenseLayer) String() string {
	return fmt.Sprintf("DenseLayer(in_features=%d, out_features=%d)",
		layer.InputFeatures, layer.OutputFeatures)
}

// SetFixedWeights 设置固定的权重和偏置（用于测试）
func (layer *DenseLayer) SetFixedWeights() {
	// 设置固定的权重值
	// 使用简单的递增模式
	rows, cols := layer.Weights.Rows, layer.Weights.Cols
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			// 权重值为 (i*cols + j) * 0.1
			layer.Weights.Set(i, j, float64(i*cols+j)*0.1)
		}
	}

	// 设置固定的偏置值
	for j := 0; j < layer.OutputFeatures; j++ {
		layer.Biases.Set(0, j, float64(j)*0.5)
	}
}

// ClearCache 清理前向传播缓存
func (layer *DenseLayer) ClearCache() {
	layer.LastInput = nil
	layer.LastOutput = nil
}
