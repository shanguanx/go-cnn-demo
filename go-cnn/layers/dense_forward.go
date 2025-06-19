package layers

import (
	"errors"
	"fmt"

	"github.com/user/go-cnn/matrix"
)

// Forward 全连接层前向传播
// 输入：input (batch_size, input_features)
// 输出：output (batch_size, output_features)
// 计算：output = input * weights + biases
func (layer *DenseLayer) Forward(input *matrix.Matrix) (*matrix.Matrix, error) {
	// 验证输入矩阵
	if input == nil {
		return nil, errors.New("输入矩阵不能为nil")
	}

	batchSize, inputFeatures := input.Rows, input.Cols
	if inputFeatures != layer.InputFeatures {
		return nil, fmt.Errorf("输入特征数不匹配：期望 %d，得到 %d",
			layer.InputFeatures, inputFeatures)
	}

	if batchSize <= 0 {
		return nil, errors.New("批次大小必须大于0")
	}

	// 缓存输入用于反向传播
	layer.LastInput = input.Copy()

	// 执行矩阵乘法：output = input * weights
	// input: (batch_size, input_features)
	// weights: (input_features, output_features)
	// result: (batch_size, output_features)
	matmulResult := input.Mul(layer.Weights)

	// 添加偏置：output = matmul_result + biases
	// matmulResult: (batch_size, output_features)
	// biases: (1, output_features)
	// 需要广播偏置到每个批次
	output := layer.addBiases(matmulResult)

	// 缓存输出用于反向传播
	layer.LastOutput = output.Copy()

	return output, nil
}

// addBiases 将偏置添加到矩阵乘法结果中
// 这是一个广播操作：将(1, output_features)的偏置添加到(batch_size, output_features)的结果中
func (layer *DenseLayer) addBiases(matmulResult *matrix.Matrix) *matrix.Matrix {
	batchSize := matmulResult.Rows
	outputFeatures := matmulResult.Cols

	// 创建输出矩阵
	output := matrix.NewMatrix(batchSize, outputFeatures)

	// 为每个批次添加偏置
	for b := 0; b < batchSize; b++ {
		for f := 0; f < outputFeatures; f++ {
			// 获取矩阵乘法结果
			matmulVal := matmulResult.At(b, f)
			// 获取偏置值
			biasVal := layer.Biases.At(0, f)
			// 计算最终输出
			finalVal := matmulVal + biasVal
			output.Set(b, f, finalVal)
		}
	}

	return output
}

// ForwardWithActivation 全连接层前向传播（带激活函数）
// 这是一个便利方法，将前向传播和激活函数结合
func (layer *DenseLayer) ForwardWithActivation(input *matrix.Matrix, activation func(*matrix.Matrix) *matrix.Matrix) (*matrix.Matrix, error) {
	// 执行前向传播
	output, err := layer.Forward(input)
	if err != nil {
		return nil, err
	}

	// 应用激活函数
	if activation != nil {
		output = activation(output)
	}

	return output, nil
}
