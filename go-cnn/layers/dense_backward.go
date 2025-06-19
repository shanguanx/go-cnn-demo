package layers

import (
	"errors"
	"fmt"

	"github.com/user/go-cnn/matrix"
)

// Backward 全连接层反向传播
// 输入：gradOutput (batch_size, output_features) - 来自上层的梯度
// 输出：gradInput (batch_size, input_features) - 传递给下层的梯度
// 同时计算权重和偏置的梯度并累积到梯度缓存中
func (layer *DenseLayer) Backward(gradOutput *matrix.Matrix) (*matrix.Matrix, error) {
	// 验证输入梯度矩阵
	if gradOutput == nil {
		return nil, errors.New("输出梯度矩阵不能为nil")
	}

	// 检查前向传播缓存
	if layer.LastInput == nil {
		return nil, errors.New("必须先执行前向传播才能进行反向传播")
	}

	batchSize, outputFeatures := gradOutput.Rows, gradOutput.Cols
	if outputFeatures != layer.OutputFeatures {
		return nil, fmt.Errorf("输出梯度特征数不匹配：期望 %d，得到 %d",
			layer.OutputFeatures, outputFeatures)
	}

	if batchSize != layer.LastInput.Rows {
		return nil, fmt.Errorf("批次大小不匹配：期望 %d，得到 %d",
			layer.LastInput.Rows, batchSize)
	}

	// 1. 计算输入梯度：gradInput = gradOutput * weights^T
	// gradOutput: (batch_size, output_features)
	// weights^T: (output_features, input_features)
	// gradInput: (batch_size, input_features)
	weightsT := layer.Weights.T()
	gradInput := gradOutput.Mul(weightsT)

	// 2. 计算权重梯度：gradWeights = input^T * gradOutput
	// input^T: (input_features, batch_size)
	// gradOutput: (batch_size, output_features)
	// gradWeights: (input_features, output_features)
	err := layer.computeWeightGradients(gradOutput)
	if err != nil {
		return nil, fmt.Errorf("计算权重梯度失败：%v", err)
	}

	// 3. 计算偏置梯度：gradBiases = sum(gradOutput, axis=0)
	// gradOutput: (batch_size, output_features)
	// gradBiases: (1, output_features)
	err = layer.computeBiasGradients(gradOutput)
	if err != nil {
		return nil, fmt.Errorf("计算偏置梯度失败：%v", err)
	}

	return gradInput, nil
}

// computeWeightGradients 计算权重梯度
// gradWeights = input^T * gradOutput
func (layer *DenseLayer) computeWeightGradients(gradOutput *matrix.Matrix) error {
	// 获取转置的输入：input^T
	inputT := layer.LastInput.T()

	// 计算权重梯度：gradWeights = input^T * gradOutput
	// inputT: (input_features, batch_size)
	// gradOutput: (batch_size, output_features)
	// result: (input_features, output_features)
	gradWeights := inputT.Mul(gradOutput)

	// 累积到权重梯度缓存中（支持多次反向传播累积）
	layer.WeightGradients = layer.WeightGradients.Add(gradWeights)

	return nil
}

// computeBiasGradients 计算偏置梯度
// gradBiases = sum(gradOutput, axis=0) - 沿批次维度求和
func (layer *DenseLayer) computeBiasGradients(gradOutput *matrix.Matrix) error {
	batchSize := gradOutput.Rows
	outputFeatures := gradOutput.Cols

	// 沿批次维度求和：将(batch_size, output_features)压缩为(1, output_features)
	for f := 0; f < outputFeatures; f++ {
		gradSum := 0.0
		for b := 0; b < batchSize; b++ {
			gradSum += gradOutput.At(b, f)
		}

		// 累积到偏置梯度缓存中
		currentGrad := layer.BiasGradients.At(0, f)
		layer.BiasGradients.Set(0, f, currentGrad+gradSum)
	}

	return nil
}

// BackwardWithActivationDerivative 全连接层反向传播（带激活函数导数）
// 这是一个便利方法，先应用激活函数导数，再进行反向传播
func (layer *DenseLayer) BackwardWithActivationDerivative(gradOutput *matrix.Matrix,
	activationDerivative func(*matrix.Matrix) *matrix.Matrix) (*matrix.Matrix, error) {

	// 检查是否有缓存的输出用于计算激活函数导数
	if layer.LastOutput == nil {
		return nil, errors.New("缺少前向传播输出缓存，无法计算激活函数导数")
	}

	// 计算激活函数导数
	var modifiedGradOutput *matrix.Matrix
	if activationDerivative != nil {
		// 计算激活函数关于输出的导数
		activationGrad := activationDerivative(layer.LastOutput)

		// 链式法则：gradOutput = gradOutput * activationDerivative
		modifiedGradOutput = gradOutput.HadamardProduct(activationGrad)
	} else {
		modifiedGradOutput = gradOutput
	}

	// 执行标准反向传播
	return layer.Backward(modifiedGradOutput)
}

// GetInputGradientsFromCache 从缓存中获取输入梯度
// 这个方法用于调试和验证
func (layer *DenseLayer) GetInputGradientsFromCache() *matrix.Matrix {
	// 这个方法假设已经执行过Backward
	// 在实际实现中，我们可能需要存储输入梯度用于调试
	return nil // 暂不实现，根据需要可以添加缓存
}
