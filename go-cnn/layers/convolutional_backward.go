package layers

import (
	"errors"
	"fmt"

	"github.com/user/go-cnn/matrix"
)

// Backward 反向传播
func (layer *ConvolutionalLayer) Backward(gradOutput *matrix.Matrix) (*matrix.Matrix, error) {
	if layer.LastInput == nil {
		return nil, errors.New("必须先调用Forward方法")
	}

	batchSize := gradOutput.Rows
	expectedOutputSize := layer.OutChannels * layer.OutputHeight * layer.OutputWidth

	if gradOutput.Cols != expectedOutputSize {
		return nil, fmt.Errorf("梯度输出尺寸不匹配：期望 %d，实际 %d", expectedOutputSize, gradOutput.Cols)
	}

	// 初始化梯度
	layer.WeightGradients = matrix.NewMatrix(layer.Weights.Rows, layer.Weights.Cols)
	layer.BiasGradients = matrix.NewMatrix(layer.Biases.Rows, layer.Biases.Cols)

	// 创建输入梯度
	inputSize := layer.InChannels * layer.InputHeight * layer.InputWidth
	gradInput := matrix.NewMatrix(batchSize, inputSize)

	// 处理每个样本
	for b := 0; b < batchSize; b++ {
		// 提取样本梯度和输入
		sampleGradOutput := layer.extractSampleGradient(gradOutput, b)
		sampleInput := layer.extractAndReshapeSample(layer.LastInput, b)

		// 计算权重梯度
		err := layer.computeWeightGradients(sampleInput, sampleGradOutput)
		if err != nil {
			return nil, fmt.Errorf("计算权重梯度失败: %v", err)
		}

		// 计算偏置梯度
		layer.computeBiasGradients(sampleGradOutput)

		// 计算输入梯度
		sampleGradInput, err := layer.computeInputGradients(sampleGradOutput)
		if err != nil {
			return nil, fmt.Errorf("计算输入梯度失败: %v", err)
		}

		// 将样本输入梯度复制到总梯度
		layer.copySampleGradientToTotal(sampleGradInput, gradInput, b)
	}

	return gradInput, nil
}

// extractSampleGradient 提取单个样本的梯度
func (layer *ConvolutionalLayer) extractSampleGradient(gradOutput *matrix.Matrix, batchIndex int) *matrix.Matrix {
	// 重塑为 (out_channels, output_height * output_width)
	sampleGrad := matrix.NewMatrix(layer.OutChannels, layer.OutputHeight*layer.OutputWidth)

	for c := 0; c < layer.OutChannels; c++ {
		for i := 0; i < layer.OutputHeight*layer.OutputWidth; i++ {
			outputIdx := c*layer.OutputHeight*layer.OutputWidth + i
			sampleGrad.Set(c, i, gradOutput.At(batchIndex, outputIdx))
		}
	}

	return sampleGrad
}

// computeWeightGradients 计算权重梯度
func (layer *ConvolutionalLayer) computeWeightGradients(input, gradOutput *matrix.Matrix) error {
	// 使用im2col转换输入
	inputCols := matrix.Im2ColWithChannels(input, layer.InChannels, layer.KernelSize, layer.KernelSize,
		layer.Stride, layer.Stride, layer.Padding, layer.Padding)

	// 计算权重梯度：grad_output @ input_cols^T
	gradWeights := gradOutput.Mul(inputCols.T())

	// 累加梯度
	for i := 0; i < layer.WeightGradients.Rows; i++ {
		for j := 0; j < layer.WeightGradients.Cols; j++ {
			currentGrad := layer.WeightGradients.At(i, j)
			newGrad := gradWeights.At(i, j)
			layer.WeightGradients.Set(i, j, currentGrad+newGrad)
		}
	}

	return nil
}

// computeBiasGradients 计算偏置梯度
func (layer *ConvolutionalLayer) computeBiasGradients(gradOutput *matrix.Matrix) {
	// 偏置梯度是梯度输出在空间维度上的求和
	for c := 0; c < layer.OutChannels; c++ {
		gradSum := 0.0
		for i := 0; i < layer.OutputHeight*layer.OutputWidth; i++ {
			gradSum += gradOutput.At(c, i)
		}

		currentBiasGrad := layer.BiasGradients.At(c, 0)
		layer.BiasGradients.Set(c, 0, currentBiasGrad+gradSum)
	}
}

// computeInputGradients 计算输入梯度
func (layer *ConvolutionalLayer) computeInputGradients(gradOutput *matrix.Matrix) (*matrix.Matrix, error) {
	// 计算梯度：weights^T @ grad_output
	gradCols := layer.Weights.T().Mul(gradOutput)

	// 使用col2im将梯度转换回输入格式
	gradInput := matrix.Col2ImWithChannels(gradCols, layer.InChannels, layer.InputHeight, layer.InputWidth,
		layer.KernelSize, layer.KernelSize, layer.Stride, layer.Stride, layer.Padding, layer.Padding)

	return gradInput, nil
}

// copySampleGradientToTotal 将样本梯度复制到总梯度
func (layer *ConvolutionalLayer) copySampleGradientToTotal(sampleGrad, totalGrad *matrix.Matrix, batchIndex int) {
	// sampleGrad格式：(channels*inputH) x inputW
	// totalGrad格式：(batchSize, channels*inputH*inputW)
	for c := 0; c < layer.InChannels; c++ {
		for h := 0; h < layer.InputHeight; h++ {
			for w := 0; w < layer.InputWidth; w++ {
				// 从sampleGrad获取值
				value := sampleGrad.At(c*layer.InputHeight+h, w)

				// 计算在totalGrad中的索引
				inputIdx := c*layer.InputHeight*layer.InputWidth + h*layer.InputWidth + w
				totalGrad.Set(batchIndex, inputIdx, value)
			}
		}
	}
}

// UpdateWeights 更新权重和偏置
func (layer *ConvolutionalLayer) UpdateWeights(learningRate float64) error {
	if layer.WeightGradients == nil || layer.BiasGradients == nil {
		return errors.New("梯度未计算，请先调用Backward方法")
	}

	// 更新权重
	for i := 0; i < layer.Weights.Rows; i++ {
		for j := 0; j < layer.Weights.Cols; j++ {
			currentWeight := layer.Weights.At(i, j)
			gradient := layer.WeightGradients.At(i, j)
			layer.Weights.Set(i, j, currentWeight-learningRate*gradient)
		}
	}

	// 更新偏置
	for i := 0; i < layer.Biases.Rows; i++ {
		for j := 0; j < layer.Biases.Cols; j++ {
			currentBias := layer.Biases.At(i, j)
			gradient := layer.BiasGradients.At(i, j)
			layer.Biases.Set(i, j, currentBias-learningRate*gradient)
		}
	}

	return nil
}

// ZeroGradients 清零梯度
func (layer *ConvolutionalLayer) ZeroGradients() {
	if layer.WeightGradients != nil {
		for i := 0; i < len(layer.WeightGradients.Data); i++ {
			layer.WeightGradients.Data[i] = 0.0
		}
	}

	if layer.BiasGradients != nil {
		for i := 0; i < len(layer.BiasGradients.Data); i++ {
			layer.BiasGradients.Data[i] = 0.0
		}
	}
}
