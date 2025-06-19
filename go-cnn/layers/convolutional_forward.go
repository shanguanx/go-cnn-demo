package layers

import (
	"errors"
	"fmt"

	"github.com/user/go-cnn/matrix"
)

// Forward 前向传播
// 输入：(batch_size, in_channels * input_height * input_width)
// 输出：(batch_size, out_channels * output_height * output_width)
func (layer *ConvolutionalLayer) Forward(input *matrix.Matrix) (*matrix.Matrix, error) {
	if layer.InputHeight == 0 || layer.InputWidth == 0 {
		return nil, errors.New("请先调用SetInputSize设置输入尺寸")
	}

	batchSize := input.Rows
	expectedInputSize := layer.InChannels * layer.InputHeight * layer.InputWidth

	if input.Cols != expectedInputSize {
		return nil, fmt.Errorf("输入尺寸不匹配：期望 %d，实际 %d", expectedInputSize, input.Cols)
	}

	// 创建输出矩阵
	outputSize := layer.OutChannels * layer.OutputHeight * layer.OutputWidth
	output := matrix.NewMatrix(batchSize, outputSize)

	// 处理每个样本
	for b := 0; b < batchSize; b++ {
		// 提取单个样本并重塑为3D形式进行im2col
		sampleInput := layer.extractAndReshapeSample(input, b)

		// 使用im2col进行卷积计算
		sampleOutput, err := layer.convolveSample(sampleInput)
		if err != nil {
			return nil, fmt.Errorf("样本 %d 卷积失败: %v", b, err)
		}

		// 将结果复制到输出矩阵
		layer.copySampleToOutput(sampleOutput, output, b)
	}

	// 缓存用于反向传播
	layer.LastInput = input.Copy()
	layer.LastOutput = output.Copy()

	return output, nil
}

// extractAndReshapeSample 提取并重塑单个样本为Im2Col需要的格式
func (layer *ConvolutionalLayer) extractAndReshapeSample(input *matrix.Matrix, batchIndex int) *matrix.Matrix {
	// 创建Im2Col期望的格式：(channels*height) x width
	sample := matrix.NewMatrix(layer.InChannels*layer.InputHeight, layer.InputWidth)

	for c := 0; c < layer.InChannels; c++ {
		for h := 0; h < layer.InputHeight; h++ {
			for w := 0; w < layer.InputWidth; w++ {
				// 从批次输入中获取值
				inputIdx := c*layer.InputHeight*layer.InputWidth + h*layer.InputWidth + w
				value := input.At(batchIndex, inputIdx)

				// 设置到sample中
				sample.Set(c*layer.InputHeight+h, w, value)
			}
		}
	}

	return sample
}

// convolveSample 对单个样本进行卷积
func (layer *ConvolutionalLayer) convolveSample(input *matrix.Matrix) (*matrix.Matrix, error) {
	// 使用im2col转换输入
	inputCols := matrix.Im2ColWithChannels(input, layer.InChannels, layer.KernelSize, layer.KernelSize,
		layer.Stride, layer.Stride, layer.Padding, layer.Padding)

	// 执行矩阵乘法：weights @ input_cols
	convResult := layer.Weights.Mul(inputCols)

	// 添加偏置
	for outChannel := 0; outChannel < layer.OutChannels; outChannel++ {
		bias := layer.Biases.At(outChannel, 0)
		for col := 0; col < convResult.Cols; col++ {
			currentVal := convResult.At(outChannel, col)
			convResult.Set(outChannel, col, currentVal+bias)
		}
	}

	return convResult, nil
}

// copySampleToOutput 将样本结果复制到批次输出
func (layer *ConvolutionalLayer) copySampleToOutput(sampleOutput, batchOutput *matrix.Matrix, batchIndex int) {
	outputSize := layer.OutChannels * layer.OutputHeight * layer.OutputWidth
	for i := 0; i < outputSize; i++ {
		row := i / (layer.OutputHeight * layer.OutputWidth)
		col := i % (layer.OutputHeight * layer.OutputWidth)
		value := sampleOutput.At(row, col)
		batchOutput.Set(batchIndex, i, value)
	}
}
