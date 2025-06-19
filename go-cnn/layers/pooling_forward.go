package layers

import (
	"fmt"
	"github.com/user/go-cnn/matrix"
	"math"
)

// Forward 池化层前向传播
// 输入: input - 2D矩阵 [batch_size, channels*height*width]
// 输出: 2D矩阵 [batch_size, channels*output_height*output_width]
func (p *PoolingLayer) Forward(input *matrix.Matrix) (*matrix.Matrix, error) {
	if !p.inputSizeSet {
		return nil, fmt.Errorf("必须先调用SetInputSize设置输入尺寸")
	}

	batchSize := input.Rows
	expectedInputSize := p.inputChannels * p.inputHeight * p.inputWidth

	// 验证输入尺寸
	if input.Cols != expectedInputSize {
		return nil, fmt.Errorf("输入尺寸不匹配: 期望%d, 实际%d", expectedInputSize, input.Cols)
	}

	// 创建输出矩阵
	outputSize := p.inputChannels * p.outputHeight * p.outputWidth
	output := matrix.NewMatrix(batchSize, outputSize)

	// 为MaxPooling创建最大值位置索引矩阵
	if p.poolType == MaxPooling {
		p.maxIndices = matrix.NewMatrix(batchSize, outputSize)
	}

	// 缓存输入用于反向传播
	p.lastInput = input.Copy()

	// 根据池化类型进行处理
	switch p.poolType {
	case MaxPooling:
		return p.forwardMaxPooling(input, output, batchSize)
	case AveragePooling:
		return p.forwardAveragePooling(input, output, batchSize)
	default:
		return nil, fmt.Errorf("不支持的池化类型: %v", p.poolType)
	}
}

// forwardMaxPooling MaxPooling前向传播实现
func (p *PoolingLayer) forwardMaxPooling(input, output *matrix.Matrix, batchSize int) (*matrix.Matrix, error) {
	// 遍历每个样本
	for b := 0; b < batchSize; b++ {
		// 遍历每个通道
		for c := 0; c < p.inputChannels; c++ {
			// 遍历输出的每个位置
			for outH := 0; outH < p.outputHeight; outH++ {
				for outW := 0; outW < p.outputWidth; outW++ {
					// 计算池化窗口在输入中的位置
					startH := outH * p.strideH
					endH := startH + p.poolHeight
					startW := outW * p.strideW
					endW := startW + p.poolWidth

					// 在池化窗口中寻找最大值
					maxVal := math.Inf(-1)
					maxH, maxW := startH, startW

					for h := startH; h < endH; h++ {
						for w := startW; w < endW; w++ {
							// 计算在2D矩阵中的索引
							inputIdx := c*p.inputHeight*p.inputWidth + h*p.inputWidth + w
							val := input.At(b, inputIdx)
							if val > maxVal {
								maxVal = val
								maxH, maxW = h, w
							}
						}
					}

					// 计算输出在2D矩阵中的索引
					outputIdx := c*p.outputHeight*p.outputWidth + outH*p.outputWidth + outW
					output.Set(b, outputIdx, maxVal)

					// 记录最大值位置索引（用于反向传播）
					maxIndex := float64(maxH*p.inputWidth + maxW)
					p.maxIndices.Set(b, outputIdx, maxIndex)
				}
			}
		}
	}

	return output, nil
}

// forwardAveragePooling AveragePooling前向传播实现
func (p *PoolingLayer) forwardAveragePooling(input, output *matrix.Matrix, batchSize int) (*matrix.Matrix, error) {
	poolSize := float64(p.poolHeight * p.poolWidth)

	// 遍历每个样本
	for b := 0; b < batchSize; b++ {
		// 遍历每个通道
		for c := 0; c < p.inputChannels; c++ {
			// 遍历输出的每个位置
			for outH := 0; outH < p.outputHeight; outH++ {
				for outW := 0; outW < p.outputWidth; outW++ {
					// 计算池化窗口在输入中的位置
					startH := outH * p.strideH
					endH := startH + p.poolHeight
					startW := outW * p.strideW
					endW := startW + p.poolWidth

					// 计算池化窗口中的平均值
					sum := 0.0
					for h := startH; h < endH; h++ {
						for w := startW; w < endW; w++ {
							// 计算在2D矩阵中的索引
							inputIdx := c*p.inputHeight*p.inputWidth + h*p.inputWidth + w
							sum += input.At(b, inputIdx)
						}
					}

					// 计算输出在2D矩阵中的索引
					outputIdx := c*p.outputHeight*p.outputWidth + outH*p.outputWidth + outW
					avgVal := sum / poolSize
					output.Set(b, outputIdx, avgVal)
				}
			}
		}
	}

	return output, nil
}
