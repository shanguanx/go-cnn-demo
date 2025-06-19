package layers

import (
	"fmt"
	"github.com/user/go-cnn/matrix"
)

// Backward 池化层反向传播
// 输入: dOutput - 输出梯度 [batch_size, channels*output_height*output_width]
// 输出: 输入梯度 [batch_size, channels*input_height*input_width]
func (p *PoolingLayer) Backward(dOutput *matrix.Matrix) (*matrix.Matrix, error) {
	if !p.inputSizeSet {
		return nil, fmt.Errorf("必须先调用SetInputSize设置输入尺寸")
	}

	if p.lastInput == nil {
		return nil, fmt.Errorf("必须先进行前向传播")
	}

	batchSize := dOutput.Rows
	expectedOutputSize := p.inputChannels * p.outputHeight * p.outputWidth

	// 验证输出梯度尺寸
	if dOutput.Cols != expectedOutputSize {
		return nil, fmt.Errorf("输出梯度尺寸不匹配: 期望%d, 实际%d", expectedOutputSize, dOutput.Cols)
	}

	// 创建输入梯度矩阵（初始化为0）
	inputSize := p.inputChannels * p.inputHeight * p.inputWidth
	dInput := matrix.NewMatrix(batchSize, inputSize)

	// 根据池化类型进行反向传播
	switch p.poolType {
	case MaxPooling:
		return p.backwardMaxPooling(dOutput, dInput, batchSize)
	case AveragePooling:
		return p.backwardAveragePooling(dOutput, dInput, batchSize)
	default:
		return nil, fmt.Errorf("不支持的池化类型: %v", p.poolType)
	}
}

// backwardMaxPooling MaxPooling反向传播实现
// MaxPooling的反向传播：只有产生最大值的位置才会接收梯度
func (p *PoolingLayer) backwardMaxPooling(dOutput, dInput *matrix.Matrix, batchSize int) (*matrix.Matrix, error) {
	if p.maxIndices == nil {
		return nil, fmt.Errorf("MaxPooling反向传播需要最大值位置信息")
	}

	// 遍历每个样本
	for b := 0; b < batchSize; b++ {
		// 遍历每个通道
		for c := 0; c < p.inputChannels; c++ {
			// 遍历输出的每个位置
			for outH := 0; outH < p.outputHeight; outH++ {
				for outW := 0; outW < p.outputWidth; outW++ {
					// 计算输出在2D矩阵中的索引
					outputIdx := c*p.outputHeight*p.outputWidth + outH*p.outputWidth + outW

					// 获取当前输出位置的梯度
					gradient := dOutput.At(b, outputIdx)

					// 获取最大值在输入中的位置索引
					maxIndex := int(p.maxIndices.At(b, outputIdx))

					// 将1D索引转换为2D坐标: index = h * width + w
					maxH := maxIndex / p.inputWidth
					maxW := maxIndex % p.inputWidth

					// 验证索引有效性
					if maxH >= 0 && maxH < p.inputHeight && maxW >= 0 && maxW < p.inputWidth {
						// 计算输入在2D矩阵中的索引
						inputIdx := c*p.inputHeight*p.inputWidth + maxH*p.inputWidth + maxW

						// 将梯度传递给产生最大值的位置
						currentGrad := dInput.At(b, inputIdx)
						dInput.Set(b, inputIdx, currentGrad+gradient)
					}
				}
			}
		}
	}

	return dInput, nil
}

// backwardAveragePooling AveragePooling反向传播实现
// AveragePooling的反向传播：梯度均匀分布到池化窗口中的每个位置
func (p *PoolingLayer) backwardAveragePooling(dOutput, dInput *matrix.Matrix, batchSize int) (*matrix.Matrix, error) {
	poolSize := float64(p.poolHeight * p.poolWidth)

	// 遍历每个样本
	for b := 0; b < batchSize; b++ {
		// 遍历每个通道
		for c := 0; c < p.inputChannels; c++ {
			// 遍历输出的每个位置
			for outH := 0; outH < p.outputHeight; outH++ {
				for outW := 0; outW < p.outputWidth; outW++ {
					// 计算输出在2D矩阵中的索引
					outputIdx := c*p.outputHeight*p.outputWidth + outH*p.outputWidth + outW

					// 获取当前输出位置的梯度
					gradient := dOutput.At(b, outputIdx)

					// 梯度平均分配到池化窗口中的每个位置
					avgGradient := gradient / poolSize

					// 计算池化窗口在输入中的位置
					startH := outH * p.strideH
					endH := startH + p.poolHeight
					startW := outW * p.strideW
					endW := startW + p.poolWidth

					// 将平均梯度分配到窗口中的每个位置
					for h := startH; h < endH; h++ {
						for w := startW; w < endW; w++ {
							if h < p.inputHeight && w < p.inputWidth {
								// 计算输入在2D矩阵中的索引
								inputIdx := c*p.inputHeight*p.inputWidth + h*p.inputWidth + w

								currentGrad := dInput.At(b, inputIdx)
								dInput.Set(b, inputIdx, currentGrad+avgGradient)
							}
						}
					}
				}
			}
		}
	}

	return dInput, nil
}

// ClearCache 清理前向传播缓存
func (p *PoolingLayer) ClearCache() {
	p.lastInput = nil
	p.maxIndices = nil
}
