package graph

import (
	"fmt"
	"math"

	"github.com/user/go-cnn/matrix"
)

// PoolingType 池化类型枚举
type PoolingType int

const (
	MaxPooling PoolingType = iota
	AveragePooling
)

// PoolOp 池化层操作 - 统一实现，消除与layers的重复
type PoolOp struct {
	// 池化类型
	poolType PoolingType

	// 池化窗口参数
	poolHeight int // 池化窗口高度
	poolWidth  int // 池化窗口宽度
	strideH    int // 高度方向步长
	strideW    int // 宽度方向步长

	// 输入输出尺寸
	inputHeight   int // 输入高度
	inputWidth    int // 输入宽度
	inputChannels int // 输入通道数
	outputHeight  int // 输出高度
	outputWidth   int // 输出宽度

	// 反向传播缓存
	cachedInput *matrix.Matrix // 输入数据缓存
	maxIndices  *matrix.Matrix // MaxPooling时记录最大值位置（仅用于MaxPooling）

	// 是否已设置输入尺寸
	inputSizeSet bool
}

// NewMaxPoolOp 创建MaxPooling操作
func NewMaxPoolOp(poolHeight, poolWidth, strideH, strideW int) *PoolOp {
	if poolHeight <= 0 || poolWidth <= 0 || strideH <= 0 || strideW <= 0 {
		panic("池化层参数必须大于0")
	}

	return &PoolOp{
		poolType:     MaxPooling,
		poolHeight:   poolHeight,
		poolWidth:    poolWidth,
		strideH:      strideH,
		strideW:      strideW,
		inputSizeSet: false,
	}
}

// NewAveragePoolOp 创建AveragePooling操作
func NewAveragePoolOp(poolHeight, poolWidth, strideH, strideW int) *PoolOp {
	if poolHeight <= 0 || poolWidth <= 0 || strideH <= 0 || strideW <= 0 {
		panic("池化层参数必须大于0")
	}

	return &PoolOp{
		poolType:     AveragePooling,
		poolHeight:   poolHeight,
		poolWidth:    poolWidth,
		strideH:      strideH,
		strideW:      strideW,
		inputSizeSet: false,
	}
}

// SetInputSize 设置输入尺寸并计算输出尺寸
func (op *PoolOp) SetInputSize(height, width, channels int) error {
	if height <= 0 || width <= 0 || channels <= 0 {
		return fmt.Errorf("输入尺寸必须大于0: height=%d, width=%d, channels=%d", height, width, channels)
	}

	// 检查池化窗口是否超出输入尺寸
	if op.poolHeight > height || op.poolWidth > width {
		return fmt.Errorf("池化窗口尺寸不能超过输入尺寸: pool=(%d,%d), input=(%d,%d)",
			op.poolHeight, op.poolWidth, height, width)
	}

	op.inputHeight = height
	op.inputWidth = width
	op.inputChannels = channels

	// 计算输出尺寸
	op.outputHeight = (height-op.poolHeight)/op.strideH + 1
	op.outputWidth = (width-op.poolWidth)/op.strideW + 1

	// 验证输出尺寸是否有效
	if op.outputHeight <= 0 || op.outputWidth <= 0 {
		return fmt.Errorf("计算得到的输出尺寸无效: output=(%d,%d), 请检查步长设置",
			op.outputHeight, op.outputWidth)
	}

	op.inputSizeSet = true
	return nil
}

// Forward 前向传播
// 输入: input - 2D矩阵 [batch_size, channels*height*width]
// 输出: 2D矩阵 [batch_size, channels*output_height*output_width]
func (op *PoolOp) Forward(inputs ...*matrix.Matrix) *matrix.Matrix {
	if len(inputs) != 1 {
		panic("PoolOp requires exactly 1 input")
	}

	input := inputs[0]

	if !op.inputSizeSet {
		panic("必须先调用SetInputSize设置输入尺寸")
	}

	batchSize := input.Rows
	expectedInputSize := op.inputChannels * op.inputHeight * op.inputWidth

	// 验证输入尺寸
	if input.Cols != expectedInputSize {
		panic(fmt.Sprintf("输入尺寸不匹配: 期望%d, 实际%d", expectedInputSize, input.Cols))
	}

	// 创建输出矩阵
	outputSize := op.inputChannels * op.outputHeight * op.outputWidth
	output := matrix.NewMatrix(batchSize, outputSize)

	// 为MaxPooling创建最大值位置索引矩阵
	if op.poolType == MaxPooling {
		op.maxIndices = matrix.NewMatrix(batchSize, outputSize)
	}

	// 缓存输入用于反向传播
	op.cachedInput = input.Copy()

	// 根据池化类型进行处理
	switch op.poolType {
	case MaxPooling:
		op.forwardMaxPooling(input, output, batchSize)
	case AveragePooling:
		op.forwardAveragePooling(input, output, batchSize)
	default:
		panic(fmt.Sprintf("不支持的池化类型: %v", op.poolType))
	}

	return output
}

// forwardMaxPooling MaxPooling前向传播实现
func (op *PoolOp) forwardMaxPooling(input, output *matrix.Matrix, batchSize int) {
	// 遍历每个样本
	for b := 0; b < batchSize; b++ {
		// 遍历每个通道
		for c := 0; c < op.inputChannels; c++ {
			// 遍历输出的每个位置
			for outH := 0; outH < op.outputHeight; outH++ {
				for outW := 0; outW < op.outputWidth; outW++ {
					// 计算池化窗口在输入中的位置
					startH := outH * op.strideH
					endH := startH + op.poolHeight
					startW := outW * op.strideW
					endW := startW + op.poolWidth

					// 在池化窗口中寻找最大值
					maxVal := math.Inf(-1)
					maxH, maxW := startH, startW

					for h := startH; h < endH; h++ {
						for w := startW; w < endW; w++ {
							// 计算在2D矩阵中的索引
							inputIdx := c*op.inputHeight*op.inputWidth + h*op.inputWidth + w
							val := input.At(b, inputIdx)
							if val > maxVal {
								maxVal = val
								maxH, maxW = h, w
							}
						}
					}

					// 计算输出在2D矩阵中的索引
					outputIdx := c*op.outputHeight*op.outputWidth + outH*op.outputWidth + outW
					output.Set(b, outputIdx, maxVal)

					// 记录最大值位置索引（用于反向传播）
					maxIndex := float64(maxH*op.inputWidth + maxW)
					op.maxIndices.Set(b, outputIdx, maxIndex)
				}
			}
		}
	}
}

// forwardAveragePooling AveragePooling前向传播实现
func (op *PoolOp) forwardAveragePooling(input, output *matrix.Matrix, batchSize int) {
	poolSize := float64(op.poolHeight * op.poolWidth)

	// 遍历每个样本
	for b := 0; b < batchSize; b++ {
		// 遍历每个通道
		for c := 0; c < op.inputChannels; c++ {
			// 遍历输出的每个位置
			for outH := 0; outH < op.outputHeight; outH++ {
				for outW := 0; outW < op.outputWidth; outW++ {
					// 计算池化窗口在输入中的位置
					startH := outH * op.strideH
					endH := startH + op.poolHeight
					startW := outW * op.strideW
					endW := startW + op.poolWidth

					// 计算池化窗口中的平均值
					sum := 0.0
					for h := startH; h < endH; h++ {
						for w := startW; w < endW; w++ {
							// 计算在2D矩阵中的索引
							inputIdx := c*op.inputHeight*op.inputWidth + h*op.inputWidth + w
							sum += input.At(b, inputIdx)
						}
					}

					// 计算输出在2D矩阵中的索引
					outputIdx := c*op.outputHeight*op.outputWidth + outH*op.outputWidth + outW
					avgVal := sum / poolSize
					output.Set(b, outputIdx, avgVal)
				}
			}
		}
	}
}

// Backward 反向传播
// 输入: gradOutput - 输出梯度 [batch_size, channels*output_height*output_width]
// 输出: 输入梯度 [batch_size, channels*input_height*input_width]
func (op *PoolOp) Backward(gradOutput *matrix.Matrix, inputs ...*Node) []*matrix.Matrix {
	if !op.inputSizeSet {
		panic("必须先调用SetInputSize设置输入尺寸")
	}

	if op.cachedInput == nil {
		panic("必须先进行前向传播")
	}

	batchSize := gradOutput.Rows
	expectedOutputSize := op.inputChannels * op.outputHeight * op.outputWidth

	// 验证输出梯度尺寸
	if gradOutput.Cols != expectedOutputSize {
		panic(fmt.Sprintf("输出梯度尺寸不匹配: 期望%d, 实际%d", expectedOutputSize, gradOutput.Cols))
	}

	// 创建输入梯度矩阵（初始化为0）
	inputSize := op.inputChannels * op.inputHeight * op.inputWidth
	gradInput := matrix.NewMatrix(batchSize, inputSize)

	// 根据池化类型进行反向传播
	switch op.poolType {
	case MaxPooling:
		op.backwardMaxPooling(gradOutput, gradInput, batchSize)
	case AveragePooling:
		op.backwardAveragePooling(gradOutput, gradInput, batchSize)
	default:
		panic(fmt.Sprintf("不支持的池化类型: %v", op.poolType))
	}

	return []*matrix.Matrix{gradInput}
}

// backwardMaxPooling MaxPooling反向传播实现
// MaxPooling的反向传播：只有产生最大值的位置才会接收梯度
func (op *PoolOp) backwardMaxPooling(gradOutput, gradInput *matrix.Matrix, batchSize int) {
	if op.maxIndices == nil {
		panic("MaxPooling反向传播需要最大值位置信息")
	}

	// 遍历每个样本
	for b := 0; b < batchSize; b++ {
		// 遍历每个通道
		for c := 0; c < op.inputChannels; c++ {
			// 遍历输出的每个位置
			for outH := 0; outH < op.outputHeight; outH++ {
				for outW := 0; outW < op.outputWidth; outW++ {
					// 计算输出在2D矩阵中的索引
					outputIdx := c*op.outputHeight*op.outputWidth + outH*op.outputWidth + outW

					// 获取当前输出位置的梯度
					gradient := gradOutput.At(b, outputIdx)

					// 获取最大值在输入中的位置索引
					maxIndex := int(op.maxIndices.At(b, outputIdx))

					// 将1D索引转换为2D坐标: index = h * width + w
					maxH := maxIndex / op.inputWidth
					maxW := maxIndex % op.inputWidth

					// 验证索引有效性
					if maxH >= 0 && maxH < op.inputHeight && maxW >= 0 && maxW < op.inputWidth {
						// 计算输入在2D矩阵中的索引
						inputIdx := c*op.inputHeight*op.inputWidth + maxH*op.inputWidth + maxW

						// 将梯度传递给产生最大值的位置
						currentGrad := gradInput.At(b, inputIdx)
						gradInput.Set(b, inputIdx, currentGrad+gradient)
					}
				}
			}
		}
	}
}

// backwardAveragePooling AveragePooling反向传播实现
// AveragePooling的反向传播：梯度均匀分布到池化窗口中的每个位置
func (op *PoolOp) backwardAveragePooling(gradOutput, gradInput *matrix.Matrix, batchSize int) {
	poolSize := float64(op.poolHeight * op.poolWidth)

	// 遍历每个样本
	for b := 0; b < batchSize; b++ {
		// 遍历每个通道
		for c := 0; c < op.inputChannels; c++ {
			// 遍历输出的每个位置
			for outH := 0; outH < op.outputHeight; outH++ {
				for outW := 0; outW < op.outputWidth; outW++ {
					// 计算输出在2D矩阵中的索引
					outputIdx := c*op.outputHeight*op.outputWidth + outH*op.outputWidth + outW

					// 获取当前输出位置的梯度
					gradient := gradOutput.At(b, outputIdx)

					// 梯度平均分配到池化窗口中的每个位置
					avgGradient := gradient / poolSize

					// 计算池化窗口在输入中的位置
					startH := outH * op.strideH
					endH := startH + op.poolHeight
					startW := outW * op.strideW
					endW := startW + op.poolWidth

					// 将平均梯度分配到窗口中的每个位置
					for h := startH; h < endH; h++ {
						for w := startW; w < endW; w++ {
							if h < op.inputHeight && w < op.inputWidth {
								// 计算输入在2D矩阵中的索引
								inputIdx := c*op.inputHeight*op.inputWidth + h*op.inputWidth + w

								currentGrad := gradInput.At(b, inputIdx)
								gradInput.Set(b, inputIdx, currentGrad+avgGradient)
							}
						}
					}
				}
			}
		}
	}
}

// Name 操作名称
func (op *PoolOp) Name() string {
	typeStr := "Max"
	if op.poolType == AveragePooling {
		typeStr = "Avg"
	}
	return fmt.Sprintf("%sPool2d(%dx%d/%dx%d)", typeStr, op.poolHeight, op.poolWidth, op.strideH, op.strideW)
}

// GetOutputSize 获取输出尺寸
func (op *PoolOp) GetOutputSize() (height, width, channels int, err error) {
	if !op.inputSizeSet {
		return 0, 0, 0, fmt.Errorf("必须先调用SetInputSize设置输入尺寸")
	}
	return op.outputHeight, op.outputWidth, op.inputChannels, nil
}

// GetInputSize 获取输入尺寸
func (op *PoolOp) GetInputSize() (height, width, channels int, err error) {
	if !op.inputSizeSet {
		return 0, 0, 0, fmt.Errorf("必须先调用SetInputSize设置输入尺寸")
	}
	return op.inputHeight, op.inputWidth, op.inputChannels, nil
}

// GetPoolingParams 获取池化参数
func (op *PoolOp) GetPoolingParams() (poolHeight, poolWidth, strideH, strideW int) {
	return op.poolHeight, op.poolWidth, op.strideH, op.strideW
}

// GetPoolingType 获取池化类型
func (op *PoolOp) GetPoolingType() PoolingType {
	return op.poolType
}

// ClearCache 清理前向传播缓存
func (op *PoolOp) ClearCache() {
	op.cachedInput = nil
	op.maxIndices = nil
}
