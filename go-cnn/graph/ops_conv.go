package graph

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/user/go-cnn/matrix"
)

// ConvOp 卷积层操作 - 统一实现，消除与layers的重复
type ConvOp struct {
	// 层参数
	InChannels  int // 输入通道数
	OutChannels int // 输出通道数
	KernelSize  int // 卷积核大小（正方形）
	Stride      int // 步长
	Padding     int // 填充
	InputHeight int // 输入高度
	InputWidth  int // 输入宽度

	// 输出尺寸
	OutputHeight int
	OutputWidth  int

	// 权重和偏置 (作为值直接存储，不是节点)
	weights *matrix.Matrix // 权重矩阵：(out_channels, in_channels * kernel_size * kernel_size)
	biases  *matrix.Matrix // 偏置向量：(out_channels, 1)

	// 反向传播缓存
	cachedInput     *matrix.Matrix // 缓存的输入 (batch_size, in_channels * input_height * input_width)
	cachedInputCols *matrix.Matrix // 缓存的im2col结果
	cachedOutput    *matrix.Matrix // 缓存的输出 (batch_size, out_channels * output_height * output_width)
	weightGradients *matrix.Matrix // 权重梯度
	biasGradients   *matrix.Matrix // 偏置梯度
}

// NewConvOp 创建新的卷积层操作
func NewConvOp(inChannels, outChannels, kernelSize, stride, padding int) *ConvOp {
	// 权重矩阵：(out_channels, in_channels * kernel_size * kernel_size)
	weightCols := inChannels * kernelSize * kernelSize
	weights := matrix.NewMatrix(outChannels, weightCols)

	// 偏置向量：(out_channels, 1)
	biases := matrix.NewMatrix(outChannels, 1)

	op := &ConvOp{
		InChannels:  inChannels,
		OutChannels: outChannels,
		KernelSize:  kernelSize,
		Stride:      stride,
		Padding:     padding,
		weights:     weights,
		biases:      biases,
	}

	// 使用He初始化权重
	op.initializeWeights()

	return op
}

// initializeWeights 使用He初始化权重
func (op *ConvOp) initializeWeights() {
	// He初始化：权重 ~ N(0, sqrt(2 / fan_in))
	fanIn := float64(op.InChannels * op.KernelSize * op.KernelSize)
	stddev := math.Sqrt(2.0 / fanIn)

	// 初始化权重
	rand.Seed(time.Now().UnixNano())
	for i := 0; i < len(op.weights.Data); i++ {
		op.weights.Data[i] = rand.NormFloat64() * stddev
	}

	// 偏置初始化为0
	for i := 0; i < len(op.biases.Data); i++ {
		op.biases.Data[i] = 0.0
	}
}

// SetFixedWeights 设置固定权重（用于测试）
func (op *ConvOp) SetFixedWeights() {
	// 设置固定的权重值 - 简单的递增模式
	for i := 0; i < op.weights.Rows; i++ {
		for j := 0; j < op.weights.Cols; j++ {
			op.weights.Set(i, j, float64(i*10+j)*0.1)
		}
	}

	// 设置固定的偏置值
	for i := 0; i < op.biases.Rows; i++ {
		op.biases.Set(i, 0, float64(i)*0.5)
	}
}

// SetInputSize 设置输入尺寸并计算输出尺寸
func (op *ConvOp) SetInputSize(height, width int) error {
	op.InputHeight = height
	op.InputWidth = width

	// 计算输出尺寸
	op.OutputHeight = (height+2*op.Padding-op.KernelSize)/op.Stride + 1
	op.OutputWidth = (width+2*op.Padding-op.KernelSize)/op.Stride + 1

	if op.OutputHeight <= 0 || op.OutputWidth <= 0 {
		return fmt.Errorf("无效的输出尺寸，请检查卷积参数")
	}

	return nil
}

// Forward 前向传播
// 输入：(batch_size, in_channels * input_height * input_width)
// 输出：(batch_size, out_channels * output_height * output_width)
func (op *ConvOp) Forward(inputs ...*matrix.Matrix) *matrix.Matrix {
	if len(inputs) != 1 {
		panic("ConvOp requires exactly 1 input")
	}

	input := inputs[0]

	if op.InputHeight == 0 || op.InputWidth == 0 {
		panic("请先调用SetInputSize设置输入尺寸")
	}

	batchSize := input.Rows
	expectedInputSize := op.InChannels * op.InputHeight * op.InputWidth

	if input.Cols != expectedInputSize {
		panic(fmt.Sprintf("输入尺寸不匹配：期望 %d，实际 %d", expectedInputSize, input.Cols))
	}

	// 创建输出矩阵
	outputSize := op.OutChannels * op.OutputHeight * op.OutputWidth
	output := matrix.NewMatrix(batchSize, outputSize)

	// 处理每个样本
	for b := 0; b < batchSize; b++ {
		// 提取单个样本并重塑为3D形式进行im2col
		sampleInput := op.extractAndReshapeSample(input, b)

		// 使用im2col进行卷积计算
		sampleOutput := op.convolveSample(sampleInput)

		// 将结果复制到输出矩阵
		op.copySampleToOutput(sampleOutput, output, b)
	}

	// 缓存用于反向传播
	op.cachedInput = input.Copy()
	op.cachedOutput = output.Copy()

	return output
}

// extractAndReshapeSample 提取并重塑单个样本为Im2Col需要的格式
func (op *ConvOp) extractAndReshapeSample(input *matrix.Matrix, batchIndex int) *matrix.Matrix {
	// 创建Im2Col期望的格式：(channels*height) x width
	sample := matrix.NewMatrix(op.InChannels*op.InputHeight, op.InputWidth)

	for c := 0; c < op.InChannels; c++ {
		for h := 0; h < op.InputHeight; h++ {
			for w := 0; w < op.InputWidth; w++ {
				// 从批次输入中获取值
				inputIdx := c*op.InputHeight*op.InputWidth + h*op.InputWidth + w
				value := input.At(batchIndex, inputIdx)

				// 设置到sample中
				sample.Set(c*op.InputHeight+h, w, value)
			}
		}
	}

	return sample
}

// convolveSample 对单个样本进行卷积
func (op *ConvOp) convolveSample(input *matrix.Matrix) *matrix.Matrix {
	// 使用im2col转换输入
	inputCols := matrix.Im2ColWithChannels(input, op.InChannels, op.KernelSize, op.KernelSize,
		op.Stride, op.Stride, op.Padding, op.Padding)

	// 执行矩阵乘法：weights @ input_cols
	convResult := op.weights.Mul(inputCols)

	// 添加偏置
	for outChannel := 0; outChannel < op.OutChannels; outChannel++ {
		bias := op.biases.At(outChannel, 0)
		for col := 0; col < convResult.Cols; col++ {
			currentVal := convResult.At(outChannel, col)
			convResult.Set(outChannel, col, currentVal+bias)
		}
	}

	return convResult
}

// copySampleToOutput 将样本结果复制到批次输出
func (op *ConvOp) copySampleToOutput(sampleOutput, batchOutput *matrix.Matrix, batchIndex int) {
	outputSize := op.OutChannels * op.OutputHeight * op.OutputWidth
	for i := 0; i < outputSize; i++ {
		row := i / (op.OutputHeight * op.OutputWidth)
		col := i % (op.OutputHeight * op.OutputWidth)
		value := sampleOutput.At(row, col)
		batchOutput.Set(batchIndex, i, value)
	}
}

// Backward 反向传播
// 输入：gradOutput (batch_size, out_channels * output_height * output_width) - 来自上层的梯度
// 输出：gradInput (batch_size, in_channels * input_height * input_width) - 传递给下层的梯度
func (op *ConvOp) Backward(gradOutput *matrix.Matrix, inputs ...*Node) []*matrix.Matrix {
	if op.cachedInput == nil {
		panic("必须先执行前向传播才能进行反向传播")
	}

	batchSize := gradOutput.Rows
	expectedOutputSize := op.OutChannels * op.OutputHeight * op.OutputWidth

	if gradOutput.Cols != expectedOutputSize {
		panic(fmt.Sprintf("梯度输出尺寸不匹配：期望 %d，实际 %d", expectedOutputSize, gradOutput.Cols))
	}

	// 初始化梯度
	op.weightGradients = matrix.NewMatrix(op.weights.Rows, op.weights.Cols)
	op.biasGradients = matrix.NewMatrix(op.biases.Rows, op.biases.Cols)

	// 创建输入梯度
	inputSize := op.InChannels * op.InputHeight * op.InputWidth
	gradInput := matrix.NewMatrix(batchSize, inputSize)

	// 处理每个样本
	for b := 0; b < batchSize; b++ {
		// 提取样本梯度和输入
		sampleGradOutput := op.extractSampleGradient(gradOutput, b)
		sampleInput := op.extractAndReshapeSample(op.cachedInput, b)

		// 计算权重梯度
		op.computeWeightGradients(sampleInput, sampleGradOutput)

		// 计算偏置梯度
		op.computeBiasGradients(sampleGradOutput)

		// 计算输入梯度
		sampleGradInput := op.computeInputGradients(sampleGradOutput)

		// 将样本输入梯度复制到总梯度
		op.copySampleGradientToTotal(sampleGradInput, gradInput, b)
	}

	return []*matrix.Matrix{gradInput}
}

// extractSampleGradient 提取单个样本的梯度
func (op *ConvOp) extractSampleGradient(gradOutput *matrix.Matrix, batchIndex int) *matrix.Matrix {
	// 重塑为 (out_channels, output_height * output_width)
	sampleGrad := matrix.NewMatrix(op.OutChannels, op.OutputHeight*op.OutputWidth)

	for c := 0; c < op.OutChannels; c++ {
		for i := 0; i < op.OutputHeight*op.OutputWidth; i++ {
			outputIdx := c*op.OutputHeight*op.OutputWidth + i
			sampleGrad.Set(c, i, gradOutput.At(batchIndex, outputIdx))
		}
	}

	return sampleGrad
}

// computeWeightGradients 计算权重梯度
func (op *ConvOp) computeWeightGradients(input, gradOutput *matrix.Matrix) {
	// 使用im2col转换输入
	inputCols := matrix.Im2ColWithChannels(input, op.InChannels, op.KernelSize, op.KernelSize,
		op.Stride, op.Stride, op.Padding, op.Padding)

	// 计算权重梯度：grad_output @ input_cols^T
	gradWeights := gradOutput.Mul(inputCols.T())

	// 累加梯度
	for i := 0; i < op.weightGradients.Rows; i++ {
		for j := 0; j < op.weightGradients.Cols; j++ {
			currentGrad := op.weightGradients.At(i, j)
			newGrad := gradWeights.At(i, j)
			op.weightGradients.Set(i, j, currentGrad+newGrad)
		}
	}
}

// computeBiasGradients 计算偏置梯度
func (op *ConvOp) computeBiasGradients(gradOutput *matrix.Matrix) {
	// 偏置梯度是梯度输出在空间维度上的求和
	for c := 0; c < op.OutChannels; c++ {
		gradSum := 0.0
		for i := 0; i < op.OutputHeight*op.OutputWidth; i++ {
			gradSum += gradOutput.At(c, i)
		}

		currentBiasGrad := op.biasGradients.At(c, 0)
		op.biasGradients.Set(c, 0, currentBiasGrad+gradSum)
	}
}

// computeInputGradients 计算输入梯度
func (op *ConvOp) computeInputGradients(gradOutput *matrix.Matrix) *matrix.Matrix {
	// 计算梯度：weights^T @ grad_output
	gradCols := op.weights.T().Mul(gradOutput)

	// 使用col2im将梯度转换回输入格式
	gradInput := matrix.Col2ImWithChannels(gradCols, op.InChannels, op.InputHeight, op.InputWidth,
		op.KernelSize, op.KernelSize, op.Stride, op.Stride, op.Padding, op.Padding)

	return gradInput
}

// copySampleGradientToTotal 将样本梯度复制到总梯度
func (op *ConvOp) copySampleGradientToTotal(sampleGrad, totalGrad *matrix.Matrix, batchIndex int) {
	// sampleGrad格式：(channels*inputH) x inputW
	// totalGrad格式：(batchSize, channels*inputH*inputW)
	for c := 0; c < op.InChannels; c++ {
		for h := 0; h < op.InputHeight; h++ {
			for w := 0; w < op.InputWidth; w++ {
				// 从sampleGrad获取值
				value := sampleGrad.At(c*op.InputHeight+h, w)

				// 计算在totalGrad中的索引
				inputIdx := c*op.InputHeight*op.InputWidth + h*op.InputWidth + w
				totalGrad.Set(batchIndex, inputIdx, value)
			}
		}
	}
}

// Name 操作名称
func (op *ConvOp) Name() string {
	return fmt.Sprintf("Conv2d(%dx%d/%d/%d, %d->%d)", op.KernelSize, op.KernelSize, op.Stride, op.Padding, op.InChannels, op.OutChannels)
}

// GetWeights 获取权重引用
func (op *ConvOp) GetWeights() *matrix.Matrix {
	return op.weights
}

// GetBiases 获取偏置引用
func (op *ConvOp) GetBiases() *matrix.Matrix {
	return op.biases
}

// GetWeightGradients 获取权重梯度副本
func (op *ConvOp) GetWeightGradients() *matrix.Matrix {
	if op.weightGradients == nil {
		return nil
	}
	return op.weightGradients.Copy()
}

// GetBiasGradients 获取偏置梯度副本
func (op *ConvOp) GetBiasGradients() *matrix.Matrix {
	if op.biasGradients == nil {
		return nil
	}
	return op.biasGradients.Copy()
}

// ZeroGradients 清零梯度
func (op *ConvOp) ZeroGradients() {
	if op.weightGradients != nil {
		for i := 0; i < len(op.weightGradients.Data); i++ {
			op.weightGradients.Data[i] = 0.0
		}
	}

	if op.biasGradients != nil {
		for i := 0; i < len(op.biasGradients.Data); i++ {
			op.biasGradients.Data[i] = 0.0
		}
	}
}
