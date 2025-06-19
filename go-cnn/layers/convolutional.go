package layers

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/user/go-cnn/matrix"
)

// ConvolutionalLayer 卷积层结构体 - 基于2D矩阵实现
type ConvolutionalLayer struct {
	// 网络参数
	Weights *matrix.Matrix // 权重矩阵：(out_channels, in_channels * kernel_size * kernel_size)
	Biases  *matrix.Matrix // 偏置向量：(out_channels, 1)

	// 层配置
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

	// 用于反向传播的缓存
	LastInput     *matrix.Matrix // 缓存的输入 (batch_size, in_channels * input_height * input_width)
	LastInputCols *matrix.Matrix // 缓存的im2col结果
	LastOutput    *matrix.Matrix // 缓存的输出 (batch_size, out_channels * output_height * output_width)

	// 梯度缓存
	WeightGradients *matrix.Matrix // 权重梯度
	BiasGradients   *matrix.Matrix // 偏置梯度
}

// NewConvolutionalLayer 创建新的卷积层
func NewConvolutionalLayer(inChannels, outChannels, kernelSize, stride, padding int) *ConvolutionalLayer {
	// 权重矩阵：(out_channels, in_channels * kernel_size * kernel_size)
	weightCols := inChannels * kernelSize * kernelSize
	weights := matrix.NewMatrix(outChannels, weightCols)

	// 偏置向量：(out_channels, 1)
	biases := matrix.NewMatrix(outChannels, 1)

	layer := &ConvolutionalLayer{
		Weights:     weights,
		Biases:      biases,
		InChannels:  inChannels,
		OutChannels: outChannels,
		KernelSize:  kernelSize,
		Stride:      stride,
		Padding:     padding,
	}

	// 初始化参数
	layer.initializeWeights()

	return layer
}

// initializeWeights 使用He初始化权重
func (layer *ConvolutionalLayer) initializeWeights() {
	// He初始化：权重 ~ N(0, sqrt(2 / fan_in))
	fanIn := float64(layer.InChannels * layer.KernelSize * layer.KernelSize)
	stddev := math.Sqrt(2.0 / fanIn)

	// 初始化权重
	rand.Seed(time.Now().UnixNano())
	for i := 0; i < len(layer.Weights.Data); i++ {
		layer.Weights.Data[i] = rand.NormFloat64() * stddev
	}

	// 偏置初始化为0
	for i := 0; i < len(layer.Biases.Data); i++ {
		layer.Biases.Data[i] = 0.0
	}
}

// SetInputSize 设置输入尺寸并计算输出尺寸
func (layer *ConvolutionalLayer) SetInputSize(height, width int) error {
	layer.InputHeight = height
	layer.InputWidth = width

	// 计算输出尺寸
	layer.OutputHeight = (height+2*layer.Padding-layer.KernelSize)/layer.Stride + 1
	layer.OutputWidth = (width+2*layer.Padding-layer.KernelSize)/layer.Stride + 1

	if layer.OutputHeight <= 0 || layer.OutputWidth <= 0 {
		return errors.New("无效的输出尺寸，请检查卷积参数")
	}

	return nil
}

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

// GetWeights 获取权重副本
func (layer *ConvolutionalLayer) GetWeights() *matrix.Matrix {
	return layer.Weights.Copy()
}

// GetBiases 获取偏置副本
func (layer *ConvolutionalLayer) GetBiases() *matrix.Matrix {
	return layer.Biases.Copy()
}

// GetWeightGradients 获取权重梯度副本
func (layer *ConvolutionalLayer) GetWeightGradients() *matrix.Matrix {
	if layer.WeightGradients == nil {
		return nil
	}
	return layer.WeightGradients.Copy()
}

// GetBiasGradients 获取偏置梯度副本
func (layer *ConvolutionalLayer) GetBiasGradients() *matrix.Matrix {
	if layer.BiasGradients == nil {
		return nil
	}
	return layer.BiasGradients.Copy()
}

// GetOutputShape 计算输出形状
func (layer *ConvolutionalLayer) GetOutputShape(batchSize int) (int, int, error) {
	if layer.OutputHeight == 0 || layer.OutputWidth == 0 {
		return 0, 0, errors.New("请先调用SetInputSize设置输入尺寸")
	}

	outputSize := layer.OutChannels * layer.OutputHeight * layer.OutputWidth
	return batchSize, outputSize, nil
}

// String 返回层的字符串表示
func (layer *ConvolutionalLayer) String() string {
	return fmt.Sprintf("ConvolutionalLayer(in_channels=%d, out_channels=%d, kernel_size=%d, stride=%d, padding=%d, input_size=%dx%d)",
		layer.InChannels, layer.OutChannels, layer.KernelSize, layer.Stride, layer.Padding,
		layer.InputHeight, layer.InputWidth)
}
