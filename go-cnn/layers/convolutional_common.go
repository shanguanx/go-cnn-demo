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

// SetFixedWeights 设置固定的权重和偏置（用于测试）
func (layer *ConvolutionalLayer) SetFixedWeights() {
	// 设置固定的权重值
	// 对于 2输入通道，3输出通道，3x3卷积核：权重矩阵形状为 (3, 18)
	// 使用简单的递增模式
	for i := 0; i < layer.Weights.Rows; i++ {
		for j := 0; j < layer.Weights.Cols; j++ {
			layer.Weights.Set(i, j, float64(i*10+j)*0.1)
		}
	}

	// 设置固定的偏置值
	for i := 0; i < layer.Biases.Rows; i++ {
		layer.Biases.Set(i, 0, float64(i)*0.5)
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

// GetInChannels 获取输入通道数
func (layer *ConvolutionalLayer) GetInChannels() int {
	return layer.InChannels
}

// GetOutChannels 获取输出通道数
func (layer *ConvolutionalLayer) GetOutChannels() int {
	return layer.OutChannels
}

// GetKernelSize 获取卷积核大小
func (layer *ConvolutionalLayer) GetKernelSize() int {
	return layer.KernelSize
}

// GetStride 获取步长
func (layer *ConvolutionalLayer) GetStride() int {
	return layer.Stride
}

// GetPadding 获取填充
func (layer *ConvolutionalLayer) GetPadding() int {
	return layer.Padding
}

// GetName 获取层名称
func (layer *ConvolutionalLayer) GetName() string {
	return "ConvolutionalLayer"
}
