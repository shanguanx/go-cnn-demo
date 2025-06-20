package graph

import (
	"fmt"

	"github.com/user/go-cnn/layers"
	"github.com/user/go-cnn/matrix"
)

// DenseLayerOp 全连接层操作 - CNN的分类器部分
type DenseLayerOp struct {
	*layers.DenseLayer
	CachedInput *matrix.Matrix
}

func NewDenseLayerOp(inputSize, outputSize int) *DenseLayerOp {
	layer := layers.NewDenseLayer(inputSize, outputSize)
	return &DenseLayerOp{
		DenseLayer: layer,
	}
}

func (op *DenseLayerOp) Forward(inputs ...*matrix.Matrix) *matrix.Matrix {
	if len(inputs) != 1 {
		panic("DenseLayerOp requires exactly 1 input")
	}

	op.CachedInput = inputs[0].Copy()
	output, err := op.DenseLayer.Forward(inputs[0])
	if err != nil {
		panic(fmt.Sprintf("DenseLayer forward error: %v", err))
	}
	return output
}

func (op *DenseLayerOp) Backward(gradOutput *matrix.Matrix, inputs ...*Node) []*matrix.Matrix {
	gradInput, err := op.DenseLayer.Backward(gradOutput)
	if err != nil {
		panic(fmt.Sprintf("DenseLayer backward error: %v", err))
	}
	return []*matrix.Matrix{gradInput}
}

func (op *DenseLayerOp) Name() string {
	return fmt.Sprintf("Dense(%d->%d)", op.DenseLayer.GetWeights().Cols, op.DenseLayer.GetWeights().Rows)
}

// SetFixedWeights 设置固定权重（用于测试）
func (op *DenseLayerOp) SetFixedWeights() {
	op.DenseLayer.SetFixedWeights()
}

// ConvolutionalLayerOp 卷积层操作 - CNN的特征提取部分
type ConvolutionalLayerOp struct {
	*layers.ConvolutionalLayer
	CachedInput *matrix.Matrix
}

func NewConvolutionalLayerOp(inChannels, outChannels, kernelSize, stride, padding int) *ConvolutionalLayerOp {
	layer := layers.NewConvolutionalLayer(inChannels, outChannels, kernelSize, stride, padding)
	return &ConvolutionalLayerOp{
		ConvolutionalLayer: layer,
	}
}

func (op *ConvolutionalLayerOp) Forward(inputs ...*matrix.Matrix) *matrix.Matrix {
	if len(inputs) != 1 {
		panic("ConvolutionalLayerOp requires exactly 1 input")
	}

	op.CachedInput = inputs[0].Copy()
	output, err := op.ConvolutionalLayer.Forward(inputs[0])
	if err != nil {
		panic(fmt.Sprintf("ConvolutionalLayer forward error: %v", err))
	}
	return output
}

func (op *ConvolutionalLayerOp) Backward(gradOutput *matrix.Matrix, inputs ...*Node) []*matrix.Matrix {
	gradInput, err := op.ConvolutionalLayer.Backward(gradOutput)
	if err != nil {
		panic(fmt.Sprintf("ConvolutionalLayer backward error: %v", err))
	}
	return []*matrix.Matrix{gradInput}
}

func (op *ConvolutionalLayerOp) Name() string {
	return fmt.Sprintf("Conv2d(%d->%d, k=%d)",
		op.ConvolutionalLayer.GetInChannels(),
		op.ConvolutionalLayer.GetOutChannels(),
		op.ConvolutionalLayer.GetKernelSize())
}

// SetFixedWeights 设置固定权重（用于测试）
func (op *ConvolutionalLayerOp) SetFixedWeights() {
	op.ConvolutionalLayer.SetFixedWeights()
}

// PoolingLayerOp 池化层操作 - CNN的下采样部分
type PoolingLayerOp struct {
	*layers.PoolingLayer
	CachedInput *matrix.Matrix
}

func NewMaxPoolingLayerOp(poolSize, stride int) *PoolingLayerOp {
	layer := layers.NewMaxPoolingLayer(poolSize, poolSize, stride, stride)
	return &PoolingLayerOp{
		PoolingLayer: layer,
	}
}

func NewAveragePoolingLayerOp(poolSize, stride int) *PoolingLayerOp {
	layer := layers.NewAveragePoolingLayer(poolSize, poolSize, stride, stride)
	return &PoolingLayerOp{
		PoolingLayer: layer,
	}
}

func (op *PoolingLayerOp) Forward(inputs ...*matrix.Matrix) *matrix.Matrix {
	if len(inputs) != 1 {
		panic("PoolingLayerOp requires exactly 1 input")
	}

	op.CachedInput = inputs[0].Copy()
	output, err := op.PoolingLayer.Forward(inputs[0])
	if err != nil {
		panic(fmt.Sprintf("PoolingLayer forward error: %v", err))
	}
	return output
}

func (op *PoolingLayerOp) Backward(gradOutput *matrix.Matrix, inputs ...*Node) []*matrix.Matrix {
	gradInput, err := op.PoolingLayer.Backward(gradOutput)
	if err != nil {
		panic(fmt.Sprintf("PoolingLayer backward error: %v", err))
	}
	return []*matrix.Matrix{gradInput}
}

func (op *PoolingLayerOp) Name() string {
	poolType := "MaxPool"
	if op.PoolingLayer.GetPoolingType() == layers.AveragePooling {
		poolType = "AvgPool"
	}
	poolHeight, poolWidth, _, _ := op.PoolingLayer.GetPoolingParams()
	return fmt.Sprintf("%s(%dx%d)", poolType, poolHeight, poolWidth)
}

// 获取层参数的函数 - 用于优化器更新权重
func GetLayerParameters(op Operation) []*Node {
	switch layerOp := op.(type) {
	case *DenseLayerOp:
		weights := layerOp.DenseLayer.GetWeights()
		biases := layerOp.DenseLayer.GetBiases()

		weightNode := NewParameter(weights, "dense_weights")
		biasNode := NewParameter(biases, "dense_biases")

		return []*Node{weightNode, biasNode}

	case *ConvolutionalLayerOp:
		weights := layerOp.ConvolutionalLayer.GetWeights()
		biases := layerOp.ConvolutionalLayer.GetBiases()

		weightNode := NewParameter(weights, "conv_weights")
		biasNode := NewParameter(biases, "conv_biases")

		return []*Node{weightNode, biasNode}

	default:
		return nil
	}
}
