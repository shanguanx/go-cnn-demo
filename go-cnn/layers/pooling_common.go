package layers

import (
	"fmt"
	"github.com/user/go-cnn/matrix"
)

// PoolingType 池化类型枚举
type PoolingType int

const (
	MaxPooling PoolingType = iota
	AveragePooling
)

// PoolingLayer 池化层结构体
type PoolingLayer struct {
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

	// 前向传播缓存（用于反向传播）
	lastInput  *matrix.Matrix // 输入数据缓存
	maxIndices *matrix.Matrix // MaxPooling时记录最大值位置（仅用于MaxPooling）

	// 是否已设置输入尺寸
	inputSizeSet bool
}

// NewMaxPoolingLayer 创建MaxPooling层
func NewMaxPoolingLayer(poolHeight, poolWidth, strideH, strideW int) *PoolingLayer {
	if poolHeight <= 0 || poolWidth <= 0 || strideH <= 0 || strideW <= 0 {
		panic("池化层参数必须大于0")
	}

	return &PoolingLayer{
		poolType:     MaxPooling,
		poolHeight:   poolHeight,
		poolWidth:    poolWidth,
		strideH:      strideH,
		strideW:      strideW,
		inputSizeSet: false,
	}
}

// NewAveragePoolingLayer 创建AveragePooling层
func NewAveragePoolingLayer(poolHeight, poolWidth, strideH, strideW int) *PoolingLayer {
	if poolHeight <= 0 || poolWidth <= 0 || strideH <= 0 || strideW <= 0 {
		panic("池化层参数必须大于0")
	}

	return &PoolingLayer{
		poolType:     AveragePooling,
		poolHeight:   poolHeight,
		poolWidth:    poolWidth,
		strideH:      strideH,
		strideW:      strideW,
		inputSizeSet: false,
	}
}

// SetInputSize 设置输入尺寸并计算输出尺寸
func (p *PoolingLayer) SetInputSize(height, width, channels int) error {
	if height <= 0 || width <= 0 || channels <= 0 {
		return fmt.Errorf("输入尺寸必须大于0: height=%d, width=%d, channels=%d", height, width, channels)
	}

	// 检查池化窗口是否超出输入尺寸
	if p.poolHeight > height || p.poolWidth > width {
		return fmt.Errorf("池化窗口尺寸不能超过输入尺寸: pool=(%d,%d), input=(%d,%d)",
			p.poolHeight, p.poolWidth, height, width)
	}

	p.inputHeight = height
	p.inputWidth = width
	p.inputChannels = channels

	// 计算输出尺寸
	p.outputHeight = (height-p.poolHeight)/p.strideH + 1
	p.outputWidth = (width-p.poolWidth)/p.strideW + 1

	// 验证输出尺寸是否有效
	if p.outputHeight <= 0 || p.outputWidth <= 0 {
		return fmt.Errorf("计算得到的输出尺寸无效: output=(%d,%d), 请检查步长设置",
			p.outputHeight, p.outputWidth)
	}

	p.inputSizeSet = true
	return nil
}

// GetOutputSize 获取输出尺寸
func (p *PoolingLayer) GetOutputSize() (height, width, channels int, err error) {
	if !p.inputSizeSet {
		return 0, 0, 0, fmt.Errorf("必须先调用SetInputSize设置输入尺寸")
	}
	return p.outputHeight, p.outputWidth, p.inputChannels, nil
}

// GetInputSize 获取输入尺寸
func (p *PoolingLayer) GetInputSize() (height, width, channels int, err error) {
	if !p.inputSizeSet {
		return 0, 0, 0, fmt.Errorf("必须先调用SetInputSize设置输入尺寸")
	}
	return p.inputHeight, p.inputWidth, p.inputChannels, nil
}

// GetPoolingParams 获取池化参数
func (p *PoolingLayer) GetPoolingParams() (poolHeight, poolWidth, strideH, strideW int) {
	return p.poolHeight, p.poolWidth, p.strideH, p.strideW
}

// GetPoolingType 获取池化类型
func (p *PoolingLayer) GetPoolingType() PoolingType {
	return p.poolType
}

// String 返回层的字符串表示
func (p *PoolingLayer) String() string {
	if !p.inputSizeSet {
		return fmt.Sprintf("PoolingLayer(type=%v, pool=(%d,%d), stride=(%d,%d), input=未设置)",
			p.poolType, p.poolHeight, p.poolWidth, p.strideH, p.strideW)
	}

	return fmt.Sprintf("PoolingLayer(type=%v, pool=(%d,%d), stride=(%d,%d), input=(%d,%d,%d), output=(%d,%d,%d))",
		p.poolType, p.poolHeight, p.poolWidth, p.strideH, p.strideW,
		p.inputHeight, p.inputWidth, p.inputChannels,
		p.outputHeight, p.outputWidth, p.inputChannels)
}
