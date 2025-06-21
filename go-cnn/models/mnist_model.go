package models

import (
	"github.com/user/go-cnn/graph"
	"github.com/user/go-cnn/matrix"
	"github.com/user/go-cnn/optimizers"
)

// MNISTModelConfig MNIST模型配置
type MNISTModelConfig struct {
	BatchSize    int
	LearningRate float64
	InputSize    int // 固定为784 (28x28)
	OutputSize   int // 固定为10 (0-9数字)
}

// DefaultMNISTConfig 默认MNIST模型配置
func DefaultMNISTConfig() MNISTModelConfig {
	return MNISTModelConfig{
		BatchSize:    32,
		LearningRate: 0.001,
		InputSize:    784,
		OutputSize:   10,
	}
}

// MNISTModelBuilder MNIST模型构建器
type MNISTModelBuilder struct {
	config MNISTModelConfig
}

// NewMNISTModelBuilder 创建MNIST模型构建器
func NewMNISTModelBuilder(config MNISTModelConfig) *MNISTModelBuilder {
	return &MNISTModelBuilder{
		config: config,
	}
}

// BuildForTraining 构建用于训练的模型
func (b *MNISTModelBuilder) BuildForTraining() (*graph.Model, *graph.Node, *graph.Node) {
	// 创建优化器
	optimizer := optimizers.NewSGD(b.config.LearningRate)

	// 创建模型
	model := graph.NewModel(optimizer)

	// 构建网络
	inputData := matrix.NewMatrix(b.config.BatchSize, b.config.InputSize)
	input := graph.Input(inputData, "input")
	output := b.buildLeNetArchitecture(input)

	// 设置模型输入和输出并收集参数
	model.SetInput(input)
	model.SetOutput(output)
	model.CollectParameters(output)

	return model, input, output
}

// BuildForInference 构建用于推理的模型
func (b *MNISTModelBuilder) BuildForInference() (*graph.Model, *graph.Node, *graph.Node) {
	// 推理时学习率无关紧要
	optimizer := optimizers.NewSGD(0.001)

	// 创建模型
	model := graph.NewModel(optimizer)

	// 构建网络
	inputData := matrix.NewMatrix(b.config.BatchSize, b.config.InputSize)
	input := graph.Input(inputData, "input")
	output := b.buildLeNetArchitecture(input)

	// 设置模型输入和输出并收集参数
	model.SetInput(input)
	model.SetOutput(output)
	model.CollectParameters(output)

	return model, input, output
}

// buildLeNetArchitecture 构建LeNet-5架构
// 输入: 28x28x1 -> Conv1(6@5x5) -> MaxPool -> Conv2(16@5x5) -> MaxPool -> FC(120) -> FC(84) -> FC(10)
func (b *MNISTModelBuilder) buildLeNetArchitecture(input *graph.Node) *graph.Node {
	// 第一层卷积: 28x28x1 -> 24x24x6 (Conv 5x5, stride=1, padding=0)
	conv1 := graph.Conv2d(input, 6, 5, 1, 0, 28, 28, 1)
	relu1 := graph.ReLU(conv1)

	// 第一层池化: 24x24x6 -> 12x12x6 (MaxPool 2x2, stride=2)
	pool1 := graph.MaxPool2d(relu1, 2, 2, 24, 24, 6)

	// 第二层卷积: 12x12x6 -> 8x8x16 (Conv 5x5, stride=1, padding=0)
	conv2 := graph.Conv2d(pool1, 16, 5, 1, 0, 12, 12, 6)
	relu2 := graph.ReLU(conv2)

	// 第二层池化: 8x8x16 -> 4x4x16 (MaxPool 2x2, stride=2)
	pool2 := graph.MaxPool2d(relu2, 2, 2, 8, 8, 16)

	// 展平: 4x4x16 = 256
	flatten := graph.Reshape(pool2, pool2.Value.Rows, 256)

	// 全连接层1: 256 -> 120
	fc1 := graph.Dense(flatten, 120)
	relu3 := graph.ReLU(fc1)

	// 全连接层2: 120 -> 84
	fc2 := graph.Dense(relu3, 84)
	relu4 := graph.ReLU(fc2)

	// 输出层: 84 -> 10 (不加Softmax，在损失函数中处理)
	output := graph.Dense(relu4, 10)

	return output
}

// GetConfig 获取模型配置
func (b *MNISTModelBuilder) GetConfig() MNISTModelConfig {
	return b.config
}

// UpdateConfig 更新模型配置
func (b *MNISTModelBuilder) UpdateConfig(config MNISTModelConfig) {
	b.config = config
}
