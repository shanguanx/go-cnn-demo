package graph

import (
	"github.com/user/go-cnn/matrix"
	"github.com/user/go-cnn/optimizers"
)

// LayerOperation 定义可训练层的接口
type LayerOperation interface {
	GetWeights() *matrix.Matrix
	GetBiases() *matrix.Matrix
	GetWeightGradients() *matrix.Matrix
	GetBiasGradients() *matrix.Matrix
	ZeroGradients()
}

// Model 结构体 - 管理模型和优化器
type Model struct {
	layerOps  []LayerOperation
	optimizer *optimizers.SGD
	output    *Node
}

// NewModel 创建新的模型实例
func NewModel(optimizer *optimizers.SGD) *Model {
	return &Model{
		layerOps:  make([]LayerOperation, 0),
		optimizer: optimizer,
	}
}

// AddLayerOp 添加可训练层操作到模型
func (m *Model) AddLayerOp(layerOp LayerOperation) {
	m.layerOps = append(m.layerOps, layerOp)
}

// CollectParameters 自动收集计算图中的所有可训练层
func (m *Model) CollectParameters(root *Node) {
	visited := make(map[*Node]bool)
	m.collectLayersRecursive(root, visited)
}

// collectLayersRecursive 递归收集可训练层
func (m *Model) collectLayersRecursive(node *Node, visited map[*Node]bool) {
	if visited[node] {
		return
	}
	visited[node] = true

	// 检查操作是否为可训练层
	if node.Op != nil {
		switch op := node.Op.(type) {
		case *ConvOp:
			// 收集卷积层操作
			m.layerOps = append(m.layerOps, op)
		case *DenseOp:
			// 收集全连接层操作
			m.layerOps = append(m.layerOps, op)
		}
	}

	// 递归处理输入节点
	for _, input := range node.Inputs {
		m.collectLayersRecursive(input, visited)
	}
}

// ZeroGrad 清零所有参数的梯度
func (m *Model) ZeroGrad() {
	for _, layerOp := range m.layerOps {
		layerOp.ZeroGradients()
	}
}

// Step 执行一步参数更新
func (m *Model) Step() {
	for _, layerOp := range m.layerOps {
		// 更新权重
		weights := layerOp.GetWeights()
		weightGrads := layerOp.GetWeightGradients()
		if weights != nil && weightGrads != nil {
			m.optimizer.Update(weights, weightGrads)
		}

		// 更新偏置
		biases := layerOp.GetBiases()
		biasGrads := layerOp.GetBiasGradients()
		if biases != nil && biasGrads != nil {
			m.optimizer.Update(biases, biasGrads)
		}
	}
}

// SetOutput 设置模型的输出节点
func (m *Model) SetOutput(output *Node) {
	m.output = output
}

// Forward 前向传播
func (m *Model) Forward(input *matrix.Matrix) *matrix.Matrix {
	if m.output == nil {
		panic("Model output not set. Call SetOutput() first.")
	}
	// 输入数据已经在构建图时设置，这里返回输出
	return m.output.Value
}

// GetParameterCount 获取模型参数数量
func (m *Model) GetParameterCount() int {
	count := 0
	for _, layerOp := range m.layerOps {
		if layerOp.GetWeights() != nil {
			count++
		}
		if layerOp.GetBiases() != nil {
			count++
		}
	}
	return count
}
