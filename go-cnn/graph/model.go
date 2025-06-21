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
	inputNode *Node   // 输入节点引用
	allNodes  []*Node // 所有节点引用，用于状态管理
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

// CollectParameters 自动收集计算图中的所有可训练层和节点
func (m *Model) CollectParameters(root *Node) {
	visited := make(map[*Node]bool)
	m.collectLayersRecursive(root, visited)

	// 收集所有节点用于状态管理
	visited = make(map[*Node]bool)
	m.collectAllNodesRecursive(root, visited)
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

// collectAllNodesRecursive 递归收集所有节点
func (m *Model) collectAllNodesRecursive(node *Node, visited map[*Node]bool) {
	if visited[node] {
		return
	}
	visited[node] = true

	// 添加当前节点
	m.allNodes = append(m.allNodes, node)

	// 递归处理输入节点
	for _, input := range node.Inputs {
		m.collectAllNodesRecursive(input, visited)
	}
}

// ZeroGrad 清零所有参数的梯度和节点状态
func (m *Model) ZeroGrad() {
	// 1. 清零参数梯度
	for _, layerOp := range m.layerOps {
		layerOp.ZeroGradients()
	}

	// 2. 清零所有节点的梯度状态
	for _, node := range m.allNodes {
		node.Gradient = nil
		node.GradComputed = false
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

// SetInput 设置模型的输入节点
func (m *Model) SetInput(input *Node) {
	m.inputNode = input
}

// Forward 自动前向传播
func (m *Model) Forward(input *matrix.Matrix) *matrix.Matrix {
	if m.output == nil {
		panic("Model output not set. Call SetOutput() first.")
	}
	if m.inputNode == nil {
		panic("Model input not set. Call SetInput() first.")
	}

	// 1. 更新输入节点的数据
	m.inputNode.Value = input

	// 2. 重新计算整个前向传播
	m.recalculateForward(m.output)

	// 3. 返回输出结果
	return m.output.Value
}

// recalculateForward 重新计算前向传播
func (m *Model) recalculateForward(node *Node) {
	// 重新计算前向传播（不需要清零梯度，因为ZeroGrad已经处理）
	visited := make(map[*Node]bool)
	m.recalculateNodeRecursive(node, visited)
}

// recalculateNodeRecursive 递归重新计算节点值
func (m *Model) recalculateNodeRecursive(node *Node, visited map[*Node]bool) {
	if visited[node] {
		return
	}
	visited[node] = true

	// 先计算输入节点
	for _, input := range node.Inputs {
		m.recalculateNodeRecursive(input, visited)
	}

	// 然后计算当前节点
	if node.Op != nil && len(node.Inputs) > 0 {
		inputValues := make([]*matrix.Matrix, len(node.Inputs))
		for i, input := range node.Inputs {
			inputValues[i] = input.Value
		}
		node.Value = node.Op.Forward(inputValues...)
	}
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

// GetLayerOps 获取所有可训练层操作（用于模型保存）
func (m *Model) GetLayerOps() []LayerOperation {
	return m.layerOps
}
