package graph

import (
	"fmt"
	"github.com/user/go-cnn/matrix"
)

// Node 表示计算图中的一个节点
type Node struct {
	// 节点唯一标识
	ID string

	// 节点名称（用于调试）
	Name string

	// 节点执行的操作
	Op Operation

	// 输入节点
	Inputs []*Node

	// 输出节点（用于反向传播）
	Outputs []*Node

	// 前向传播的值
	Value *matrix.Matrix

	// 反向传播的梯度
	Gradient *matrix.Matrix

	// 是否需要计算梯度
	RequiresGrad bool

	// 是否是叶子节点（例如参数）
	IsLeaf bool

	// 梯度是否已经计算
	GradComputed bool
}

// NewNode 创建一个新的计算图节点
func NewNode(value *matrix.Matrix, requiresGrad bool, op Operation, inputs ...*Node) *Node {
	node := &Node{
		ID:           generateNodeID(),
		Op:           op,
		Inputs:       inputs,
		Value:        value,
		RequiresGrad: requiresGrad,
		IsLeaf:       len(inputs) == 0,
		GradComputed: false,
	}

	// 设置输出关系
	for _, input := range inputs {
		input.Outputs = append(input.Outputs, node)
	}

	return node
}

// NewVariable 创建一个变量节点（叶子节点）
func NewVariable(value *matrix.Matrix, requiresGrad bool, name string) *Node {
	node := &Node{
		ID:           generateNodeID(),
		Name:         name,
		Op:           nil,
		Inputs:       nil,
		Value:        value,
		RequiresGrad: requiresGrad,
		IsLeaf:       true,
		GradComputed: false,
	}
	return node
}

// NewParameter 创建一个参数节点（需要梯度的变量）
func NewParameter(value *matrix.Matrix, name string) *Node {
	return NewVariable(value, true, name)
}

// NewConstant 创建一个常量节点（不需要梯度的变量）
func NewConstant(value *matrix.Matrix, name string) *Node {
	return NewVariable(value, false, name)
}

// Backward 对当前节点执行反向传播
func (n *Node) Backward() {
	if !n.RequiresGrad {
		return
	}

	// 如果梯度为nil，初始化梯度
	if n.Gradient == nil {
		if n.Value.Rows == 1 && n.Value.Cols == 1 {
			// 对于标量，梯度初始化为1
			n.Gradient = matrix.Ones(1, 1)
		} else {
			// 对于非标量，梯度初始化为与输出相同形状的全1矩阵
			// 这通常用于损失函数或调试目的
			n.Gradient = matrix.Ones(n.Value.Rows, n.Value.Cols)
		}
	}

	// 执行拓扑排序的反向传播
	sorted := topologicalSort(n)

	// 从后向前遍历（反向传播方向）
	for i := len(sorted) - 1; i >= 0; i-- {
		node := sorted[i]
		if !node.RequiresGrad || node.GradComputed {
			continue
		}

		// 如果不是叶子节点，计算对输入的梯度
		if !node.IsLeaf && node.Op != nil && node.Gradient != nil {
			gradInputs := node.Op.Backward(node.Gradient, node.Inputs...)

			// 将梯度传递给输入节点
			if len(gradInputs) != len(node.Inputs) {
				panic(fmt.Sprintf("节点 %s 的梯度数量不匹配：期望 %d 个，得到 %d 个",
					node.Name, len(node.Inputs), len(gradInputs)))
			}

			for j, input := range node.Inputs {
				if input.RequiresGrad {
					if input.Gradient == nil {
						input.Gradient = gradInputs[j].Copy()
					} else {
						// 梯度累积
						input.Gradient.AddInPlace(gradInputs[j])
					}
				}
			}
		}

		node.GradComputed = true
	}
}

// ZeroGrad 清零梯度
func (n *Node) ZeroGrad() {
	visited := make(map[*Node]bool)
	n.zeroGradRecursive(visited)
}

func (n *Node) zeroGradRecursive(visited map[*Node]bool) {
	if visited[n] {
		return
	}
	visited[n] = true

	n.Gradient = nil
	n.GradComputed = false

	for _, input := range n.Inputs {
		input.zeroGradRecursive(visited)
	}
}

// Detach 从计算图中分离，返回一个新的不需要梯度的节点
func (n *Node) Detach() *Node {
	return NewConstant(n.Value.Copy(), n.Name+"_detached")
}

// 拓扑排序用于反向传播
func topologicalSort(node *Node) []*Node {
	visited := make(map[*Node]bool)
	sorted := make([]*Node, 0)

	var visit func(*Node)
	visit = func(n *Node) {
		if visited[n] {
			return
		}
		visited[n] = true

		for _, input := range n.Inputs {
			visit(input)
		}

		sorted = append(sorted, n)
	}

	visit(node)
	return sorted
}

// 生成唯一的节点ID
var nodeCounter int

func generateNodeID() string {
	nodeCounter++
	return fmt.Sprintf("node_%d", nodeCounter)
}

// GetShape 获取节点值的形状
func (n *Node) GetShape() (int, int) {
	if n.Value != nil {
		return n.Value.Rows, n.Value.Cols
	}
	return 0, 0
}

// String 返回节点的字符串表示
func (n *Node) String() string {
	rows, cols := n.GetShape()
	opName := "Variable"
	if n.Op != nil {
		opName = n.Op.Name()
	}

	name := n.Name
	if name == "" {
		name = n.ID
	}

	return fmt.Sprintf("Node(%s, op=%s, shape=[%d,%d], requires_grad=%v)",
		name, opName, rows, cols, n.RequiresGrad)
}
