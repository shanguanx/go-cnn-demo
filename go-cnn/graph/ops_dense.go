package graph

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/user/go-cnn/matrix"
)

// DenseOp 全连接层操作 - 统一实现，消除与layers的重复
type DenseOp struct {
	// 层参数
	InputFeatures  int
	OutputFeatures int

	// 权重和偏置 (作为值直接存储，不是节点)
	weights *matrix.Matrix // (input_features, output_features)
	biases  *matrix.Matrix // (1, output_features)

	// 反向传播缓存
	cachedInput     *matrix.Matrix // 缓存的输入 (batch_size, input_features)
	weightGradients *matrix.Matrix // 权重梯度 (input_features, output_features)
	biasGradients   *matrix.Matrix // 偏置梯度 (1, output_features)
}

// NewDenseOp 创建新的全连接层操作
func NewDenseOp(inputFeatures, outputFeatures int) *DenseOp {
	op := &DenseOp{
		InputFeatures:  inputFeatures,
		OutputFeatures: outputFeatures,
		weights:        matrix.NewMatrix(inputFeatures, outputFeatures),
		biases:         matrix.NewMatrix(1, outputFeatures),
	}

	// 使用He初始化权重
	op.initializeWeights()

	return op
}

// initializeWeights 使用He初始化权重
func (op *DenseOp) initializeWeights() {
	// He初始化：权重 ~ N(0, sqrt(2 / fan_in))
	fanIn := float64(op.InputFeatures)
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
func (op *DenseOp) SetFixedWeights() {
	// 设置固定的权重值 - 简单的递增模式
	for i := 0; i < op.weights.Rows; i++ {
		for j := 0; j < op.weights.Cols; j++ {
			op.weights.Set(i, j, float64(i*op.weights.Cols+j)*0.1)
		}
	}

	// 设置固定的偏置值
	for j := 0; j < op.biases.Cols; j++ {
		op.biases.Set(0, j, float64(j)*0.5)
	}
}

// Forward 前向传播
// 输入：input (batch_size, input_features)
// 输出：output (batch_size, output_features)
// 计算：output = input * weights + biases
func (op *DenseOp) Forward(inputs ...*matrix.Matrix) *matrix.Matrix {
	if len(inputs) != 1 {
		panic("DenseOp requires exactly 1 input")
	}

	input := inputs[0]
	_, inputFeatures := input.Rows, input.Cols

	// 验证输入维度
	if inputFeatures != op.InputFeatures {
		panic(fmt.Sprintf("输入特征数不匹配：期望 %d，得到 %d",
			op.InputFeatures, inputFeatures))
	}

	// 缓存输入用于反向传播
	op.cachedInput = input.Copy()

	// 执行矩阵乘法：output = input * weights
	// input: (batch_size, input_features)
	// weights: (input_features, output_features)
	// result: (batch_size, output_features)
	matmulResult := input.Mul(op.weights)

	// 添加偏置：output = matmul_result + biases
	// 使用广播操作
	output := op.addBiases(matmulResult)

	return output
}

// addBiases 将偏置添加到矩阵乘法结果中（广播操作）
func (op *DenseOp) addBiases(matmulResult *matrix.Matrix) *matrix.Matrix {
	batchSize := matmulResult.Rows
	outputFeatures := matmulResult.Cols

	// 创建输出矩阵
	output := matrix.NewMatrix(batchSize, outputFeatures)

	// 为每个批次添加偏置
	for b := 0; b < batchSize; b++ {
		for f := 0; f < outputFeatures; f++ {
			matmulVal := matmulResult.At(b, f)
			biasVal := op.biases.At(0, f)
			output.Set(b, f, matmulVal+biasVal)
		}
	}

	return output
}

// Backward 反向传播
// 输入：gradOutput (batch_size, output_features) - 来自上层的梯度
// 输出：gradInput (batch_size, input_features) - 传递给下层的梯度
func (op *DenseOp) Backward(gradOutput *matrix.Matrix, inputs ...*Node) []*matrix.Matrix {
	if op.cachedInput == nil {
		panic("必须先执行前向传播才能进行反向传播")
	}

	_, outputFeatures := gradOutput.Rows, gradOutput.Cols
	if outputFeatures != op.OutputFeatures {
		panic(fmt.Sprintf("输出梯度特征数不匹配：期望 %d，得到 %d",
			op.OutputFeatures, outputFeatures))
	}

	// 1. 计算输入梯度：gradInput = gradOutput * weights^T
	weightsT := op.weights.T()
	gradInput := gradOutput.Mul(weightsT)

	// 2. 计算权重梯度：gradWeights = input^T * gradOutput
	op.computeWeightGradients(gradOutput)

	// 3. 计算偏置梯度：gradBiases = sum(gradOutput, axis=0)
	op.computeBiasGradients(gradOutput)

	return []*matrix.Matrix{gradInput}
}

// computeWeightGradients 计算权重梯度
func (op *DenseOp) computeWeightGradients(gradOutput *matrix.Matrix) {
	// 初始化权重梯度（如果需要）
	if op.weightGradients == nil {
		op.weightGradients = matrix.NewMatrix(op.InputFeatures, op.OutputFeatures)
	}

	// 计算权重梯度：gradWeights = input^T * gradOutput
	inputT := op.cachedInput.T()
	gradWeights := inputT.Mul(gradOutput)

	// 累积梯度
	op.weightGradients = op.weightGradients.Add(gradWeights)
}

// computeBiasGradients 计算偏置梯度
func (op *DenseOp) computeBiasGradients(gradOutput *matrix.Matrix) {
	// 初始化偏置梯度（如果需要）
	if op.biasGradients == nil {
		op.biasGradients = matrix.NewMatrix(1, op.OutputFeatures)
	}

	batchSize := gradOutput.Rows
	outputFeatures := gradOutput.Cols

	// 沿批次维度求和
	for f := 0; f < outputFeatures; f++ {
		gradSum := 0.0
		for b := 0; b < batchSize; b++ {
			gradSum += gradOutput.At(b, f)
		}

		// 累积偏置梯度
		currentGrad := op.biasGradients.At(0, f)
		op.biasGradients.Set(0, f, currentGrad+gradSum)
	}
}

// Name 操作名称
func (op *DenseOp) Name() string {
	return fmt.Sprintf("Dense(%d->%d)", op.InputFeatures, op.OutputFeatures)
}

// GetWeights 获取权重引用
func (op *DenseOp) GetWeights() *matrix.Matrix {
	return op.weights
}

// GetBiases 获取偏置引用
func (op *DenseOp) GetBiases() *matrix.Matrix {
	return op.biases
}

// GetWeightGradients 获取权重梯度副本
func (op *DenseOp) GetWeightGradients() *matrix.Matrix {
	if op.weightGradients == nil {
		return nil
	}
	return op.weightGradients.Copy()
}

// GetBiasGradients 获取偏置梯度副本
func (op *DenseOp) GetBiasGradients() *matrix.Matrix {
	if op.biasGradients == nil {
		return nil
	}
	return op.biasGradients.Copy()
}

// ZeroGradients 清零梯度
func (op *DenseOp) ZeroGradients() {
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
