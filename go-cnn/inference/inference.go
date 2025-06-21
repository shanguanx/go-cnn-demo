package inference

import (
	"fmt"

	"github.com/user/go-cnn/data"
	"github.com/user/go-cnn/graph"
	"github.com/user/go-cnn/matrix"
)

// EvaluationResult 评估结果
type EvaluationResult struct {
	Accuracy          float64         // 总准确率
	TotalSamples      int             // 总样本数
	CorrectSamples    int             // 正确样本数
	ClassAccuracy     map[int]float64 // 各类别准确率
	ConfusionMatrix   [][]int         // 混淆矩阵
	AverageConfidence float64         // 平均置信度
}

// InferenceResult 单次推理结果
type InferenceResult struct {
	Predictions     *matrix.Matrix // 预测概率/logits
	PredictedLabels []int          // 预测标签
	Accuracy        float64        // 准确率（如果有真实标签）
	Confidence      []float64      // 每个样本的置信度
}

// Inferencer 推理器接口
type Inferencer interface {
	// 单样本推理
	Predict(input *matrix.Matrix) *matrix.Matrix

	// 批量推理
	PredictBatch(inputs *matrix.Matrix) *matrix.Matrix

	// 带准确率评估的推理
	Evaluate(dataset data.Dataset, batchSize int) EvaluationResult

	// 推理 + 指标计算
	InferWithMetrics(inputs, labels *matrix.Matrix) InferenceResult
}

// ModelInferencer 模型推理器
type ModelInferencer struct {
	inputNode  *graph.Node
	outputNode *graph.Node
	batchSize  int
}

// NewInferencer 创建推理器
func NewInferencer(inputNode, outputNode *graph.Node, batchSize int) *ModelInferencer {
	return &ModelInferencer{
		inputNode:  inputNode,
		outputNode: outputNode,
		batchSize:  batchSize,
	}
}

// Predict 单样本推理
func (inf *ModelInferencer) Predict(input *matrix.Matrix) *matrix.Matrix {
	if input.Rows != 1 {
		panic("Predict方法只支持单样本输入 (shape: [1, features])")
	}
	return inf.PredictBatch(input)
}

// PredictBatch 批量推理（核心方法）
func (inf *ModelInferencer) PredictBatch(inputs *matrix.Matrix) *matrix.Matrix {
	// 1. 验证输入形状
	if inputs.Rows > inf.batchSize {
		panic(fmt.Sprintf("输入批大小 %d 超过模型设计的批大小 %d", inputs.Rows, inf.batchSize))
	}

	// 2. 设置输入数据
	inf.inputNode.Value = inputs

	// 3. 前向传播（无梯度）
	inf.forwardInferenceOnly()

	// 4. 返回预测结果
	return inf.outputNode.Value.Copy()
}

// InferWithMetrics 推理 + 指标计算
func (inf *ModelInferencer) InferWithMetrics(inputs, labels *matrix.Matrix) InferenceResult {
	// 1. 执行推理
	predictions := inf.PredictBatch(inputs)

	// 2. 计算预测标签
	predictedLabels := make([]int, predictions.Rows)
	confidence := make([]float64, predictions.Rows)
	correct := 0

	for i := 0; i < predictions.Rows; i++ {
		// 找到预测标签
		predictedLabels[i] = Argmax(predictions, i)

		// 计算置信度（最大概率值）
		confidence[i] = predictions.At(i, predictedLabels[i])

		// 如果有标签，计算准确率
		if labels != nil {
			trueLabel := int(labels.At(i, 0))
			if predictedLabels[i] == trueLabel {
				correct++
			}
		}
	}

	accuracy := 0.0
	if labels != nil {
		accuracy = float64(correct) / float64(predictions.Rows) * 100.0
	}

	return InferenceResult{
		Predictions:     predictions,
		PredictedLabels: predictedLabels,
		Accuracy:        accuracy,
		Confidence:      confidence,
	}
}

// Evaluate 在数据集上完整评估
func (inf *ModelInferencer) Evaluate(dataset data.Dataset, evalBatchSize int) EvaluationResult {
	loader := data.NewDataLoader(dataset, evalBatchSize)

	totalCorrect := 0
	totalSamples := 0
	classCorrect := make(map[int]int)
	classTotal := make(map[int]int)

	// 初始化混淆矩阵 (10x10 for MNIST)
	confusionMatrix := make([][]int, 10)
	for i := range confusionMatrix {
		confusionMatrix[i] = make([]int, 10)
	}

	totalConfidence := 0.0
	confidenceCount := 0

	loader.Reset()
	for {
		batchData, batchLabels, hasMore := loader.Next()
		if !hasMore {
			break
		}

		// 跳过不完整批次（与训练保持一致）
		if batchData.Rows != evalBatchSize {
			continue
		}

		// 推理
		result := inf.InferWithMetrics(batchData, batchLabels)

		// 累积统计
		batchCorrect := int(result.Accuracy * float64(batchData.Rows) / 100.0)
		totalCorrect += batchCorrect
		totalSamples += batchData.Rows

		// 累积置信度
		for _, conf := range result.Confidence {
			totalConfidence += conf
			confidenceCount++
		}

		// 更新类别统计和混淆矩阵
		for i := 0; i < batchData.Rows; i++ {
			trueLabel := int(batchLabels.At(i, 0))
			predictedLabel := result.PredictedLabels[i]

			classTotal[trueLabel]++
			if predictedLabel == trueLabel {
				classCorrect[trueLabel]++
			}

			// 更新混淆矩阵
			if trueLabel >= 0 && trueLabel < 10 && predictedLabel >= 0 && predictedLabel < 10 {
				confusionMatrix[trueLabel][predictedLabel]++
			}
		}
	}

	// 计算各类别准确率
	classAccuracy := make(map[int]float64)
	for class := 0; class < 10; class++ {
		if classTotal[class] > 0 {
			classAccuracy[class] = float64(classCorrect[class]) / float64(classTotal[class]) * 100.0
		} else {
			classAccuracy[class] = 0.0
		}
	}

	averageConfidence := 0.0
	if confidenceCount > 0 {
		averageConfidence = totalConfidence / float64(confidenceCount)
	}

	return EvaluationResult{
		Accuracy:          float64(totalCorrect) / float64(totalSamples) * 100.0,
		TotalSamples:      totalSamples,
		CorrectSamples:    totalCorrect,
		ClassAccuracy:     classAccuracy,
		ConfusionMatrix:   confusionMatrix,
		AverageConfidence: averageConfidence,
	}
}

// forwardInferenceOnly 推理专用前向传播（不计算梯度）
func (inf *ModelInferencer) forwardInferenceOnly() {
	// 清理状态但不设置梯度相关
	inf.clearInferenceStates(inf.outputNode, make(map[*graph.Node]bool))

	// 只计算前向传播，不准备梯度
	inf.recalculateInferenceForward(inf.outputNode, make(map[*graph.Node]bool))
}

// clearInferenceStates 清理推理状态（不涉及梯度）
func (inf *ModelInferencer) clearInferenceStates(node *graph.Node, visited map[*graph.Node]bool) {
	if visited[node] {
		return
	}
	visited[node] = true

	// 推理模式下不需要梯度相关清理
	// 只需要确保计算状态清洁

	for _, input := range node.Inputs {
		inf.clearInferenceStates(input, visited)
	}
}

// recalculateInferenceForward 递归重新计算节点值（推理模式）
func (inf *ModelInferencer) recalculateInferenceForward(node *graph.Node, visited map[*graph.Node]bool) {
	if visited[node] {
		return
	}
	visited[node] = true

	// 先计算输入节点
	for _, input := range node.Inputs {
		inf.recalculateInferenceForward(input, visited)
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
