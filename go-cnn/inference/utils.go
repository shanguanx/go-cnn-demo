package inference

import (
	"fmt"
	"math"

	"github.com/user/go-cnn/matrix"
)

// Argmax 找到最大值索引
func Argmax(predictions *matrix.Matrix, rowIndex int) int {
	maxIdx := 0
	maxVal := predictions.At(rowIndex, 0)
	for j := 1; j < predictions.Cols; j++ {
		if predictions.At(rowIndex, j) > maxVal {
			maxVal = predictions.At(rowIndex, j)
			maxIdx = j
		}
	}
	return maxIdx
}

// ArgmaxBatch 批量计算argmax
func ArgmaxBatch(predictions *matrix.Matrix) []int {
	result := make([]int, predictions.Rows)
	for i := 0; i < predictions.Rows; i++ {
		result[i] = Argmax(predictions, i)
	}
	return result
}

// Softmax 计算softmax概率（如果需要概率输出）
func Softmax(logits *matrix.Matrix) *matrix.Matrix {
	result := matrix.NewMatrix(logits.Rows, logits.Cols)

	for i := 0; i < logits.Rows; i++ {
		// 找到最大值以提高数值稳定性
		maxVal := logits.At(i, 0)
		for j := 1; j < logits.Cols; j++ {
			if logits.At(i, j) > maxVal {
				maxVal = logits.At(i, j)
			}
		}

		// 计算exp(x - max)的和
		sum := 0.0
		for j := 0; j < logits.Cols; j++ {
			expVal := math.Exp(logits.At(i, j) - maxVal)
			result.Set(i, j, expVal)
			sum += expVal
		}

		// 归一化
		for j := 0; j < logits.Cols; j++ {
			result.Set(i, j, result.At(i, j)/sum)
		}
	}

	return result
}

// GetTopK 获取Top-K预测
func GetTopK(predictions *matrix.Matrix, rowIndex, k int) []int {
	if k > predictions.Cols {
		k = predictions.Cols
	}

	// 创建索引-值对
	type IndexValue struct {
		Index int
		Value float64
	}

	pairs := make([]IndexValue, predictions.Cols)
	for j := 0; j < predictions.Cols; j++ {
		pairs[j] = IndexValue{
			Index: j,
			Value: predictions.At(rowIndex, j),
		}
	}

	// 简单的选择排序获取top-k
	result := make([]int, k)
	for i := 0; i < k; i++ {
		maxIdx := i
		for j := i + 1; j < len(pairs); j++ {
			if pairs[j].Value > pairs[maxIdx].Value {
				maxIdx = j
			}
		}
		// 交换
		pairs[i], pairs[maxIdx] = pairs[maxIdx], pairs[i]
		result[i] = pairs[i].Index
	}

	return result
}

// CalculateAccuracy 计算批量准确率
func CalculateAccuracy(predictions, labels *matrix.Matrix) (correct, total int) {
	total = predictions.Rows
	for i := 0; i < total; i++ {
		predictedLabel := Argmax(predictions, i)
		trueLabel := int(labels.At(i, 0))

		if predictedLabel == trueLabel {
			correct++
		}
	}
	return correct, total
}

// PrintEvaluationResult 格式化打印评估结果
func PrintEvaluationResult(result EvaluationResult) {
	fmt.Printf("=== 模型评估结果 ===\n")
	fmt.Printf("总准确率: %.2f%% (%d/%d)\n",
		result.Accuracy, result.CorrectSamples, result.TotalSamples)
	fmt.Printf("平均置信度: %.4f\n", result.AverageConfidence)

	fmt.Printf("\n各类别准确率:\n")
	for class := 0; class < 10; class++ {
		if acc, exists := result.ClassAccuracy[class]; exists {
			fmt.Printf("  数字 %d: %.2f%%\n", class, acc)
		}
	}

	fmt.Printf("\n混淆矩阵:\n")
	fmt.Printf("真实\\预测 ")
	for j := 0; j < 10; j++ {
		fmt.Printf("%4d", j)
	}
	fmt.Printf("\n")

	for i := 0; i < 10; i++ {
		fmt.Printf("%8d ", i)
		for j := 0; j < 10; j++ {
			fmt.Printf("%4d", result.ConfusionMatrix[i][j])
		}
		fmt.Printf("\n")
	}
}

// PrintInferenceResult 格式化打印推理结果
func PrintInferenceResult(result InferenceResult) {
	fmt.Printf("=== 推理结果 ===\n")
	fmt.Printf("批大小: %d\n", len(result.PredictedLabels))
	if result.Accuracy > 0 {
		fmt.Printf("准确率: %.2f%%\n", result.Accuracy)
	}

	// 打印前几个样本的预测结果
	maxSamples := 10
	if len(result.PredictedLabels) < maxSamples {
		maxSamples = len(result.PredictedLabels)
	}

	fmt.Printf("\n前 %d 个样本的预测:\n", maxSamples)
	for i := 0; i < maxSamples; i++ {
		fmt.Printf("样本 %d: 预测=%d, 置信度=%.4f\n",
			i, result.PredictedLabels[i], result.Confidence[i])
	}
}
