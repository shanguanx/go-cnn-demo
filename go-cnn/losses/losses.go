package losses

import (
	"errors"
	"math"

	"github.com/user/go-cnn/matrix"
)

// CrossEntropyLoss 计算交叉熵损失
// predictions: 预测值矩阵 (batch_size, num_classes)，经过softmax处理
// targets: 目标值矩阵 (batch_size, num_classes)，one-hot编码或标量索引
// 返回标量损失值
func CrossEntropyLoss(predictions, targets *matrix.Matrix) (float64, error) {
	if predictions.Rows != targets.Rows {
		return 0, errors.New("预测值和目标值的批量大小不匹配")
	}

	var loss float64
	batchSize := float64(predictions.Rows)
	epsilon := 1e-15 // 防止log(0)

	// 处理one-hot编码的目标值
	if predictions.Cols == targets.Cols {
		for i := 0; i < predictions.Rows; i++ {
			for j := 0; j < predictions.Cols; j++ {
				pred := math.Max(predictions.At(i, j), epsilon)
				pred = math.Min(pred, 1.0-epsilon)
				target := targets.At(i, j)
				if target > 0 {
					loss -= target * math.Log(pred)
				}
			}
		}
	} else if targets.Cols == 1 {
		// 处理标量索引的目标值
		for i := 0; i < predictions.Rows; i++ {
			targetIdx := int(targets.At(i, 0))
			if targetIdx < 0 || targetIdx >= predictions.Cols {
				return 0, errors.New("目标索引超出范围")
			}
			pred := math.Max(predictions.At(i, targetIdx), epsilon)
			loss -= math.Log(pred)
		}
	} else {
		return 0, errors.New("目标值维度不正确")
	}

	return loss / batchSize, nil
}

// CrossEntropyLossDerivative 计算交叉熵损失对预测值的导数
// predictions: 预测值矩阵 (batch_size, num_classes)
// targets: 目标值矩阵 (batch_size, num_classes) 或 (batch_size, 1)
// 返回梯度矩阵，与predictions形状相同
func CrossEntropyLossDerivative(predictions, targets *matrix.Matrix) (*matrix.Matrix, error) {
	if predictions.Rows != targets.Rows {
		return nil, errors.New("预测值和目标值的批量大小不匹配")
	}

	gradient := matrix.Zeros(predictions.Rows, predictions.Cols)
	batchSize := float64(predictions.Rows)
	epsilon := 1e-15

	// 处理one-hot编码的目标值
	if predictions.Cols == targets.Cols {
		for i := 0; i < predictions.Rows; i++ {
			for j := 0; j < predictions.Cols; j++ {
				pred := math.Max(predictions.At(i, j), epsilon)
				pred = math.Min(pred, 1.0-epsilon)
				target := targets.At(i, j)
				gradient.Set(i, j, (pred-target)/batchSize)
			}
		}
	} else if targets.Cols == 1 {
		// 处理标量索引的目标值
		for i := 0; i < predictions.Rows; i++ {
			targetIdx := int(targets.At(i, 0))
			if targetIdx < 0 || targetIdx >= predictions.Cols {
				return nil, errors.New("目标索引超出范围")
			}
			
			// 对于所有类别，梯度为 pred/batch_size
			for j := 0; j < predictions.Cols; j++ {
				pred := math.Max(predictions.At(i, j), epsilon)
				pred = math.Min(pred, 1.0-epsilon)
				if j == targetIdx {
					// 对于正确类别，梯度为 (pred - 1) / batch_size
					gradient.Set(i, j, (pred-1.0)/batchSize)
				} else {
					// 对于错误类别，梯度为 pred / batch_size
					gradient.Set(i, j, pred/batchSize)
				}
			}
		}
	} else {
		return nil, errors.New("目标值维度不正确")
	}

	return gradient, nil
}

// SoftmaxCrossEntropyLoss 结合Softmax和交叉熵的优化实现
// logits: 未经softmax处理的原始输出 (batch_size, num_classes)
// targets: 目标值矩阵 (batch_size, num_classes) 或 (batch_size, 1)
// 返回损失值和softmax概率
func SoftmaxCrossEntropyLoss(logits, targets *matrix.Matrix) (float64, *matrix.Matrix, error) {
	if logits.Rows != targets.Rows {
		return 0, nil, errors.New("输入和目标值的批量大小不匹配")
	}

	// 计算softmax概率
	softmaxProbs := matrix.Zeros(logits.Rows, logits.Cols)
	var totalLoss float64

	for i := 0; i < logits.Rows; i++ {
		// 找到每行的最大值以提高数值稳定性
		maxVal := logits.At(i, 0)
		for j := 1; j < logits.Cols; j++ {
			if logits.At(i, j) > maxVal {
				maxVal = logits.At(i, j)
			}
		}

		// 计算exp和sum
		var sumExp float64
		for j := 0; j < logits.Cols; j++ {
			expVal := math.Exp(logits.At(i, j) - maxVal)
			softmaxProbs.Set(i, j, expVal)
			sumExp += expVal
		}

		// 归一化得到概率
		for j := 0; j < logits.Cols; j++ {
			prob := softmaxProbs.At(i, j) / sumExp
			softmaxProbs.Set(i, j, prob)
		}

		// 计算当前样本的损失
		if targets.Cols == 1 {
			// 标量索引目标
			targetIdx := int(targets.At(i, 0))
			if targetIdx < 0 || targetIdx >= logits.Cols {
				return 0, nil, errors.New("目标索引超出范围")
			}
			prob := math.Max(softmaxProbs.At(i, targetIdx), 1e-15)
			totalLoss -= math.Log(prob)
		} else if targets.Cols == logits.Cols {
			// one-hot编码目标
			for j := 0; j < targets.Cols; j++ {
				if targets.At(i, j) > 0 {
					prob := math.Max(softmaxProbs.At(i, j), 1e-15)
					totalLoss -= targets.At(i, j) * math.Log(prob)
				}
			}
		} else {
			return 0, nil, errors.New("目标值维度不正确")
		}
	}

	avgLoss := totalLoss / float64(logits.Rows)
	return avgLoss, softmaxProbs, nil
}

// SoftmaxCrossEntropyLossDerivative 计算Softmax+交叉熵的联合导数
// softmaxProbs: softmax概率 (batch_size, num_classes)
// targets: 目标值矩阵 (batch_size, num_classes) 或 (batch_size, 1)
// 返回对logits的梯度
func SoftmaxCrossEntropyLossDerivative(softmaxProbs, targets *matrix.Matrix) (*matrix.Matrix, error) {
	if softmaxProbs.Rows != targets.Rows {
		return nil, errors.New("概率和目标值的批量大小不匹配")
	}

	gradient := matrix.Zeros(softmaxProbs.Rows, softmaxProbs.Cols)
	batchSize := float64(softmaxProbs.Rows)

	if targets.Cols == 1 {
		// 标量索引目标
		for i := 0; i < softmaxProbs.Rows; i++ {
			targetIdx := int(targets.At(i, 0))
			if targetIdx < 0 || targetIdx >= softmaxProbs.Cols {
				return nil, errors.New("目标索引超出范围")
			}
			
			for j := 0; j < softmaxProbs.Cols; j++ {
				if j == targetIdx {
					gradient.Set(i, j, (softmaxProbs.At(i, j)-1.0)/batchSize)
				} else {
					gradient.Set(i, j, softmaxProbs.At(i, j)/batchSize)
				}
			}
		}
	} else if targets.Cols == softmaxProbs.Cols {
		// one-hot编码目标
		for i := 0; i < softmaxProbs.Rows; i++ {
			for j := 0; j < softmaxProbs.Cols; j++ {
				gradient.Set(i, j, (softmaxProbs.At(i, j)-targets.At(i, j))/batchSize)
			}
		}
	} else {
		return nil, errors.New("目标值维度不正确")
	}

	return gradient, nil
}

// MeanSquaredErrorLoss 计算均方误差损失
// predictions: 预测值矩阵
// targets: 目标值矩阵
// 返回MSE损失值
func MeanSquaredErrorLoss(predictions, targets *matrix.Matrix) (float64, error) {
	if predictions.Rows != targets.Rows || predictions.Cols != targets.Cols {
		return 0, errors.New("预测值和目标值的形状不匹配")
	}

	var sumSquaredError float64
	totalElements := float64(predictions.Rows * predictions.Cols)

	for i := 0; i < len(predictions.Data); i++ {
		diff := predictions.Data[i] - targets.Data[i]
		sumSquaredError += diff * diff
	}

	return sumSquaredError / totalElements, nil
}

// MeanSquaredErrorLossDerivative 计算MSE损失的导数
// predictions: 预测值矩阵
// targets: 目标值矩阵
// 返回梯度矩阵
func MeanSquaredErrorLossDerivative(predictions, targets *matrix.Matrix) (*matrix.Matrix, error) {
	if predictions.Rows != targets.Rows || predictions.Cols != targets.Cols {
		return nil, errors.New("预测值和目标值的形状不匹配")
	}

	gradient := matrix.Zeros(predictions.Rows, predictions.Cols)
	totalElements := float64(predictions.Rows * predictions.Cols)

	for i := 0; i < len(predictions.Data); i++ {
		// MSE导数: 2 * (pred - target) / n
		gradient.Data[i] = 2.0 * (predictions.Data[i] - targets.Data[i]) / totalElements
	}

	return gradient, nil
}

// BinaryCrossEntropyLoss 计算二分类交叉熵损失
// predictions: 预测概率 (batch_size, 1) 或 (batch_size,)，值域[0,1]
// targets: 目标标签 (batch_size, 1) 或 (batch_size,)，值为0或1
func BinaryCrossEntropyLoss(predictions, targets *matrix.Matrix) (float64, error) {
	if predictions.Rows != targets.Rows {
		return 0, errors.New("预测值和目标值的批量大小不匹配")
	}

	var loss float64
	batchSize := float64(predictions.Rows)
	epsilon := 1e-15

	for i := 0; i < predictions.Rows; i++ {
		for j := 0; j < predictions.Cols; j++ {
			pred := math.Max(predictions.At(i, j), epsilon)
			pred = math.Min(pred, 1.0-epsilon)
			target := targets.At(i, j)
			
			// BCE = -[y*log(p) + (1-y)*log(1-p)]
			loss -= target*math.Log(pred) + (1.0-target)*math.Log(1.0-pred)
		}
	}

	return loss / batchSize, nil
}

// BinaryCrossEntropyLossDerivative 计算二分类交叉熵损失的导数
// predictions: 预测概率
// targets: 目标标签
func BinaryCrossEntropyLossDerivative(predictions, targets *matrix.Matrix) (*matrix.Matrix, error) {
	if predictions.Rows != targets.Rows || predictions.Cols != targets.Cols {
		return nil, errors.New("预测值和目标值的形状不匹配")
	}

	gradient := matrix.Zeros(predictions.Rows, predictions.Cols)
	batchSize := float64(predictions.Rows)
	epsilon := 1e-15

	for i := 0; i < predictions.Rows; i++ {
		for j := 0; j < predictions.Cols; j++ {
			pred := math.Max(predictions.At(i, j), epsilon)
			pred = math.Min(pred, 1.0-epsilon)
			target := targets.At(i, j)
			
			// BCE导数: (pred - target) / [pred * (1 - pred)] / batch_size
			// 简化为: (pred - target) / (pred * (1 - pred)) / batch_size
			gradient.Set(i, j, (pred-target)/(pred*(1.0-pred))/batchSize)
		}
	}

	return gradient, nil
}