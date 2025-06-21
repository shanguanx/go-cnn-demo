package main

import (
	"fmt"
	"log"
	"time"

	"github.com/user/go-cnn/data"
	"github.com/user/go-cnn/graph"
	"github.com/user/go-cnn/inference"
	"github.com/user/go-cnn/matrix"
	"github.com/user/go-cnn/optimizers"
)

func main() {
	fmt.Println("开始训练MNIST手写数字识别CNN...")

	// 训练超参数
	const (
		batchSize    = 32
		numEpochs    = 10
		learningRate = 0.017
		trainPath    = "/Users/dxm/Desktop/dl/digit-recognizer/train.csv"
	)

	// 1. 加载训练数据
	fmt.Println("加载训练数据...")
	trainDataset, err := data.LoadMNISTFromCSV(trainPath, true)
	if err != nil {
		log.Fatalf("加载训练数据失败: %v", err)
	}
	fmt.Printf("训练数据加载完成: %d 个样本\n", trainDataset.Len())

	// 创建数据加载器
	trainLoader := data.NewDataLoader(trainDataset, batchSize)
	fmt.Printf("数据加载器创建完成: %d 个批次\n", trainLoader.NumBatches())

	// 2. 创建优化器
	optimizer := optimizers.NewSGD(learningRate)
	fmt.Printf("SGD优化器创建完成: 学习率 = %.4f\n", learningRate)

	// 3. 创建模型
	model := graph.NewModel(optimizer)
	fmt.Println("模型创建完成")

	// 4. 构建CNN网络架构
	fmt.Println("构建CNN网络架构...")
	// 创建输入节点 (将在训练循环中设置实际数据)
	inputData := matrix.NewMatrix(batchSize, 784)
	input := graph.Input(inputData, "input")

	// 构建简化版LeNet网络
	output := buildSimpleCNN(input)

	// 设置模型输出并收集参数
	model.SetOutput(output)
	model.CollectParameters(output)

	paramCount := model.GetParameterCount()
	fmt.Printf("网络构建完成: %d 个可训练参数\n", paramCount)

	// 创建推理器
	inferencer := inference.NewInferencer(input, output, batchSize)
	fmt.Println("推理器创建完成")

	// 5. 开始训练
	fmt.Println("开始训练...")
	startTime := time.Now()

	for epoch := 0; epoch < numEpochs; epoch++ {
		epochStartTime := time.Now()
		trainLoader.Reset()

		epochLoss := 0.0
		epochCorrect := 0
		epochTotal := 0
		batchCount := 0

		// 训练一个epoch
		for {
			batchData, batchLabels, hasMore := trainLoader.Next()
			if !hasMore {
				break
			}

			// 跳过不完整的批次（方案3：简单跳过）
			if batchData.Rows != batchSize {
				fmt.Printf("跳过不完整批次: %d 个样本（期望 %d）\n", batchData.Rows, batchSize)
				continue
			}

			// 前向传播
			// 更新输入节点的数据
			input.Value = batchData

			// 重新计算整个网络的前向传播
			recalculateForward(output)

			// 计算损失
			targets := graph.Input(batchLabels, "targets")
			lossNode := graph.SoftmaxCrossEntropyLoss(output, targets, true)

			// 损失值已经在创建节点时计算
			batchLoss := lossNode.Value.At(0, 0)
			epochLoss += batchLoss

			// 反向传播
			model.ZeroGrad()
			lossNode.Backward()

			// 参数更新
			model.Step()

			batchCount++

			// 每100个批次打印一次进度（包含准确率）
			if batchCount%100 == 0 {
				// 计算准确率并累积到epoch统计中
				result := inferencer.InferWithMetrics(batchData, batchLabels)
				batchCorrect := int(result.Accuracy * float64(batchSize) / 100.0)
				epochCorrect += batchCorrect
				epochTotal += batchSize

				fmt.Printf("Epoch %d, Batch %d, Loss: %.4f, Accuracy: %.2f%%\n",
					epoch+1, batchCount, batchLoss, result.Accuracy)
			}
		}

		// Epoch结束统计
		avgLoss := epochLoss / float64(batchCount)
		epochDuration := time.Since(epochStartTime)

		// 计算epoch准确率（基于每100个batch的累积统计）
		epochAccuracy := 0.0
		if epochTotal > 0 {
			epochAccuracy = float64(epochCorrect) / float64(epochTotal) * 100.0
		}

		fmt.Printf("Epoch %d/%d 完成 - 平均损失: %.4f, 训练准确率: %.2f%% (%d/%d), 用时: %v\n",
			epoch+1, numEpochs, avgLoss, epochAccuracy, epochCorrect, epochTotal, epochDuration)
	}

	totalDuration := time.Since(startTime)
	fmt.Printf("训练完成! 总用时: %v\n", totalDuration)

	// 6. 简单验证（使用推理器）
	fmt.Println("开始验证...")
	validateWithInferencer(inferencer, trainDataset, batchSize)
}

// validateWithInferencer 使用推理器进行验证
func validateWithInferencer(inferencer *inference.ModelInferencer, dataset *data.MNISTDataset, batchSize int) {
	fmt.Println("使用推理器进行模型验证...")

	// 使用推理器的完整评估功能
	// 这里只评估部分数据以节省时间
	validationBatchSize := batchSize
	result := inferencer.Evaluate(dataset, validationBatchSize)

	// 打印详细的评估结果
	inference.PrintEvaluationResult(result)
}

// buildSimpleCNN 构建简化的CNN网络
func buildSimpleCNN(input *graph.Node) *graph.Node {
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
	// 注意：这里使用固定批大小，实际训练中需要处理动态批大小
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

// recalculateForward 重新计算前向传播
func recalculateForward(node *graph.Node) {
	// 1. 清零所有节点的梯度状态（重要！）
	clearGradientStates(node, make(map[*graph.Node]bool))

	// 2. 重新计算前向传播
	recalculateNodeRecursive(node, make(map[*graph.Node]bool))
}

// clearGradientStates 清零所有节点的梯度状态
func clearGradientStates(node *graph.Node, visited map[*graph.Node]bool) {
	if visited[node] {
		return
	}
	visited[node] = true

	// 清零梯度相关状态
	node.Gradient = nil
	node.GradComputed = false

	// 递归清零输入节点
	for _, input := range node.Inputs {
		clearGradientStates(input, visited)
	}
}

// recalculateNodeRecursive 递归重新计算节点值
func recalculateNodeRecursive(node *graph.Node, visited map[*graph.Node]bool) {
	if visited[node] {
		return
	}
	visited[node] = true

	// 先计算输入节点
	for _, input := range node.Inputs {
		recalculateNodeRecursive(input, visited)
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
