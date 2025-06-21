package main

import (
	"fmt"
	"github.com/user/go-cnn/graph"
	"log"
	"time"

	"github.com/user/go-cnn/data"
	"github.com/user/go-cnn/inference"
	"github.com/user/go-cnn/models"
	"github.com/user/go-cnn/storage"
)

// RunTraining 执行MNIST模型训练
func RunTraining() {
	fmt.Println("开始训练MNIST手写数字识别CNN...")

	// 训练超参数
	const (
		batchSize    = 32
		numEpochs    = 2
		learningRate = 0.017
		trainPath    = "../train.csv"
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

	// 2. 创建模型构建器和模型
	fmt.Println("构建CNN网络架构...")
	config := models.MNISTModelConfig{
		BatchSize:    batchSize,
		LearningRate: learningRate,
		InputSize:    784,
		OutputSize:   10,
	}

	builder := models.NewMNISTModelBuilder(config)
	model, input, output := builder.BuildForTraining()

	paramCount := model.GetParameterCount()
	fmt.Printf("网络构建完成: %d 个可训练参数, 学习率: %.4f\n", paramCount, learningRate)

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

			// 1. 清零梯度（统一管理）
			model.ZeroGrad()

			// 2. 前向传播（自动）
			outputResult := model.Forward(batchData)

			// 3. 计算损失
			targets := graph.Input(batchLabels, "targets")
			lossNode := graph.SoftmaxCrossEntropyLoss(output, targets, true)

			// 损失值已经在创建节点时计算
			batchLoss := lossNode.Value.At(0, 0)
			epochLoss += batchLoss

			// 4. 反向传播
			lossNode.Backward()

			// 5. 参数更新
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

			// 避免未使用变量警告
			_ = outputResult
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

	// 5.5. 保存训练好的模型
	fmt.Println("保存模型...")
	saver := storage.NewJSONModelSaver()
	modelPath := "mnist_model.json"
	err = saver.SaveModel(model, modelPath)
	if err != nil {
		log.Printf("保存模型失败: %v", err)
	}

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
