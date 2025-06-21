package main

import (
	"fmt"
	"log"

	"github.com/user/go-cnn/data"
	"github.com/user/go-cnn/inference"
	"github.com/user/go-cnn/models"
	"github.com/user/go-cnn/storage"
)

// RunInference 执行MNIST模型推理
func RunInference() {
	fmt.Println("MNIST手写数字识别 - 推理模式")
	fmt.Println("================================")

	// 配置参数
	const (
		modelPath = "mnist_model.json"
		testPath  = "../train.csv" // 使用训练数据的一部分作为测试
		batchSize = 32
	)

	// 1. 使用模型构建器重建网络结构（必须与训练时完全相同）
	fmt.Println("重建CNN网络结构...")
	config := models.MNISTModelConfig{
		BatchSize:    batchSize,
		LearningRate: 0.001, // 推理模式下学习率无关紧要
		InputSize:    784,
		OutputSize:   10,
	}

	builder := models.NewMNISTModelBuilder(config)
	model, input, output := builder.BuildForInference()

	fmt.Printf("网络结构已重建: %d 个可训练参数组\n", model.GetParameterCount())

	// 3. 加载保存的模型参数
	fmt.Printf("从 %s 加载模型参数...\n", modelPath)
	saver := storage.NewJSONModelSaver()
	err := saver.LoadModel(model, modelPath)
	if err != nil {
		log.Fatalf("加载模型失败: %v", err)
	}
	fmt.Println("✅ 模型参数加载成功!")

	// 4. 创建推理器
	inferencer := inference.NewInferencer(input, output, batchSize)
	fmt.Println("✅ 推理器创建完成")

	// 5. 加载测试数据
	fmt.Printf("从 %s 加载测试数据...\n", testPath)
	testDataset, err := data.LoadMNISTFromCSV(testPath, true)
	if err != nil {
		log.Fatalf("加载测试数据失败: %v", err)
	}
	fmt.Printf("✅ 测试数据加载完成: %d 个样本\n", testDataset.Len())

	// 6. 执行推理测试
	fmt.Println("\n开始推理测试...")
	fmt.Println("================================")

	// 6.1 快速测试 - 前几个批次
	performQuickTest(inferencer, testDataset, batchSize)

	// 6.2 详细评估 - 更全面的性能分析
	fmt.Println("\n进行详细性能评估...")
	performDetailedEvaluation(inferencer, testDataset, batchSize)

	fmt.Println("\n🎉 推理测试完成!")
}

// performQuickTest 快速测试前几个批次
func performQuickTest(inferencer *inference.ModelInferencer, dataset *data.MNISTDataset, batchSize int) {
	loader := data.NewDataLoader(dataset, batchSize)
	loader.Reset()

	fmt.Println("快速测试 - 前5个批次:")
	fmt.Println("批次\t准确率\t\t损失预估")
	fmt.Println("----\t------\t\t--------")

	totalCorrect := 0
	totalSamples := 0
	maxBatches := 5

	for i := 0; i < maxBatches; i++ {
		batchData, batchLabels, hasMore := loader.Next()
		if !hasMore {
			break
		}

		if batchData.Rows != batchSize {
			continue
		}

		// 推理
		result := inferencer.InferWithMetrics(batchData, batchLabels)
		batchCorrect := int(result.Accuracy * float64(batchSize) / 100.0)
		totalCorrect += batchCorrect
		totalSamples += batchSize

		// 简单的损失估算（基于错误率）
		estimatedLoss := (100.0 - result.Accuracy) / 100.0 * 2.3 // 粗略估算

		fmt.Printf("%d\t%.2f%%\t\t%.4f\n", i+1, result.Accuracy, estimatedLoss)
	}

	overallAccuracy := float64(totalCorrect) / float64(totalSamples) * 100.0
	fmt.Printf("\n快速测试总结: %.2f%% (%d/%d)\n", overallAccuracy, totalCorrect, totalSamples)
}

// performDetailedEvaluation 详细性能评估
func performDetailedEvaluation(inferencer *inference.ModelInferencer, dataset *data.MNISTDataset, batchSize int) {
	// 使用推理器的完整评估功能
	result := inferencer.Evaluate(dataset, batchSize)

	// 打印详细结果
	inference.PrintEvaluationResult(result)
}
