package main

import (
	"fmt"
	"log"

	"github.com/user/go-cnn/activations"
	"github.com/user/go-cnn/graph"
	"github.com/user/go-cnn/matrix"
)

func main() {
	fmt.Println("=== 前向传播演示 ===")
	fmt.Println("展示数据如何经过一系列函数变换")
	fmt.Println()

	// 1. 创建输入数据：一个简单的3x3图像
	input := matrix.NewMatrixFromData([]float64{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	}, 3, 3)

	fmt.Println("1. 原始输入数据:")
	fmt.Println(input.String())
	fmt.Println()

	// 2. 第一个函数变换：卷积层
	fmt.Println("2. 第一个函数变换：卷积层")
	fmt.Println("函数：f1(x, W, b) = W ⊛ x + b")

	// 创建卷积层：1输入通道，2输出通道，2x2卷积核，步长1，填充0
	convOp := graph.NewConvOp(1, 2, 2, 1, 0)
	err := convOp.SetInputSize(3, 3)
	if err != nil {
		log.Fatal(err)
	}

	// 设置简单的权重进行演示
	// 第一个卷积核：[[1, 1], [1, 1]]
	// 第二个卷积核：[[0, 1], [1, 0]]
	weights := convOp.GetWeights()
	for i := 0; i < 4; i++ {
		weights.Set(0, i, 1.0) // 全1卷积核
	}
	weights.Set(1, 0, 0.0) // 第二个卷积核
	weights.Set(1, 1, 1.0)
	weights.Set(1, 2, 1.0)
	weights.Set(1, 3, 0.0)

	// 偏置设为0
	biases := convOp.GetBiases()
	biases.Set(0, 0, 0.0)
	biases.Set(1, 0, 0.0)

	// 准备输入：展平为(batch_size=1, channels*height*width=1*3*3=9)
	flatInput := matrix.NewMatrix(1, 9)
	for i := 0; i < 9; i++ {
		flatInput.Set(0, i, input.Data[i])
	}

	// 执行卷积变换
	convOutput := convOp.Forward(flatInput)

	fmt.Println("卷积层输出 (2个特征图，每个2x2):")
	fmt.Println(convOutput.String())
	fmt.Println()

	// 3. 第二个函数变换：ReLU激活
	fmt.Println("3. 第二个函数变换：ReLU激活")
	fmt.Println("函数：f2(x) = max(0, x)")

	reluOutput := activations.ReLU(convOutput)
	fmt.Println("ReLU激活后:")
	fmt.Println(reluOutput.String())
	fmt.Println()

	// 4. 第三个函数变换：全连接层（模拟）
	fmt.Println("4. 第三个函数变换：全连接层")
	fmt.Println("函数：f3(x, W, b) = Wx + b")

	// 模拟全连接层：将8个输入连接到3个输出
	// 权重矩阵：3x8
	fcWeights := matrix.NewMatrixFromData([]float64{
		0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
		0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
		0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
	}, 3, 8)

	fcBias := matrix.NewMatrixFromData([]float64{0.1, 0.2, 0.3}, 3, 1)

	// 执行全连接变换
	fcOutput := fcWeights.Mul(reluOutput.T()) // 注意转置
	fcOutput = fcOutput.Add(fcBias)

	fmt.Println("全连接层输出 (3个类别):")
	fmt.Println(fcOutput.String())
	fmt.Println()

	// 5. 第四个函数变换：Softmax
	fmt.Println("5. 第四个函数变换：Softmax")
	fmt.Println("函数：f4(x_i) = exp(x_i) / Σ exp(x_j)")

	softmaxOutput := activations.Softmax(fcOutput.T()) // 转置回来
	fmt.Println("Softmax输出 (概率分布):")
	fmt.Println(softmaxOutput.String())

	// 验证概率和为1
	sum := 0.0
	for i := 0; i < softmaxOutput.Cols; i++ {
		sum += softmaxOutput.At(0, i)
	}
	fmt.Printf("概率和: %.6f (应该接近1.0)\n", sum)
	fmt.Println()

	// 6. 总结前向传播过程
	fmt.Println("=== 前向传播总结 ===")
	fmt.Println("输入 → 卷积层 → ReLU → 全连接层 → Softmax → 输出")
	fmt.Println("  ↓      ↓        ↓        ↓         ↓       ↓")
	fmt.Println(" 3x3   2x2x2    2x2x2     3x1       3x1     3x1")
	fmt.Println()

	fmt.Println("每一步都是函数变换:")
	fmt.Println("1. 卷积层: f1(x, W, b) = W ⊛ x + b")
	fmt.Println("2. ReLU:   f2(x) = max(0, x)")
	fmt.Println("3. 全连接: f3(x, W, b) = Wx + b")
	fmt.Println("4. Softmax: f4(x) = softmax(x)")
	fmt.Println()

	fmt.Println("最终输出是输入经过所有函数变换后的结果！")
}
