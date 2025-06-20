package tests

import (
	"testing"

	"github.com/user/go-cnn/graph"
	"github.com/user/go-cnn/matrix"
)

// TestSimpleCNN 测试一个非常简单的CNN网络
func TestSimpleCNN(t *testing.T) {
	t.Log("=== 简单CNN测试 ===")

	// 创建一个非常小的输入: 4x4 图像，1个通道
	// 输入形状: (1, 16) - batch_size=1, 1*4*4=16
	inputData := []float64{
		0.1, 0.2, 0.3, 0.4,
		0.5, 0.6, 0.7, 0.8,
		0.9, 1.0, 0.1, 0.2,
		0.3, 0.4, 0.5, 0.6,
	}
	inputMatrix := matrix.NewMatrixFromData(inputData, 1, 16)
	input := graph.Input(inputMatrix, "input")

	t.Log("输入形状:", input.Value.Rows, "x", input.Value.Cols)
	t.Log("输入数据:")
	t.Log(input.Value.String())

	// 1. 卷积层: 1输入通道 -> 2输出通道, 3x3卷积核, stride=1, padding=0
	// 输入: 4x4, 卷积核: 3x3, 输出: 2x2
	// 输出形状: (1, 2*2*2) = (1, 8)
	conv1 := graph.Conv2dWithFixedWeights(input, 2, 3, 1, 0, 4, 4, 1)

	// 验证固定权重设置
	if conv1Op, ok := conv1.Op.(*graph.ConvolutionalLayerOp); ok {
		t.Log("卷积层1使用固定权重初始化")
		// 打印权重以验证
		weights := conv1Op.ConvolutionalLayer.GetWeights()
		t.Log("卷积层1权重:")
		t.Log(weights.String())
	}
	t.Log("卷积层1输出形状:", conv1.Value.Rows, "x", conv1.Value.Cols)
	t.Log("卷积层1输出:")
	t.Log(conv1.Value.String())

	// 2. ReLU激活
	relu1 := graph.ReLU(conv1)
	t.Log("ReLU1输出形状:", relu1.Value.Rows, "x", relu1.Value.Cols)
	t.Log("ReLU1输出:")
	t.Log(relu1.Value.String())

	// 3. 全连接层: 8 -> 2 (分类为2类)
	dense1 := graph.DenseWithFixedWeights(relu1, 2)

	// 验证固定权重设置
	if dense1Op, ok := dense1.Op.(*graph.DenseLayerOp); ok {
		t.Log("全连接层使用固定权重初始化")
		// 打印权重以验证
		weights := dense1Op.DenseLayer.GetWeights()
		t.Log("全连接层权重:")
		t.Log(weights.String())
	}
	t.Log("全连接层输出形状:", dense1.Value.Rows, "x", dense1.Value.Cols)
	t.Log("全连接层输出:")
	t.Log(dense1.Value.String())

	// 4. 计算损失 (假设标签是类别1)
	labels := graph.NewConstant(matrix.NewMatrixFromData([]float64{1}, 1, 1), "label")
	loss := graph.SoftmaxCrossEntropyLoss(dense1, labels, true)

	t.Log("\n=== 损失计算 ===")
	t.Logf("损失值: %f", loss.Value.At(0, 0))
	t.Logf("损失节点RequiresGrad: %v", loss.RequiresGrad)
	t.Logf("损失节点IsLeaf: %v", loss.IsLeaf)
	t.Logf("损失节点初始梯度: %v", loss.Gradient)

	// 5. 反向传播
	t.Log("\n=== 开始反向传播 ===")
	loss.Backward()
	t.Log("反向传播完成")
	t.Logf("损失节点反向传播后梯度: %v", loss.Gradient)

	// 6. 检查梯度
	t.Log("\n=== 梯度检查 ===")

	// 检查全连接层的梯度
	if dense1.Gradient != nil {
		t.Log("全连接层Node梯度:")
		t.Log(dense1.Gradient.String())
	} else {
		t.Log("全连接层Node梯度: nil")
	}

	// 检查全连接层权重梯度
	if dense1Op, ok := dense1.Op.(*graph.DenseLayerOp); ok {
		weightGrads := dense1Op.GetWeightGradients()
		biasGrads := dense1Op.GetBiasGradients()
		if weightGrads != nil {
			t.Log("全连接层权重梯度:")
			t.Log(weightGrads.String())
		} else {
			t.Log("全连接层权重梯度: nil")
		}
		if biasGrads != nil {
			t.Log("全连接层偏置梯度:")
			t.Log(biasGrads.String())
		} else {
			t.Log("全连接层偏置梯度: nil")
		}
	} else {
		t.Log("全连接层操作类型不匹配")
	}

	// 检查ReLU的梯度
	if relu1.Gradient != nil {
		t.Log("ReLU1 Node梯度:")
		t.Log(relu1.Gradient.String())
	} else {
		t.Log("ReLU1 Node梯度: nil")
	}

	// 检查卷积层的梯度
	if conv1.Gradient != nil {
		t.Log("卷积层1 Node梯度:")
		t.Log(conv1.Gradient.String())
	} else {
		t.Log("卷积层1 Node梯度: nil")
	}

	// 检查卷积层权重梯度
	if conv1Op, ok := conv1.Op.(*graph.ConvolutionalLayerOp); ok {
		weightGrads := conv1Op.GetWeightGradients()
		biasGrads := conv1Op.GetBiasGradients()
		if weightGrads != nil {
			t.Log("卷积层1权重梯度:")
			t.Log(weightGrads.String())
		} else {
			t.Log("卷积层1权重梯度: nil")
		}
		if biasGrads != nil {
			t.Log("卷积层1偏置梯度:")
			t.Log(biasGrads.String())
		} else {
			t.Log("卷积层1偏置梯度: nil")
		}
	} else {
		t.Log("卷积层操作类型不匹配")
	}

	// 检查输入的梯度
	if input.Gradient != nil {
		t.Log("输入Node梯度:")
		t.Log(input.Gradient.String())
	} else {
		t.Log("输入Node梯度: nil")
	}

	t.Log("\n=== 简单CNN测试完成 ===")
}
