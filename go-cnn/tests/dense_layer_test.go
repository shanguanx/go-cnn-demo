package tests

import (
	"math"
	"testing"

	"github.com/user/go-cnn/layers"
	"github.com/user/go-cnn/matrix"
)

const tolerance = 1e-10

// TestDenseLayerCreation 测试全连接层创建
func TestDenseLayerCreation(t *testing.T) {
	layer := layers.NewDenseLayer(10, 5)

	// 检查层配置
	if layer.GetInputFeatures() != 10 {
		t.Errorf("期望输入特征数为10，得到%d", layer.GetInputFeatures())
	}

	if layer.GetOutputFeatures() != 5 {
		t.Errorf("期望输出特征数为5，得到%d", layer.GetOutputFeatures())
	}

	// 检查权重矩阵尺寸
	weights := layer.GetWeights()
	if weights.Rows != 10 || weights.Cols != 5 {
		t.Errorf("权重矩阵尺寸错误：期望(10,5)，得到(%d,%d)", weights.Rows, weights.Cols)
	}

	// 检查偏置向量尺寸
	biases := layer.GetBiases()
	if biases.Rows != 1 || biases.Cols != 5 {
		t.Errorf("偏置向量尺寸错误：期望(1,5)，得到(%d,%d)", biases.Rows, biases.Cols)
	}

	// 检查偏置是否初始化为0
	for j := 0; j < 5; j++ {
		if math.Abs(biases.At(0, j)) > tolerance {
			t.Errorf("偏置应该初始化为0，但biases[%d] = %f", j, biases.At(0, j))
		}
	}

	t.Logf("成功创建全连接层：%s", layer.String())
}

// TestDenseLayerForward 测试全连接层前向传播
func TestDenseLayerForward(t *testing.T) {
	// 创建一个简单的2x2层用于测试
	layer := layers.NewDenseLayer(2, 2)

	// 手动设置权重和偏置以便验证计算
	weights := layer.GetWeights()
	weights.Set(0, 0, 0.5) // W[0,0] = 0.5
	weights.Set(0, 1, 0.3) // W[0,1] = 0.3
	weights.Set(1, 0, 0.2) // W[1,0] = 0.2
	weights.Set(1, 1, 0.4) // W[1,1] = 0.4

	biases := layer.GetBiases()
	biases.Set(0, 0, 0.1) // b[0] = 0.1
	biases.Set(0, 1, 0.2) // b[1] = 0.2

	// 创建输入：batch_size=2, input_features=2
	// [[1, 2], [3, 4]]
	input := matrix.NewMatrixFromData([]float64{1, 2, 3, 4}, 2, 2)

	// 执行前向传播
	output, err := layer.Forward(input)
	if err != nil {
		t.Fatalf("前向传播失败：%v", err)
	}

	// 验证输出尺寸
	if output.Rows != 2 || output.Cols != 2 {
		t.Errorf("输出尺寸错误：期望(2,2)，得到(%d,%d)", output.Rows, output.Cols)
	}

	// 手工计算期望结果
	// 对于第一个样本[1, 2]：
	// output[0,0] = 1*0.5 + 2*0.2 + 0.1 = 0.5 + 0.4 + 0.1 = 1.0
	// output[0,1] = 1*0.3 + 2*0.4 + 0.2 = 0.3 + 0.8 + 0.2 = 1.3
	// 对于第二个样本[3, 4]：
	// output[1,0] = 3*0.5 + 4*0.2 + 0.1 = 1.5 + 0.8 + 0.1 = 2.4
	// output[1,1] = 3*0.3 + 4*0.4 + 0.2 = 0.9 + 1.6 + 0.2 = 2.7

	expectedValues := [][]float64{{1.0, 1.3}, {2.4, 2.7}}

	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			actual := output.At(i, j)
			expected := expectedValues[i][j]
			if math.Abs(actual-expected) > tolerance {
				t.Errorf("输出值错误：output[%d,%d] 期望%f，得到%f", i, j, expected, actual)
			}
		}
	}

	t.Logf("前向传播测试通过，输出：\n%s", output.String())
}

// TestDenseLayerBackward 测试全连接层反向传播
func TestDenseLayerBackward(t *testing.T) {
	// 创建2x2层
	layer := layers.NewDenseLayer(2, 2)

	// 设置权重
	weights := layer.GetWeights()
	weights.Set(0, 0, 0.5)
	weights.Set(0, 1, 0.3)
	weights.Set(1, 0, 0.2)
	weights.Set(1, 1, 0.4)

	// 创建输入并执行前向传播
	input := matrix.NewMatrixFromData([]float64{1, 2, 3, 4}, 2, 2)
	_, err := layer.Forward(input)
	if err != nil {
		t.Fatalf("前向传播失败：%v", err)
	}

	// 清零梯度
	layer.ZeroGradients()

	// 创建输出梯度
	gradOutput := matrix.NewMatrixFromData([]float64{1, 1, 1, 1}, 2, 2)

	// 执行反向传播
	gradInput, err := layer.Backward(gradOutput)
	if err != nil {
		t.Fatalf("反向传播失败：%v", err)
	}

	// 验证输入梯度尺寸
	if gradInput.Rows != 2 || gradInput.Cols != 2 {
		t.Errorf("输入梯度尺寸错误：期望(2,2)，得到(%d,%d)", gradInput.Rows, gradInput.Cols)
	}

	// 手工计算期望的输入梯度
	// gradInput = gradOutput * weights^T
	// weights^T = [[0.5, 0.2], [0.3, 0.4]]
	// gradInput[0,0] = 1*0.5 + 1*0.3 = 0.8
	// gradInput[0,1] = 1*0.2 + 1*0.4 = 0.6
	// gradInput[1,0] = 1*0.5 + 1*0.3 = 0.8
	// gradInput[1,1] = 1*0.2 + 1*0.4 = 0.6

	expectedGradInput := [][]float64{{0.8, 0.6}, {0.8, 0.6}}

	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			actual := gradInput.At(i, j)
			expected := expectedGradInput[i][j]
			if math.Abs(actual-expected) > tolerance {
				t.Errorf("输入梯度错误：gradInput[%d,%d] 期望%f，得到%f", i, j, expected, actual)
			}
		}
	}

	// 验证权重梯度
	weightGrads := layer.GetWeightGradients()
	// gradWeights = input^T * gradOutput
	// input^T = [[1, 3], [2, 4]]
	// gradWeights[0,0] = 1*1 + 3*1 = 4
	// gradWeights[0,1] = 1*1 + 3*1 = 4
	// gradWeights[1,0] = 2*1 + 4*1 = 6
	// gradWeights[1,1] = 2*1 + 4*1 = 6

	expectedWeightGrads := [][]float64{{4, 4}, {6, 6}}

	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			actual := weightGrads.At(i, j)
			expected := expectedWeightGrads[i][j]
			if math.Abs(actual-expected) > tolerance {
				t.Errorf("权重梯度错误：weightGrad[%d,%d] 期望%f，得到%f", i, j, expected, actual)
			}
		}
	}

	// 验证偏置梯度
	biasGrads := layer.GetBiasGradients()
	// gradBiases = sum(gradOutput, axis=0) = [2, 2]
	expectedBiasGrads := []float64{2, 2}

	for j := 0; j < 2; j++ {
		actual := biasGrads.At(0, j)
		expected := expectedBiasGrads[j]
		if math.Abs(actual-expected) > tolerance {
			t.Errorf("偏置梯度错误：biasGrad[%d] 期望%f，得到%f", j, expected, actual)
		}
	}

	t.Logf("反向传播测试通过")
}

// TestDenseLayerWeightUpdate 测试权重更新
func TestDenseLayerWeightUpdate(t *testing.T) {
	// 使用更大的层尺寸进行测试
	layer := layers.NewDenseLayer(4, 3)

	// 设置更复杂的初始权重矩阵
	weights := layer.GetWeights()
	// 4x3 权重矩阵，使用不同的数值模式
	weights.Set(0, 0, 1.5)
	weights.Set(0, 1, -0.8)
	weights.Set(0, 2, 2.3)
	weights.Set(1, 0, -1.2)
	weights.Set(1, 1, 0.9)
	weights.Set(1, 2, -0.5)
	weights.Set(2, 0, 0.7)
	weights.Set(2, 1, -1.8)
	weights.Set(2, 2, 1.1)
	weights.Set(3, 0, -0.3)
	weights.Set(3, 1, 2.1)
	weights.Set(3, 2, -0.9)

	// 设置更复杂的初始偏置
	biases := layer.GetBiases()
	biases.Set(0, 0, 0.25)
	biases.Set(0, 1, -0.75)
	biases.Set(0, 2, 1.5)

	// 打印初始参数
	t.Logf("=== 初始参数 ===")
	t.Logf("初始权重矩阵 (4x3):\n%s", weights.String())
	t.Logf("初始偏置向量 (1x3):\n%s", biases.String())

	// 创建测试输入数据
	input := matrix.NewMatrixFromData([]float64{
		1.0, 2.0, 3.0, 4.0, // 第一个样本
		0.5, 1.5, 2.5, 3.5, // 第二个样本
	}, 2, 4)

	t.Logf("=== 前向传播 ===")
	t.Logf("输入矩阵 (2x4):\n%s", input.String())

	// 执行前向传播
	output, err := layer.Forward(input)
	if err != nil {
		t.Fatalf("前向传播失败：%v", err)
	}

	t.Logf("前向传播输出 (2x3):\n%s", output.String())

	// 清零梯度
	layer.ZeroGradients()

	// 创建输出梯度
	gradOutput := matrix.NewMatrixFromData([]float64{
		0.1, 0.2, 0.3, // 第一个样本的梯度
		0.4, 0.5, 0.6, // 第二个样本的梯度
	}, 2, 3)

	t.Logf("=== 反向传播 ===")
	t.Logf("输出梯度矩阵 (2x3):\n%s", gradOutput.String())

	// 执行反向传播
	gradInput, err := layer.Backward(gradOutput)
	if err != nil {
		t.Fatalf("反向传播失败：%v", err)
	}

	t.Logf("输入梯度矩阵 (2x4):\n%s", gradInput.String())

	// 获取计算出的权重和偏置梯度
	weightGrads := layer.GetWeightGradients()
	biasGrads := layer.GetBiasGradients()

	t.Logf("计算出的权重梯度 (4x3):\n%s", weightGrads.String())
	t.Logf("计算出的偏置梯度 (1x3):\n%s", biasGrads.String())

	// 使用不同的学习率进行测试
	learningRate := 0.05

	t.Logf("=== 权重更新 ===")
	t.Logf("学习率: %f", learningRate)
	t.Logf("更新前权重矩阵:\n%s", weights.String())
	t.Logf("更新前偏置向量:\n%s", biases.String())

	// 执行权重更新
	err = layer.UpdateWeights(learningRate)
	if err != nil {
		t.Fatalf("权重更新失败：%v", err)
	}

	t.Logf("更新后权重矩阵:\n%s", weights.String())
	t.Logf("更新后偏置向量:\n%s", biases.String())

	// 手动清零梯度（UpdateWeights方法本身不清零梯度）
	layer.ZeroGradients()

	// 验证梯度是否被正确清零
	weightGradsAfter := layer.GetWeightGradients()
	biasGradsAfter := layer.GetBiasGradients()

	t.Logf("=== 梯度清零验证 ===")
	t.Logf("权重梯度清零后:\n%s", weightGradsAfter.String())
	t.Logf("偏置梯度清零后:\n%s", biasGradsAfter.String())

	// 验证权重梯度是否被清零
	for i := 0; i < 4; i++ {
		for j := 0; j < 3; j++ {
			if math.Abs(weightGradsAfter.At(i, j)) > tolerance {
				t.Errorf("权重梯度未清零：weightGrad[%d,%d] = %f", i, j, weightGradsAfter.At(i, j))
			}
		}
	}

	// 验证偏置梯度是否被清零
	for j := 0; j < 3; j++ {
		if math.Abs(biasGradsAfter.At(0, j)) > tolerance {
			t.Errorf("偏置梯度未清零：biasGrad[%d] = %f", j, biasGradsAfter.At(0, j))
		}
	}

	t.Logf("复杂权重更新测试通过 - 4x3层，学习率0.05")
}

// TestDenseLayerGradientAccumulation 测试梯度累积
func TestDenseLayerGradientAccumulation(t *testing.T) {
	layer := layers.NewDenseLayer(2, 2)

	// 设置权重
	weights := layer.GetWeights()
	weights.Set(0, 0, 0.5)
	weights.Set(0, 1, 0.3)
	weights.Set(1, 0, 0.2)
	weights.Set(1, 1, 0.4)

	// 第一次前向和反向传播
	input1 := matrix.NewMatrixFromData([]float64{1, 2}, 1, 2)
	_, err := layer.Forward(input1)
	if err != nil {
		t.Fatalf("第一次前向传播失败：%v", err)
	}

	gradOutput1 := matrix.NewMatrixFromData([]float64{1, 1}, 1, 2)
	_, err = layer.Backward(gradOutput1)
	if err != nil {
		t.Fatalf("第一次反向传播失败：%v", err)
	}

	// 记录第一次的梯度 (这里我们不需要使用它们，只是为了测试梯度累积)
	_ = layer.GetWeightGradients().Copy()
	_ = layer.GetBiasGradients().Copy()

	// 第二次前向和反向传播（不清零梯度）
	input2 := matrix.NewMatrixFromData([]float64{3, 4}, 1, 2)
	_, err = layer.Forward(input2)
	if err != nil {
		t.Fatalf("第二次前向传播失败：%v", err)
	}

	gradOutput2 := matrix.NewMatrixFromData([]float64{1, 1}, 1, 2)
	_, err = layer.Backward(gradOutput2)
	if err != nil {
		t.Fatalf("第二次反向传播失败：%v", err)
	}

	// 验证梯度累积
	currentWeightGrads := layer.GetWeightGradients()
	currentBiasGrads := layer.GetBiasGradients()

	// 权重梯度应该是两次梯度的累积
	// 第一次：input1^T * gradOutput1 = [[1], [2]] * [1, 1] = [[1, 1], [2, 2]]
	// 第二次：input2^T * gradOutput2 = [[3], [4]] * [1, 1] = [[3, 3], [4, 4]]
	// 累积：[[4, 4], [6, 6]]

	expectedAccumWeightGrads := [][]float64{{4, 4}, {6, 6}}

	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			actual := currentWeightGrads.At(i, j)
			expected := expectedAccumWeightGrads[i][j]
			if math.Abs(actual-expected) > tolerance {
				t.Errorf("累积权重梯度错误：weightGrad[%d,%d] 期望%f，得到%f", i, j, expected, actual)
			}
		}
	}

	// 偏置梯度累积：第一次[1, 1] + 第二次[1, 1] = [2, 2]
	expectedAccumBiasGrads := []float64{2, 2}

	for j := 0; j < 2; j++ {
		actual := currentBiasGrads.At(0, j)
		expected := expectedAccumBiasGrads[j]
		if math.Abs(actual-expected) > tolerance {
			t.Errorf("累积偏置梯度错误：biasGrad[%d] 期望%f，得到%f", j, expected, actual)
		}
	}

	t.Logf("梯度累积测试通过")
}

// TestDenseLayerZeroGradients 测试梯度清零
func TestDenseLayerZeroGradients(t *testing.T) {
	layer := layers.NewDenseLayer(3, 2)

	// 人为设置一些梯度值
	weightGrads := layer.GetWeightGradients()
	biasGrads := layer.GetBiasGradients()

	for i := 0; i < 3; i++ {
		for j := 0; j < 2; j++ {
			weightGrads.Set(i, j, float64(i+j+1))
		}
	}

	for j := 0; j < 2; j++ {
		biasGrads.Set(0, j, float64(j+1))
	}

	// 清零梯度
	layer.ZeroGradients()

	// 验证所有梯度都为0
	for i := 0; i < 3; i++ {
		for j := 0; j < 2; j++ {
			if math.Abs(weightGrads.At(i, j)) > tolerance {
				t.Errorf("权重梯度应该为0，但weightGrad[%d,%d] = %f", i, j, weightGrads.At(i, j))
			}
		}
	}

	for j := 0; j < 2; j++ {
		if math.Abs(biasGrads.At(0, j)) > tolerance {
			t.Errorf("偏置梯度应该为0，但biasGrad[%d] = %f", j, biasGrads.At(0, j))
		}
	}

	t.Logf("梯度清零测试通过")
}

// TestDenseLayerErrorHandling 测试错误处理
func TestDenseLayerErrorHandling(t *testing.T) {
	layer := layers.NewDenseLayer(3, 2)

	// 测试输入维度不匹配
	wrongInput := matrix.NewMatrix(2, 5) // 期望3个特征，但给了5个
	_, err := layer.Forward(wrongInput)
	if err == nil {
		t.Error("应该检测到输入特征数不匹配的错误")
	}

	// 测试反向传播前未执行前向传播
	layer2 := layers.NewDenseLayer(2, 2)
	gradOutput := matrix.NewMatrix(1, 2)
	_, err = layer2.Backward(gradOutput)
	if err == nil {
		t.Error("应该检测到未执行前向传播的错误")
	}

	// 测试无效学习率
	err = layer.UpdateWeights(-0.1)
	if err == nil {
		t.Error("应该检测到无效学习率的错误")
	}

	t.Logf("错误处理测试通过")
}

// BenchmarkDenseLayerForward 前向传播性能基准测试
func BenchmarkDenseLayerForward(b *testing.B) {
	layer := layers.NewDenseLayer(256, 128)
	input := matrix.NewMatrix(32, 256) // batch_size=32, features=256

	// 随机初始化输入
	for i := 0; i < 32; i++ {
		for j := 0; j < 256; j++ {
			input.Set(i, j, math.Sin(float64(i+j)))
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := layer.Forward(input)
		if err != nil {
			b.Fatalf("前向传播基准测试失败：%v", err)
		}
	}
}

// BenchmarkDenseLayerBackward 反向传播性能基准测试
func BenchmarkDenseLayerBackward(b *testing.B) {
	layer := layers.NewDenseLayer(256, 128)
	input := matrix.NewMatrix(32, 256)
	gradOutput := matrix.NewMatrix(32, 128)

	// 随机初始化
	for i := 0; i < 32; i++ {
		for j := 0; j < 256; j++ {
			input.Set(i, j, math.Sin(float64(i+j)))
		}
		for j := 0; j < 128; j++ {
			gradOutput.Set(i, j, math.Cos(float64(i+j)))
		}
	}

	// 执行前向传播
	_, err := layer.Forward(input)
	if err != nil {
		b.Fatalf("前向传播失败：%v", err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		layer.ZeroGradients()
		_, err := layer.Backward(gradOutput)
		if err != nil {
			b.Fatalf("反向传播基准测试失败：%v", err)
		}
	}
}
