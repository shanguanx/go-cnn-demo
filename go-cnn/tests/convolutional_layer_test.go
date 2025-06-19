package tests

import (
	"math"
	"testing"

	"github.com/user/go-cnn/layers"
	"github.com/user/go-cnn/matrix"
)

// TestConvolutionalLayerCreation 测试卷积层创建
func TestConvolutionalLayerCreation(t *testing.T) {
	layer := layers.NewConvolutionalLayer(3, 16, 5, 1, 2)

	if layer.InChannels != 3 {
		t.Errorf("期望输入通道数 3，实际 %d", layer.InChannels)
	}

	if layer.OutChannels != 16 {
		t.Errorf("期望输出通道数 16，实际 %d", layer.OutChannels)
	}

	if layer.KernelSize != 5 {
		t.Errorf("期望卷积核大小 5，实际 %d", layer.KernelSize)
	}

	if layer.Stride != 1 {
		t.Errorf("期望步长 1，实际 %d", layer.Stride)
	}

	if layer.Padding != 2 {
		t.Errorf("期望填充 2，实际 %d", layer.Padding)
	}

	// 检查权重形状：(out_channels, in_channels * kernel_size * kernel_size)
	expectedWeightRows := 16
	expectedWeightCols := 3 * 5 * 5 // 75
	if layer.Weights.Rows != expectedWeightRows || layer.Weights.Cols != expectedWeightCols {
		t.Errorf("权重形状错误：期望 (%d,%d)，实际 (%d,%d)",
			expectedWeightRows, expectedWeightCols, layer.Weights.Rows, layer.Weights.Cols)
	}

	// 检查偏置形状：(out_channels, 1)
	expectedBiasRows := 16
	expectedBiasCols := 1
	if layer.Biases.Rows != expectedBiasRows || layer.Biases.Cols != expectedBiasCols {
		t.Errorf("偏置形状错误：期望 (%d,%d)，实际 (%d,%d)",
			expectedBiasRows, expectedBiasCols, layer.Biases.Rows, layer.Biases.Cols)
	}
}

// TestConvolutionalLayerForward 测试前向传播
func TestConvolutionalLayerForward(t *testing.T) {
	// 创建简单的卷积层：1输入通道，1输出通道，3x3卷积核，步长1，填充1
	layer := layers.NewConvolutionalLayer(1, 1, 3, 1, 1)

	// 设置输入尺寸
	err := layer.SetInputSize(3, 3)
	if err != nil {
		t.Fatalf("设置输入尺寸失败：%v", err)
	}

	// 手动设置简单的权重和偏置进行测试
	for i := 0; i < len(layer.Weights.Data); i++ {
		layer.Weights.Data[i] = 1.0 // 全1权重
	}
	layer.Biases.Set(0, 0, 0.0) // 零偏置

	// 创建3组样本的输入：(batch_size=3, channels*height*width=1*3*3=9)
	input := matrix.NewMatrix(3, 9)

	// 样本0: 全1输入
	for i := 0; i < 9; i++ {
		input.Set(0, i, 1.0)
	}

	// 样本1: 递增输入 (1,2,3,4,5,6,7,8,9)
	for i := 0; i < 9; i++ {
		input.Set(1, i, float64(i+1))
	}

	// 样本2: 交替输入 (1,0,1,0,1,0,1,0,1)
	for i := 0; i < 9; i++ {
		if i%2 == 0 {
			input.Set(2, i, 1.0)
		} else {
			input.Set(2, i, 0.0)
		}
	}

	t.Log("输入矩阵:")
	t.Log(input.String())

	// 前向传播
	output, err := layer.Forward(input)
	t.Log("输出矩阵:")
	t.Log(output.String())
	if err != nil {
		t.Fatalf("前向传播失败：%v", err)
	}

	// 检查输出形状：(batch_size=3, out_channels*height*width=1*3*3=9)
	expectedRows := 3
	expectedCols := 9
	if output.Rows != expectedRows || output.Cols != expectedCols {
		t.Errorf("输出形状错误：期望 (%d,%d)，实际 (%d,%d)",
			expectedRows, expectedCols, output.Rows, output.Cols)
	}

	// 样本0的预期输出：全1输入，padding=1，3x3卷积核全1权重
	// 角落位置(4个有效输入): 4, 边缘位置(6个有效输入): 6, 中心位置(9个有效输入): 9
	expectedValues0 := []float64{4, 6, 4, 6, 9, 6, 4, 6, 4}

	// 样本1的预期输出：递增输入(1,2,3,4,5,6,7,8,9)
	// 根据Python验证结果更新
	expectedValues1 := []float64{
		12, 21, 16, // 第一行: 12, 21, 16
		27, 45, 33, // 第二行: 27, 45, 33
		24, 39, 28, // 第三行: 24, 39, 28
	}

	// 样本2的预期输出：交替输入(1,0,1,0,1,0,1,0,1)
	// 根据Python验证结果更新
	expectedValues2 := []float64{
		2, 3, 2, // 第一行: 2, 3, 2
		3, 5, 3, // 第二行: 3, 5, 3
		2, 3, 2, // 第三行: 2, 3, 2
	}

	// 验证样本0的输出
	for i := 0; i < 9; i++ {
		expected := expectedValues0[i]
		actual := output.At(0, i)
		if math.Abs(actual-expected) > 1e-6 {
			t.Errorf("样本0输出值错误：位置 %d，期望 %.6f，实际 %.6f", i, expected, actual)
		}
	}

	// 验证样本1的输出
	for i := 0; i < 9; i++ {
		expected := expectedValues1[i]
		actual := output.At(1, i)
		if math.Abs(actual-expected) > 1e-6 {
			t.Errorf("样本1输出值错误：位置 %d，期望 %.6f，实际 %.6f", i, expected, actual)
		}
	}

	// 验证样本2的输出
	for i := 0; i < 9; i++ {
		expected := expectedValues2[i]
		actual := output.At(2, i)
		if math.Abs(actual-expected) > 1e-6 {
			t.Errorf("样本2输出值错误：位置 %d，期望 %.6f，实际 %.6f", i, expected, actual)
		}
	}

	t.Log("所有样本的卷积结果验证通过！")
}

// TestConvolutionalLayerForwardBatch 测试批量前向传播
func TestConvolutionalLayerForwardBatch(t *testing.T) {
	layer := layers.NewConvolutionalLayer(2, 3, 3, 1, 1)

	// 设置固定权重（用于对拍测试）
	layer.SetFixedWeights()

	// 设置输入尺寸
	err := layer.SetInputSize(4, 4)
	if err != nil {
		t.Fatalf("设置输入尺寸失败：%v", err)
	}

	// 创建批量输入：(batch_size=2, channels*height*width=2*4*4=32)
	input := matrix.NewMatrix(2, 32)

	// 第一个样本：使用递增的值 (0, 1, 2, ..., 31)
	for j := 0; j < 32; j++ {
		input.Set(0, j, float64(j))
	}

	// 第二个样本：使用递减的值 (31, 30, 29, ..., 0)
	for j := 0; j < 32; j++ {
		input.Set(1, j, float64(31-j))
	}

	output, err := layer.Forward(input)
	t.Log("输入矩阵:")
	t.Log(input.String())
	t.Log("输出矩阵:")
	t.Log(output.String())

	// 打印权重和偏置信息（用于Python对拍）
	t.Log("权重矩阵:")
	t.Log(layer.Weights.String())
	t.Log("偏置向量:")
	t.Log(layer.Biases.String())

	if err != nil {
		t.Fatalf("批量前向传播失败：%v", err)
	}

	// 检查输出形状：(batch_size=2, out_channels*height*width=3*4*4=48)
	expectedRows := 2
	expectedCols := 48
	if output.Rows != expectedRows || output.Cols != expectedCols {
		t.Errorf("批量输出形状错误：期望 (%d,%d)，实际 (%d,%d)",
			expectedRows, expectedCols, output.Rows, output.Cols)
	}

	// 验证两个样本的输出确实不同
	sample1Output := make([]float64, expectedCols)
	sample2Output := make([]float64, expectedCols)

	for j := 0; j < expectedCols; j++ {
		sample1Output[j] = output.At(0, j)
		sample2Output[j] = output.At(1, j)
	}

	// 检查两个样本的输出是否不同
	allSame := true
	for j := 0; j < expectedCols; j++ {
		if sample1Output[j] != sample2Output[j] {
			allSame = false
			break
		}
	}

	if allSame {
		t.Error("批量处理失败：两个样本的输出完全相同，说明批量处理可能有问题")
	}

	t.Logf("批量处理验证通过：两个样本的输出不同，说明批量处理正常工作")
}

// TestConvolutionalLayerBackward 测试反向传播
func TestConvolutionalLayerBackward(t *testing.T) {
	layer := layers.NewConvolutionalLayer(1, 1, 3, 1, 1)

	// 设置输入尺寸
	err := layer.SetInputSize(3, 3)
	if err != nil {
		t.Fatalf("设置输入尺寸失败：%v", err)
	}

	// 创建输入
	input := matrix.NewMatrix(1, 9)
	for i := 0; i < 9; i++ {
		input.Set(0, i, float64(i+1)) // 填充1-9
	}

	// 前向传播
	output, err := layer.Forward(input)
	if err != nil {
		t.Fatalf("前向传播失败：%v", err)
	}

	// 创建梯度输出（假设损失对输出的梯度都是1）
	gradOutput := matrix.NewMatrix(output.Rows, output.Cols)
	for i := 0; i < gradOutput.Rows; i++ {
		for j := 0; j < gradOutput.Cols; j++ {
			gradOutput.Set(i, j, 1.0)
		}
	}

	// 反向传播
	gradInput, err := layer.Backward(gradOutput)
	if err != nil {
		t.Fatalf("反向传播失败：%v", err)
	}

	// 检查输入梯度形状
	if gradInput.Rows != input.Rows || gradInput.Cols != input.Cols {
		t.Errorf("输入梯度形状错误：期望 (%d,%d)，实际 (%d,%d)",
			input.Rows, input.Cols, gradInput.Rows, gradInput.Cols)
	}

	// 检查权重梯度是否计算
	if layer.WeightGradients == nil {
		t.Error("权重梯度未计算")
	}

	// 检查偏置梯度是否计算
	if layer.BiasGradients == nil {
		t.Error("偏置梯度未计算")
	}

	// 权重梯度形状应该与权重相同
	if layer.WeightGradients.Rows != layer.Weights.Rows ||
		layer.WeightGradients.Cols != layer.Weights.Cols {
		t.Errorf("权重梯度形状错误：期望 (%d,%d)，实际 (%d,%d)",
			layer.Weights.Rows, layer.Weights.Cols,
			layer.WeightGradients.Rows, layer.WeightGradients.Cols)
	}

	// 偏置梯度形状应该与偏置相同
	if layer.BiasGradients.Rows != layer.Biases.Rows ||
		layer.BiasGradients.Cols != layer.Biases.Cols {
		t.Errorf("偏置梯度形状错误：期望 (%d,%d)，实际 (%d,%d)",
			layer.Biases.Rows, layer.Biases.Cols,
			layer.BiasGradients.Rows, layer.BiasGradients.Cols)
	}
}

// TestConvolutionalLayerGradientCheck 梯度检查测试
func TestConvolutionalLayerGradientCheck(t *testing.T) {
	layer := layers.NewConvolutionalLayer(1, 1, 3, 1, 1)

	// 设置输入尺寸
	err := layer.SetInputSize(4, 4)
	if err != nil {
		t.Fatalf("设置输入尺寸失败：%v", err)
	}

	// 创建小的输入进行梯度检查
	input := matrix.NewMatrix(1, 16)
	for i := 0; i < 16; i++ {
		input.Set(0, i, float64(i)*0.1)
	}

	// 前向传播
	output, err := layer.Forward(input)
	if err != nil {
		t.Fatalf("前向传播失败：%v", err)
	}

	// 创建简单的损失：输出的平方和
	loss := 0.0
	for i := 0; i < output.Rows; i++ {
		for j := 0; j < output.Cols; j++ {
			val := output.At(i, j)
			loss += val * val
		}
	}

	// 损失对输出的梯度：2 * output
	gradOutput := matrix.NewMatrix(output.Rows, output.Cols)
	for i := 0; i < gradOutput.Rows; i++ {
		for j := 0; j < gradOutput.Cols; j++ {
			gradOutput.Set(i, j, 2.0*output.At(i, j))
		}
	}

	// 反向传播
	_, err = layer.Backward(gradOutput)
	if err != nil {
		t.Fatalf("反向传播失败：%v", err)
	}

	// 数值梯度检查（只检查第一个权重）
	epsilon := 1e-5
	originalWeight := layer.Weights.At(0, 0)

	// 正向扰动
	layer.Weights.Set(0, 0, originalWeight+epsilon)
	outputPlus, _ := layer.Forward(input)
	lossPlus := 0.0
	for i := 0; i < outputPlus.Rows; i++ {
		for j := 0; j < outputPlus.Cols; j++ {
			val := outputPlus.At(i, j)
			lossPlus += val * val
		}
	}

	// 负向扰动
	layer.Weights.Set(0, 0, originalWeight-epsilon)
	outputMinus, _ := layer.Forward(input)
	lossMinus := 0.0
	for i := 0; i < outputMinus.Rows; i++ {
		for j := 0; j < outputMinus.Cols; j++ {
			val := outputMinus.At(i, j)
			lossMinus += val * val
		}
	}

	// 恢复原始权重
	layer.Weights.Set(0, 0, originalWeight)

	// 计算数值梯度
	numGradient := (lossPlus - lossMinus) / (2 * epsilon)

	// 重新计算解析梯度
	layer.Forward(input)
	layer.Backward(gradOutput)
	analGradient := layer.WeightGradients.At(0, 0)

	// 比较梯度
	relativeError := math.Abs(numGradient-analGradient) /
		(math.Abs(numGradient) + math.Abs(analGradient) + 1e-8)

	if relativeError > 1e-3 {
		t.Errorf("梯度检查失败：数值梯度 %.6f，解析梯度 %.6f，相对误差 %.6f",
			numGradient, analGradient, relativeError)
	}
}

// TestConvolutionalLayerWeightUpdate 测试权重更新
func TestConvolutionalLayerWeightUpdate(t *testing.T) {
	layer := layers.NewConvolutionalLayer(1, 1, 3, 1, 1)

	// 设置输入尺寸
	err := layer.SetInputSize(3, 3)
	if err != nil {
		t.Fatalf("设置输入尺寸失败：%v", err)
	}

	// 保存原始权重
	originalWeights := layer.GetWeights()
	originalBiases := layer.GetBiases()

	// 创建输入并进行前向和反向传播
	input := matrix.NewMatrix(1, 9)
	for i := 0; i < 9; i++ {
		input.Set(0, i, 1.0)
	}

	output, _ := layer.Forward(input)
	gradOutput := matrix.NewMatrix(output.Rows, output.Cols)
	for i := 0; i < gradOutput.Rows; i++ {
		for j := 0; j < gradOutput.Cols; j++ {
			gradOutput.Set(i, j, 1.0)
		}
	}

	layer.Backward(gradOutput)

	// 更新权重
	learningRate := 0.01
	err = layer.UpdateWeights(learningRate)
	if err != nil {
		t.Fatalf("权重更新失败：%v", err)
	}

	// 检查权重是否发生变化
	weightChanged := false
	for i := 0; i < layer.Weights.Rows; i++ {
		for j := 0; j < layer.Weights.Cols; j++ {
			if math.Abs(layer.Weights.At(i, j)-originalWeights.At(i, j)) > 1e-10 {
				weightChanged = true
				break
			}
		}
		if weightChanged {
			break
		}
	}

	if !weightChanged {
		t.Error("权重未发生更新")
	}

	// 检查偏置是否发生变化
	biasChanged := false
	for i := 0; i < layer.Biases.Rows; i++ {
		for j := 0; j < layer.Biases.Cols; j++ {
			if math.Abs(layer.Biases.At(i, j)-originalBiases.At(i, j)) > 1e-10 {
				biasChanged = true
				break
			}
		}
		if biasChanged {
			break
		}
	}

	if !biasChanged {
		t.Error("偏置未发生更新")
	}
}

// TestConvolutionalLayerZeroGradients 测试梯度清零
func TestConvolutionalLayerZeroGradients(t *testing.T) {
	layer := layers.NewConvolutionalLayer(1, 1, 3, 1, 1)

	// 设置输入尺寸
	err := layer.SetInputSize(3, 3)
	if err != nil {
		t.Fatalf("设置输入尺寸失败：%v", err)
	}

	// 进行前向和反向传播以产生梯度
	input := matrix.NewMatrix(1, 9)
	// 设置非零输入
	for i := 0; i < 9; i++ {
		input.Set(0, i, float64(i+1))
	}

	output, _ := layer.Forward(input)
	gradOutput := matrix.NewMatrix(output.Rows, output.Cols)
	// 设置非零梯度输出
	for i := 0; i < gradOutput.Rows; i++ {
		for j := 0; j < gradOutput.Cols; j++ {
			gradOutput.Set(i, j, 1.0)
		}
	}

	layer.Backward(gradOutput)

	// 确认梯度不为零
	hasNonZeroWeightGrad := false
	for i := 0; i < len(layer.WeightGradients.Data); i++ {
		if math.Abs(layer.WeightGradients.Data[i]) > 1e-10 {
			hasNonZeroWeightGrad = true
			break
		}
	}

	if !hasNonZeroWeightGrad {
		t.Error("权重梯度应该不为零")
	}

	// 清零梯度
	layer.ZeroGradients()

	// 检查权重梯度是否清零
	for i := 0; i < len(layer.WeightGradients.Data); i++ {
		if math.Abs(layer.WeightGradients.Data[i]) > 1e-10 {
			t.Errorf("权重梯度未清零：位置 %d，值 %.6f", i, layer.WeightGradients.Data[i])
		}
	}

	// 检查偏置梯度是否清零
	for i := 0; i < len(layer.BiasGradients.Data); i++ {
		if math.Abs(layer.BiasGradients.Data[i]) > 1e-10 {
			t.Errorf("偏置梯度未清零：位置 %d，值 %.6f", i, layer.BiasGradients.Data[i])
		}
	}
}

// TestConvolutionalLayerOutputShape 测试输出形状计算
func TestConvolutionalLayerOutputShape(t *testing.T) {
	testCases := []struct {
		inChannels   int
		outChannels  int
		kernelSize   int
		stride       int
		padding      int
		inputHeight  int
		inputWidth   int
		batchSize    int
		expectedRows int
		expectedCols int
		shouldError  bool
	}{
		// 正常情况
		{1, 16, 3, 1, 1, 28, 28, 2, 2, 16 * 28 * 28, false},
		{3, 32, 5, 1, 2, 32, 32, 4, 4, 32 * 32 * 32, false},
		{1, 8, 3, 2, 1, 10, 10, 1, 1, 8 * 5 * 5, false},
		// 错误情况
		{1, 8, 5, 1, 0, 3, 3, 1, 0, 0, true}, // 输出尺寸为负
	}

	for i, tc := range testCases {
		layer := layers.NewConvolutionalLayer(tc.inChannels, tc.outChannels,
			tc.kernelSize, tc.stride, tc.padding)

		err := layer.SetInputSize(tc.inputHeight, tc.inputWidth)
		if tc.shouldError {
			if err == nil {
				t.Errorf("测试用例 %d：期望错误但没有发生", i)
			}
			continue
		}

		if err != nil {
			t.Errorf("测试用例 %d：设置输入尺寸失败：%v", i, err)
			continue
		}

		rows, cols, err := layer.GetOutputShape(tc.batchSize)
		if err != nil {
			t.Errorf("测试用例 %d：获取输出形状失败：%v", i, err)
			continue
		}

		if rows != tc.expectedRows || cols != tc.expectedCols {
			t.Errorf("测试用例 %d：输出形状错误：期望 (%d,%d)，实际 (%d,%d)",
				i, tc.expectedRows, tc.expectedCols, rows, cols)
		}
	}
}

// TestConvolutionalLayerErrorHandling 测试错误处理
func TestConvolutionalLayerErrorHandling(t *testing.T) {
	layer := layers.NewConvolutionalLayer(2, 4, 3, 1, 1)

	// 测试在没有设置输入尺寸的情况下进行前向传播
	wrongInput := matrix.NewMatrix(1, 32)
	_, err := layer.Forward(wrongInput)
	if err == nil {
		t.Error("期望因未设置输入尺寸而失败")
	}

	// 设置输入尺寸
	layer.SetInputSize(4, 4)

	// 测试错误的输入尺寸
	wrongSizeInput := matrix.NewMatrix(1, 16) // 16而不是32
	_, err = layer.Forward(wrongSizeInput)
	if err == nil {
		t.Error("期望因输入尺寸不匹配而失败")
	}

	// 测试在没有前向传播的情况下进行反向传播
	gradOutput := matrix.NewMatrix(1, 64)
	_, err = layer.Backward(gradOutput)
	if err == nil {
		t.Error("期望因没有前向传播而失败")
	}

	// 测试在没有反向传播的情况下更新权重
	err = layer.UpdateWeights(0.01)
	if err == nil {
		t.Error("期望因没有计算梯度而失败")
	}
}

// 基准测试
func BenchmarkConvolutionalLayerForward(b *testing.B) {
	layer := layers.NewConvolutionalLayer(3, 32, 5, 1, 2)
	layer.SetInputSize(32, 32)
	input := matrix.NewMatrix(8, 3*32*32) // 批量大小8的输入

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		layer.Forward(input)
	}
}

func BenchmarkConvolutionalLayerBackward(b *testing.B) {
	layer := layers.NewConvolutionalLayer(3, 32, 5, 1, 2)
	layer.SetInputSize(32, 32)
	input := matrix.NewMatrix(8, 3*32*32)

	// 进行一次前向传播
	output, _ := layer.Forward(input)
	gradOutput := matrix.NewMatrix(output.Rows, output.Cols)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		layer.Backward(gradOutput)
	}
}
