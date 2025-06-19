package tests

import (
	"math"
	"testing"

	"github.com/user/go-cnn/layers"
	"github.com/user/go-cnn/matrix"
)

// TestMaxPoolingBasic 基础MaxPooling测试
func TestMaxPoolingBasic(t *testing.T) {
	// 创建2x2 MaxPooling层，stride=2
	layer := layers.NewMaxPoolingLayer(2, 2, 2, 2)

	// 设置输入尺寸: 4x4x1
	err := layer.SetInputSize(4, 4, 1)
	if err != nil {
		t.Fatalf("设置输入尺寸失败: %v", err)
	}

	// 创建测试输入 [1, 1*4*4] = [1, 16]
	input := matrix.NewMatrix(1, 16)

	// 设置测试数据 (模拟4x4图像)
	// [[ 1  2  3  4]
	//  [ 5  6  7  8]
	//  [ 9 10 11 12]
	//  [13 14 15 16]]
	for i := 0; i < 16; i++ {
		input.Set(0, i, float64(i+1))
	}

	// 执行前向传播
	output, err := layer.Forward(input)
	if err != nil {
		t.Fatalf("前向传播失败: %v", err)
	}

	// 验证输出形状: [1, 1*2*2] = [1, 4]
	if output.Rows != 1 || output.Cols != 4 {
		t.Errorf("输出形状错误: 期望[1,4], 实际[%d,%d]", output.Rows, output.Cols)
	}

	// 验证输出值
	// 2x2窗口的最大值应该是:
	// 左上: max(1,2,5,6) = 6     右上: max(3,4,7,8) = 8
	// 左下: max(9,10,13,14) = 14  右下: max(11,12,15,16) = 16
	expectedOutput := []float64{6, 8, 14, 16}

	for i := 0; i < 4; i++ {
		actual := output.At(0, i)
		expected := expectedOutput[i]
		if actual != expected {
			t.Errorf("输出值错误[%d]: 期望%f, 实际%f", i, expected, actual)
		}
	}
}

// TestAveragePoolingBasic 基础AveragePooling测试
func TestAveragePoolingBasic(t *testing.T) {
	// 创建2x2 AveragePooling层，stride=2
	layer := layers.NewAveragePoolingLayer(2, 2, 2, 2)

	// 设置输入尺寸: 4x4x1
	err := layer.SetInputSize(4, 4, 1)
	if err != nil {
		t.Fatalf("设置输入尺寸失败: %v", err)
	}

	// 创建测试输入
	input := matrix.NewMatrix(1, 16)
	for i := 0; i < 16; i++ {
		input.Set(0, i, float64(i+1))
	}

	// 执行前向传播
	output, err := layer.Forward(input)
	if err != nil {
		t.Fatalf("前向传播失败: %v", err)
	}

	// 验证输出值
	// 2x2窗口的平均值应该是:
	// 左上: avg(1,2,5,6) = 3.5     右上: avg(3,4,7,8) = 5.5
	// 左下: avg(9,10,13,14) = 11.5  右下: avg(11,12,15,16) = 13.5
	expectedOutput := []float64{3.5, 5.5, 11.5, 13.5}

	for i := 0; i < 4; i++ {
		actual := output.At(0, i)
		expected := expectedOutput[i]
		if math.Abs(actual-expected) > 1e-10 {
			t.Errorf("输出值错误[%d]: 期望%f, 实际%f", i, expected, actual)
		}
	}
}

// TestMaxPoolingBackward 测试MaxPooling反向传播
func TestMaxPoolingBackward(t *testing.T) {
	layer := layers.NewMaxPoolingLayer(2, 2, 2, 2)
	err := layer.SetInputSize(4, 4, 1)
	if err != nil {
		t.Fatalf("设置输入尺寸失败: %v", err)
	}

	// 创建测试输入
	input := matrix.NewMatrix(1, 16)
	for i := 0; i < 16; i++ {
		input.Set(0, i, float64(i+1))
	}

	// 前向传播
	_, err = layer.Forward(input)
	if err != nil {
		t.Fatalf("前向传播失败: %v", err)
	}

	// 创建输出梯度（全1）
	dOutput := matrix.NewMatrix(1, 4)
	for i := 0; i < 4; i++ {
		dOutput.Set(0, i, 1.0)
	}

	// 反向传播
	dInput, err := layer.Backward(dOutput)
	if err != nil {
		t.Fatalf("反向传播失败: %v", err)
	}

	// 验证输入梯度形状
	if dInput.Rows != 1 || dInput.Cols != 16 {
		t.Errorf("输入梯度形状错误: 期望[1,16], 实际[%d,%d]", dInput.Rows, dInput.Cols)
	}

	// 验证梯度只在最大值位置为1，其他位置为0
	// 最大值位置索引: 5(value=6), 7(value=8), 13(value=14), 15(value=16)
	expectedGradient := make([]float64, 16)
	expectedGradient[5] = 1.0  // 6的位置
	expectedGradient[7] = 1.0  // 8的位置
	expectedGradient[13] = 1.0 // 14的位置
	expectedGradient[15] = 1.0 // 16的位置

	for i := 0; i < 16; i++ {
		actual := dInput.At(0, i)
		expected := expectedGradient[i]
		if actual != expected {
			t.Errorf("输入梯度错误[%d]: 期望%f, 实际%f", i, expected, actual)
		}
	}
}

// TestPoolingLayerCreation 测试池化层创建
func TestPoolingLayerCreation(t *testing.T) {
	// 测试MaxPooling层创建
	maxLayer := layers.NewMaxPoolingLayer(3, 3, 1, 1)
	if maxLayer == nil {
		t.Fatal("创建MaxPooling层失败")
	}

	if maxLayer.GetPoolingType() != layers.MaxPooling {
		t.Errorf("池化类型错误: 期望MaxPooling, 实际%v", maxLayer.GetPoolingType())
	}

	// 测试AveragePooling层创建
	avgLayer := layers.NewAveragePoolingLayer(2, 2, 2, 2)
	if avgLayer == nil {
		t.Fatal("创建AveragePooling层失败")
	}

	if avgLayer.GetPoolingType() != layers.AveragePooling {
		t.Errorf("池化类型错误: 期望AveragePooling, 实际%v", avgLayer.GetPoolingType())
	}
}
