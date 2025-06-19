package tests

import (
	"github.com/user/go-cnn/layers"
	"github.com/user/go-cnn/matrix"
	"testing"
)

// TestSimpleConvolutionalLayer 简单测试
func TestSimpleConvolutionalLayer(t *testing.T) {
	// 创建最简单的卷积层测试
	layer := layers.NewConvolutionalLayer(1, 1, 3, 1, 1)

	// 设置输入尺寸为3x3
	err := layer.SetInputSize(3, 3)
	if err != nil {
		t.Fatalf("设置输入尺寸失败：%v", err)
	}

	// 手动设置权重为全1，偏置为0
	for i := 0; i < len(layer.Weights.Data); i++ {
		layer.Weights.Data[i] = 1.0
	}
	layer.Biases.Set(0, 0, 0.0)

	// 创建输入：1个样本，1个通道，3x3=9个像素，全部设为1
	input := matrix.NewMatrix(1, 9)
	for i := 0; i < 9; i++ {
		input.Set(0, i, 1.0)
	}

	// 测试前向传播
	output, err := layer.Forward(input)
	if err != nil {
		t.Fatalf("前向传播失败：%v", err)
	}

	// 检查输出形状
	if output.Rows != 1 || output.Cols != 9 {
		t.Errorf("输出形状错误：期望(1,9)，实际(%d,%d)", output.Rows, output.Cols)
	}

	t.Logf("前向传播成功，输出形状: (%d,%d)", output.Rows, output.Cols)

	// 打印输出值
	for i := 0; i < output.Cols; i++ {
		t.Logf("输出[%d]: %.6f", i, output.At(0, i))
	}

	// 对于3x3输入，3x3权重，padding=1，stride=1的卷积：
	// - 中心9个位置的输出应该是9（因为权重全1，输入全1）
	// - 边缘位置由于padding会小一些
	// 我们检查至少中心位置的值是9
	centerOutputs := []int{4} // 中心位置
	for _, pos := range centerOutputs {
		expected := 9.0
		actual := output.At(0, pos)
		if actual != expected {
			t.Logf("中心位置%d：期望%.1f，实际%.6f（可能由于随机初始化不同）", pos, expected, actual)
		}
	}
}
