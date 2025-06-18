package tests

import (
	"testing"
	"github.com/user/go-cnn/matrix"
)

// TestIm2ColBasic 测试基本的im2col操作
func TestIm2ColBasic(t *testing.T) {
	// 创建一个3x3的输入图像
	input := matrix.NewMatrixFromData([]float64{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	}, 3, 3)
	
	// 使用2x2卷积核，stride=1，padding=0
	kernelH, kernelW := 2, 2
	strideH, strideW := 1, 1
	padH, padW := 0, 0
	
	result := matrix.Im2Col(input, kernelH, kernelW, strideH, strideW, padH, padW)
	
	// 验证输出维度
	expectedRows := kernelH * kernelW // 4
	expectedCols := 2 * 2 // 2x2输出特征图
	
	if result.Rows != expectedRows || result.Cols != expectedCols {
		t.Errorf("Expected dimensions %dx%d, got %dx%d", expectedRows, expectedCols, result.Rows, result.Cols)
	}
	
	// 验证第一列（左上角窗口：1,2,4,5）
	expected := []float64{1, 2, 4, 5}
	for i := 0; i < 4; i++ {
		if result.At(i, 0) != expected[i] {
			t.Errorf("Column 0, row %d: expected %f, got %f", i, expected[i], result.At(i, 0))
		}
	}
	
	// 验证第二列（右上角窗口：2,3,5,6）
	expected = []float64{2, 3, 5, 6}
	for i := 0; i < 4; i++ {
		if result.At(i, 1) != expected[i] {
			t.Errorf("Column 1, row %d: expected %f, got %f", i, expected[i], result.At(i, 1))
		}
	}
}

// TestIm2ColWithPadding 测试带填充的im2col操作
func TestIm2ColWithPadding(t *testing.T) {
	// 创建一个2x2的输入图像
	input := matrix.NewMatrixFromData([]float64{
		1, 2,
		3, 4,
	}, 2, 2)
	
	// 使用3x3卷积核，stride=1，padding=1
	kernelH, kernelW := 3, 3
	strideH, strideW := 1, 1
	padH, padW := 1, 1
	
	result := matrix.Im2Col(input, kernelH, kernelW, strideH, strideW, padH, padW)
	
	// 验证输出维度
	expectedRows := kernelH * kernelW // 9
	expectedCols := 2 * 2 // 2x2输出特征图
	
	if result.Rows != expectedRows || result.Cols != expectedCols {
		t.Errorf("Expected dimensions %dx%d, got %dx%d", expectedRows, expectedCols, result.Rows, result.Cols)
	}
	
	// 验证第一列（左上角窗口，包含填充的零）
	expected := []float64{0, 0, 0, 0, 1, 2, 0, 3, 4}
	for i := 0; i < 9; i++ {
		if result.At(i, 0) != expected[i] {
			t.Errorf("Column 0, row %d: expected %f, got %f", i, expected[i], result.At(i, 0))
		}
	}
}

// TestCol2Im 测试col2im操作
func TestCol2Im(t *testing.T) {
	// 创建一个列矩阵（模拟im2col的输出）
	colMatrix := matrix.NewMatrixFromData([]float64{
		1, 2, 4, 5,
		2, 3, 5, 6,
		4, 5, 7, 8,
		5, 6, 8, 9,
	}, 4, 4)
	
	// 参数
	inputH, inputW := 3, 3
	kernelH, kernelW := 2, 2
	strideH, strideW := 1, 1
	padH, padW := 0, 0
	
	result := matrix.Col2Im(colMatrix, inputH, inputW, kernelH, kernelW, strideH, strideW, padH, padW)
	
	// 验证输出维度
	if result.Rows != inputH || result.Cols != inputW {
		t.Errorf("Expected dimensions %dx%d, got %dx%d", inputH, inputW, result.Rows, result.Cols)
	}
	
	// 由于重叠区域会累加，验证一些已知值
	// 中心位置(1,1)应该被访问4次，值为5
	expectedCenter := 5.0 * 4 // 5被累加4次
	if result.At(1, 1) != expectedCenter {
		t.Errorf("Center value: expected %f, got %f", expectedCenter, result.At(1, 1))
	}
}

// TestIm2ColCol2ImRoundTrip 测试im2col和col2im的往返一致性
func TestIm2ColCol2ImRoundTrip(t *testing.T) {
	// 创建简单的输入图像
	input := matrix.NewMatrixFromData([]float64{
		1, 0, 0,
		0, 0, 0,
		0, 0, 0,
	}, 3, 3)
	
	// 参数
	kernelH, kernelW := 2, 2
	strideH, strideW := 2, 2 // 使用stride=2避免重叠
	padH, padW := 0, 0
	
	// Im2Col
	colMatrix := matrix.Im2Col(input, kernelH, kernelW, strideH, strideW, padH, padW)
	
	// Col2Im
	reconstructed := matrix.Col2Im(colMatrix, input.Rows, input.Cols, kernelH, kernelW, strideH, strideW, padH, padW)
	
	// 验证重构是否正确（由于stride=2，没有重叠，应该完全一致）
	tolerance := 1e-10
	if !input.Equals(reconstructed, tolerance) {
		t.Errorf("Reconstructed matrix doesn't match original")
		t.Logf("Original:\n%s", input.String())
		t.Logf("Reconstructed:\n%s", reconstructed.String())
	}
}

// TestIm2ColWithChannels 测试多通道im2col操作
func TestIm2ColWithChannels(t *testing.T) {
	// 创建2通道的2x2图像
	input := matrix.NewMatrixFromData([]float64{
		1, 2, // 通道0
		3, 4,
		5, 6, // 通道1
		7, 8,
	}, 4, 2)
	
	channels := 2
	kernelH, kernelW := 2, 2
	strideH, strideW := 1, 1
	padH, padW := 0, 0
	
	result := matrix.Im2ColWithChannels(input, channels, kernelH, kernelW, strideH, strideW, padH, padW)
	
	// 验证输出维度
	expectedRows := channels * kernelH * kernelW // 2*2*2=8
	expectedCols := 1 * 1 // 1x1输出特征图
	
	if result.Rows != expectedRows || result.Cols != expectedCols {
		t.Errorf("Expected dimensions %dx%d, got %dx%d", expectedRows, expectedCols, result.Rows, result.Cols)
	}
	
	// 验证输出值（应该是按通道展开的窗口）
	expected := []float64{1, 2, 3, 4, 5, 6, 7, 8}
	for i := 0; i < expectedRows; i++ {
		if result.At(i, 0) != expected[i] {
			t.Errorf("Row %d: expected %f, got %f", i, expected[i], result.At(i, 0))
		}
	}
}

// BenchmarkIm2Col 性能测试
func BenchmarkIm2Col(b *testing.B) {
	// 创建较大的输入图像（模拟MNIST 28x28）
	input := matrix.Random(28, 28, 0, 1)
	
	kernelH, kernelW := 5, 5
	strideH, strideW := 1, 1
	padH, padW := 2, 2
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		matrix.Im2Col(input, kernelH, kernelW, strideH, strideW, padH, padW)
	}
}

// BenchmarkCol2Im 性能测试
func BenchmarkCol2Im(b *testing.B) {
	// 创建列矩阵
	colMatrix := matrix.Random(25, 784, 0, 1) // 5x5卷积核在28x28图像上
	
	inputH, inputW := 28, 28
	kernelH, kernelW := 5, 5
	strideH, strideW := 1, 1
	padH, padW := 2, 2
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		matrix.Col2Im(colMatrix, inputH, inputW, kernelH, kernelW, strideH, strideW, padH, padW)
	}
}