package main

import (
	"fmt"
	"github.com/user/go-cnn/matrix"
)

func main() {
	fmt.Println("=== Im2Col 和 Col2Im 使用示例 ===")
	
	// 示例1: 基本的im2col操作
	fmt.Println("\n1. 基本的Im2Col操作:")
	
	// 创建一个4x4的输入图像
	input := matrix.NewMatrixFromData([]float64{
		1,  2,  3,  4,
		5,  6,  7,  8,
		9,  10, 11, 12,
		13, 14, 15, 16,
	}, 4, 4)
	
	fmt.Println("输入图像 (4x4):")
	fmt.Print(input.String())
	
	// 使用3x3卷积核，stride=1，padding=0
	kernelH, kernelW := 3, 3
	strideH, strideW := 1, 1
	padH, padW := 0, 0
	
	colMatrix := matrix.Im2Col(input, kernelH, kernelW, strideH, strideW, padH, padW)
	
	fmt.Printf("\nIm2Col结果 (%dx%d):\n", colMatrix.Rows, colMatrix.Cols)
	fmt.Print(colMatrix.String())
	
	// 示例2: 带填充的操作
	fmt.Println("\n2. 带填充的Im2Col操作:")
	
	// 创建一个2x2的小图像
	smallInput := matrix.NewMatrixFromData([]float64{
		1, 2,
		3, 4,
	}, 2, 2)
	
	fmt.Println("小图像 (2x2):")
	fmt.Print(smallInput.String())
	
	// 使用3x3卷积核，stride=1，padding=1
	kernelH, kernelW = 3, 3
	strideH, strideW = 1, 1
	padH, padW = 1, 1
	
	colMatrixPadded := matrix.Im2Col(smallInput, kernelH, kernelW, strideH, strideW, padH, padW)
	
	fmt.Printf("\n带填充的Im2Col结果 (%dx%d):\n", colMatrixPadded.Rows, colMatrixPadded.Cols)
	fmt.Print(colMatrixPadded.String())
	
	// 示例3: Col2Im操作
	fmt.Println("\n3. Col2Im操作:")
	
	// 使用上面的结果进行col2im
	reconstructed := matrix.Col2Im(colMatrixPadded, 2, 2, kernelH, kernelW, strideH, strideW, padH, padW)
	
	fmt.Println("重构的图像 (2x2):")
	fmt.Print(reconstructed.String())
	
	// 示例4: 多通道操作
	fmt.Println("\n4. 多通道Im2Col操作:")
	
	// 创建一个2通道的3x3图像
	multiChannelInput := matrix.NewMatrixFromData([]float64{
		1, 2, 3, // 通道0
		4, 5, 6,
		7, 8, 9,
		10, 11, 12, // 通道1
		13, 14, 15,
		16, 17, 18,
	}, 6, 3) // 6行3列，表示2个通道的3x3图像
	
	fmt.Println("多通道输入 (2通道 x 3x3):")
	fmt.Print(multiChannelInput.String())
	
	channels := 2
	kernelH, kernelW = 2, 2
	strideH, strideW = 1, 1
	padH, padW = 0, 0
	
	multiChannelCol := matrix.Im2ColWithChannels(multiChannelInput, channels, kernelH, kernelW, strideH, strideW, padH, padW)
	
	fmt.Printf("\n多通道Im2Col结果 (%dx%d):\n", multiChannelCol.Rows, multiChannelCol.Cols)
	fmt.Print(multiChannelCol.String())
	
	// 示例5: 卷积运算示例
	fmt.Println("\n5. 使用Im2Col进行卷积运算:")
	
	// 创建一个简单的3x3输入
	convInput := matrix.NewMatrixFromData([]float64{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	}, 3, 3)
	
	// 创建一个2x2卷积核
	kernel := matrix.NewMatrixFromData([]float64{
		1, 0,
		0, 1,
	}, 2, 2)
	
	fmt.Println("输入 (3x3):")
	fmt.Print(convInput.String())
	
	fmt.Println("卷积核 (2x2):")
	fmt.Print(kernel.String())
	
	// 使用im2col转换输入
	convCol := matrix.Im2Col(convInput, 2, 2, 1, 1, 0, 0)
	
	// 将卷积核重塑为行向量
	kernelRow := kernel.Reshape(1, 4)
	
	fmt.Println("Im2Col转换后的输入:")
	fmt.Print(convCol.String())
	
	fmt.Println("卷积核行向量:")
	fmt.Print(kernelRow.String())
	
	// 执行矩阵乘法进行卷积
	convResult := kernelRow.Mul(convCol)
	
	fmt.Printf("\n卷积结果 (%dx%d):\n", convResult.Rows, convResult.Cols)
	fmt.Print(convResult.String())
	
	// 重塑为2x2特征图
	featureMap := convResult.Reshape(2, 2)
	
	fmt.Println("特征图 (2x2):")
	fmt.Print(featureMap.String())
	
	fmt.Println("\n=== 示例完成 ===")
	fmt.Println("Im2Col和Col2Im函数已成功实现并测试!")
	fmt.Println("这些函数将卷积操作转换为高效的矩阵乘法，是CNN实现的关键优化。")
}