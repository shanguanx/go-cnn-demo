package matrix

// Im2Col 将图像的卷积窗口转换为列向量，用于高效的卷积运算
// 参数：
//   - input: 输入图像矩阵 (H x W)
//   - kernelH, kernelW: 卷积核的高度和宽度
//   - strideH, strideW: 步长
//   - padH, padW: 填充大小
// 返回: 列矩阵，每列包含一个卷积窗口的所有元素
func Im2Col(input *Matrix, kernelH, kernelW, strideH, strideW, padH, padW int) *Matrix {
	inputH, inputW := input.Rows, input.Cols
	
	// 计算输出特征图的尺寸
	outputH := (inputH+2*padH-kernelH)/strideH + 1
	outputW := (inputW+2*padW-kernelW)/strideW + 1
	
	// 创建输出矩阵: (kernelH*kernelW) x (outputH*outputW)
	colMatrix := NewMatrix(kernelH*kernelW, outputH*outputW)
	
	// 遍历输出特征图的每个位置
	colIdx := 0
	for yOut := 0; yOut < outputH; yOut++ {
		for xOut := 0; xOut < outputW; xOut++ {
			// 计算在输入图像中的起始位置
			yStart := yOut*strideH - padH
			xStart := xOut*strideW - padW
			
			// 提取卷积窗口的数据
			rowIdx := 0
			for ky := 0; ky < kernelH; ky++ {
				for kx := 0; kx < kernelW; kx++ {
					y := yStart + ky
					x := xStart + kx
					
					var value float64
					// 检查是否在边界内
					if y >= 0 && y < inputH && x >= 0 && x < inputW {
						value = input.At(y, x)
					} else {
						value = 0.0 // 填充零值
					}
					
					colMatrix.Set(rowIdx, colIdx, value)
					rowIdx++
				}
			}
			colIdx++
		}
	}
	
	return colMatrix
}

// Col2Im 将列向量转换回图像格式，用于反向传播
// 参数：
//   - colMatrix: 列矩阵 (kernelH*kernelW) x (outputH*outputW)
//   - inputH, inputW: 原始输入图像的高度和宽度
//   - kernelH, kernelW: 卷积核的高度和宽度
//   - strideH, strideW: 步长
//   - padH, padW: 填充大小
// 返回: 重构的图像矩阵
func Col2Im(colMatrix *Matrix, inputH, inputW, kernelH, kernelW, strideH, strideW, padH, padW int) *Matrix {
	// 计算输出特征图的尺寸
	outputH := (inputH+2*padH-kernelH)/strideH + 1
	outputW := (inputW+2*padW-kernelW)/strideW + 1
	
	// 创建输出图像矩阵
	output := NewMatrix(inputH, inputW)
	
	// 遍历输出特征图的每个位置
	colIdx := 0
	for yOut := 0; yOut < outputH; yOut++ {
		for xOut := 0; xOut < outputW; xOut++ {
			// 计算在输入图像中的起始位置
			yStart := yOut*strideH - padH
			xStart := xOut*strideW - padW
			
			// 将列向量的数据分配回图像对应位置
			rowIdx := 0
			for ky := 0; ky < kernelH; ky++ {
				for kx := 0; kx < kernelW; kx++ {
					y := yStart + ky
					x := xStart + kx
					
					// 检查是否在边界内
					if y >= 0 && y < inputH && x >= 0 && x < inputW {
						currentValue := output.At(y, x)
						newValue := colMatrix.At(rowIdx, colIdx)
						output.Set(y, x, currentValue+newValue)
					}
					rowIdx++
				}
			}
			colIdx++
		}
	}
	
	return output
}

// Im2ColWithChannels 处理多通道图像的im2col操作
// 参数：
//   - input: 输入图像矩阵，按行排列 (channels*H) x W
//   - channels: 通道数
//   - kernelH, kernelW: 卷积核的高度和宽度
//   - strideH, strideW: 步长
//   - padH, padW: 填充大小
// 返回: 列矩阵，每列包含一个卷积窗口的所有元素
func Im2ColWithChannels(input *Matrix, channels, kernelH, kernelW, strideH, strideW, padH, padW int) *Matrix {
	inputH := input.Rows / channels
	inputW := input.Cols
	
	// 计算输出特征图的尺寸
	outputH := (inputH+2*padH-kernelH)/strideH + 1
	outputW := (inputW+2*padW-kernelW)/strideW + 1
	
	// 创建输出矩阵: (channels*kernelH*kernelW) x (outputH*outputW)
	colMatrix := NewMatrix(channels*kernelH*kernelW, outputH*outputW)
	
	// 遍历输出特征图的每个位置
	colIdx := 0
	for yOut := 0; yOut < outputH; yOut++ {
		for xOut := 0; xOut < outputW; xOut++ {
			// 计算在输入图像中的起始位置
			yStart := yOut*strideH - padH
			xStart := xOut*strideW - padW
			
			// 提取卷积窗口的数据
			rowIdx := 0
			for c := 0; c < channels; c++ {
				for ky := 0; ky < kernelH; ky++ {
					for kx := 0; kx < kernelW; kx++ {
						y := yStart + ky
						x := xStart + kx
						
						var value float64
						// 检查是否在边界内
						if y >= 0 && y < inputH && x >= 0 && x < inputW {
							value = input.At(c*inputH+y, x)
						} else {
							value = 0.0 // 填充零值
						}
						
						colMatrix.Set(rowIdx, colIdx, value)
						rowIdx++
					}
				}
			}
			colIdx++
		}
	}
	
	return colMatrix
}

// Col2ImWithChannels 处理多通道图像的col2im操作
// 参数：
//   - colMatrix: 列矩阵 (channels*kernelH*kernelW) x (outputH*outputW)
//   - channels: 通道数
//   - inputH, inputW: 原始输入图像的高度和宽度
//   - kernelH, kernelW: 卷积核的高度和宽度
//   - strideH, strideW: 步长
//   - padH, padW: 填充大小
// 返回: 重构的图像矩阵
func Col2ImWithChannels(colMatrix *Matrix, channels, inputH, inputW, kernelH, kernelW, strideH, strideW, padH, padW int) *Matrix {
	// 计算输出特征图的尺寸
	outputH := (inputH+2*padH-kernelH)/strideH + 1
	outputW := (inputW+2*padW-kernelW)/strideW + 1
	
	// 创建输出图像矩阵
	output := NewMatrix(channels*inputH, inputW)
	
	// 遍历输出特征图的每个位置
	colIdx := 0
	for yOut := 0; yOut < outputH; yOut++ {
		for xOut := 0; xOut < outputW; xOut++ {
			// 计算在输入图像中的起始位置
			yStart := yOut*strideH - padH
			xStart := xOut*strideW - padW
			
			// 将列向量的数据分配回图像对应位置
			rowIdx := 0
			for c := 0; c < channels; c++ {
				for ky := 0; ky < kernelH; ky++ {
					for kx := 0; kx < kernelW; kx++ {
						y := yStart + ky
						x := xStart + kx
						
						// 检查是否在边界内
						if y >= 0 && y < inputH && x >= 0 && x < inputW {
							currentValue := output.At(c*inputH+y, x)
							newValue := colMatrix.At(rowIdx, colIdx)
							output.Set(c*inputH+y, x, currentValue+newValue)
						}
						rowIdx++
					}
				}
			}
			colIdx++
		}
	}
	
	return output
}