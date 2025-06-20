package graph

import (
	"fmt"
	"github.com/user/go-cnn/matrix"
	"math"
)

// GradientCheck 执行梯度检查来验证反向传播的正确性
// 使用数值梯度与反向传播计算的梯度进行比较
func GradientCheck(f func() *Node, params []*Node, epsilon float64, tolerance float64) error {
	// 如果没有指定epsilon，使用默认值
	if epsilon == 0 {
		epsilon = 1e-7
	}

	// 如果没有指定tolerance，使用默认值
	if tolerance == 0 {
		tolerance = 1e-5
	}

	fmt.Printf("开始梯度检查，epsilon=%.2e, tolerance=%.2e\n", epsilon, tolerance)

	for _, param := range params {
		if !param.RequiresGrad {
			continue
		}

		fmt.Printf("\n检查参数 %s (shape: %dx%d)\n", param.Name, param.Value.Rows, param.Value.Cols)

		// 保存原始参数值
		originalValue := param.Value.Copy()

		// 计算解析梯度（通过反向传播）
		loss := f()
		loss.ZeroGrad()
		// 初始化损失的梯度为1（对于标量损失）
		if loss.Gradient == nil {
			if loss.Value.Rows == 1 && loss.Value.Cols == 1 {
				loss.Gradient = matrix.Ones(1, 1)
			} else {
				loss.Gradient = matrix.Ones(loss.Value.Rows, loss.Value.Cols)
			}
		}
		loss.Backward()

		if param.Gradient == nil {
			return fmt.Errorf("参数 %s 的梯度为nil，可能不在计算图中", param.Name)
		}
		analyticalGrad := param.Gradient.Copy()

		// 计算数值梯度
		numericalGrad := matrix.Zeros(param.Value.Rows, param.Value.Cols)

		// 对参数的每个元素计算数值梯度
		for row := 0; row < param.Value.Rows; row++ {
			for col := 0; col < param.Value.Cols; col++ {
				// 保存原始值
				origVal := param.Value.At(row, col)

				// 前向差分：f(x + h)
				param.Value.Set(row, col, origVal+epsilon)
				lossPlus := f()
				fPlus := lossPlus.Value.At(0, 0)

				// 后向差分：f(x - h)
				param.Value.Set(row, col, origVal-epsilon)
				lossMinus := f()
				fMinus := lossMinus.Value.At(0, 0)

				// 中心差分：(f(x+h) - f(x-h)) / (2*h)
				numGrad := (fPlus - fMinus) / (2 * epsilon)
				numericalGrad.Set(row, col, numGrad)

				// 恢复原始值
				param.Value.Set(row, col, origVal)
			}
		}

		// 恢复原始参数值
		param.Value = originalValue

		// 计算相对误差
		diff := analyticalGrad.Sub(numericalGrad)
		diffNorm := 0.0
		analyticalNorm := 0.0
		numericalNorm := 0.0

		for row := 0; row < diff.Rows; row++ {
			for col := 0; col < diff.Cols; col++ {
				d := diff.At(row, col)
				a := analyticalGrad.At(row, col)
				n := numericalGrad.At(row, col)

				diffNorm += d * d
				analyticalNorm += a * a
				numericalNorm += n * n
			}
		}

		diffNorm = math.Sqrt(diffNorm)
		analyticalNorm = math.Sqrt(analyticalNorm)
		numericalNorm = math.Sqrt(numericalNorm)

		// 计算相对误差
		relativeError := diffNorm / (analyticalNorm + numericalNorm + 1e-8)

		fmt.Printf("  差值范数: %.6e\n", diffNorm)
		fmt.Printf("  解析梯度范数: %.6e\n", analyticalNorm)
		fmt.Printf("  数值梯度范数: %.6e\n", numericalNorm)
		fmt.Printf("  相对误差: %.6e\n", relativeError)

		// 显示一些具体的梯度值比较（前5个元素）
		fmt.Println("  前几个元素的梯度比较:")
		count := 0
		for row := 0; row < param.Value.Rows && count < 5; row++ {
			for col := 0; col < param.Value.Cols && count < 5; col++ {
				a := analyticalGrad.At(row, col)
				n := numericalGrad.At(row, col)
				fmt.Printf("    [%d,%d] 解析: %+.6e, 数值: %+.6e, 差: %+.6e\n",
					row, col, a, n, a-n)
				count++
			}
		}

		// 检查是否通过
		if relativeError > tolerance {
			return fmt.Errorf("参数 %s 的梯度检查失败: 相对误差 %.6e > 容差 %.6e",
				param.Name, relativeError, tolerance)
		} else {
			fmt.Printf("  ✓ 通过！相对误差在容差范围内\n")
		}
	}

	fmt.Println("\n所有梯度检查通过！")
	return nil
}

// SimpleGradientCheck 对单个函数进行简单的梯度检查
func SimpleGradientCheck(f func(*matrix.Matrix) float64,
	df func(*matrix.Matrix) *matrix.Matrix,
	x *matrix.Matrix,
	epsilon float64) error {

	if epsilon == 0 {
		epsilon = 1e-7
	}

	// 计算解析梯度
	analyticalGrad := df(x)

	// 计算数值梯度
	numericalGrad := matrix.Zeros(x.Rows, x.Cols)

	for i := 0; i < x.Rows; i++ {
		for j := 0; j < x.Cols; j++ {
			// 保存原始值
			orig := x.At(i, j)

			// 计算f(x + h)
			x.Set(i, j, orig+epsilon)
			fPlus := f(x)

			// 计算f(x - h)
			x.Set(i, j, orig-epsilon)
			fMinus := f(x)

			// 中心差分
			numGrad := (fPlus - fMinus) / (2 * epsilon)
			numericalGrad.Set(i, j, numGrad)

			// 恢复原始值
			x.Set(i, j, orig)
		}
	}

	// 计算误差
	maxError := 0.0
	for i := 0; i < x.Rows; i++ {
		for j := 0; j < x.Cols; j++ {
			a := analyticalGrad.At(i, j)
			n := numericalGrad.At(i, j)
			error := math.Abs(a - n)
			if error > maxError {
				maxError = error
			}
		}
	}

	if maxError > 1e-5 {
		return fmt.Errorf("梯度检查失败: 最大误差 %.6e", maxError)
	}

	return nil
}
