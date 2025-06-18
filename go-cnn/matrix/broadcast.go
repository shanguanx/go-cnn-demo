package matrix

import (
	"fmt"
)

// canBroadcast 检查两个形状是否可以进行广播操作
// 根据NumPy广播规则，从最后一个维度开始比较，维度大小必须相等或其中一个为1
func canBroadcast(shape1, shape2 []int) bool {
	maxLen := len(shape1)
	if len(shape2) > maxLen {
		maxLen = len(shape2)
	}
	
	for i := 0; i < maxLen; i++ {
		dim1 := 1
		dim2 := 1
		
		if i < len(shape1) {
			dim1 = shape1[len(shape1)-1-i]
		}
		if i < len(shape2) {
			dim2 = shape2[len(shape2)-1-i]
		}
		
		if dim1 != dim2 && dim1 != 1 && dim2 != 1 {
			return false
		}
	}
	
	return true
}

// getBroadcastShape 计算两个形状广播后的结果形状
// 返回广播操作后的维度大小数组
func getBroadcastShape(shape1, shape2 []int) []int {
	if !canBroadcast(shape1, shape2) {
		panic(fmt.Sprintf("cannot broadcast shapes %v and %v", shape1, shape2))
	}
	
	maxLen := len(shape1)
	if len(shape2) > maxLen {
		maxLen = len(shape2)
	}
	
	result := make([]int, maxLen)
	
	for i := 0; i < maxLen; i++ {
		dim1 := 1
		dim2 := 1
		
		if i < len(shape1) {
			dim1 = shape1[len(shape1)-1-i]
		}
		if i < len(shape2) {
			dim2 = shape2[len(shape2)-1-i]
		}
		
		if dim1 > dim2 {
			result[maxLen-1-i] = dim1
		} else {
			result[maxLen-1-i] = dim2
		}
	}
	
	return result
}

// BroadcastAdd 执行矩阵的广播加法运算
// 支持标量、行向量、列向量与矩阵的广播加法
func (m *Matrix) BroadcastAdd(other *Matrix) *Matrix {
	if !canBroadcast(m.Shape, other.Shape) {
		panic(fmt.Sprintf("cannot broadcast shapes %v and %v", m.Shape, other.Shape))
	}
	
	if m.Rows == other.Rows && m.Cols == other.Cols {
		return m.Add(other)
	}
	
	if other.Rows == 1 && other.Cols == 1 {
		return m.AddScalar(other.Data[0])
	}
	
	if other.Rows == 1 && other.Cols == m.Cols {
		result := m.Copy()
		for i := 0; i < m.Rows; i++ {
			for j := 0; j < m.Cols; j++ {
				result.Set(i, j, m.At(i, j)+other.At(0, j))
			}
		}
		return result
	}
	
	if other.Rows == m.Rows && other.Cols == 1 {
		result := m.Copy()
		for i := 0; i < m.Rows; i++ {
			for j := 0; j < m.Cols; j++ {
				result.Set(i, j, m.At(i, j)+other.At(i, 0))
			}
		}
		return result
	}
	
	panic(fmt.Sprintf("broadcast add not implemented for shapes %v and %v", m.Shape, other.Shape))
}

// BroadcastSub 执行矩阵的广播减法运算
// 支持标量、行向量、列向量与矩阵的广播减法
func (m *Matrix) BroadcastSub(other *Matrix) *Matrix {
	if !canBroadcast(m.Shape, other.Shape) {
		panic(fmt.Sprintf("cannot broadcast shapes %v and %v", m.Shape, other.Shape))
	}
	
	if m.Rows == other.Rows && m.Cols == other.Cols {
		return m.Sub(other)
	}
	
	if other.Rows == 1 && other.Cols == 1 {
		return m.AddScalar(-other.Data[0])
	}
	
	if other.Rows == 1 && other.Cols == m.Cols {
		result := m.Copy()
		for i := 0; i < m.Rows; i++ {
			for j := 0; j < m.Cols; j++ {
				result.Set(i, j, m.At(i, j)-other.At(0, j))
			}
		}
		return result
	}
	
	if other.Rows == m.Rows && other.Cols == 1 {
		result := m.Copy()
		for i := 0; i < m.Rows; i++ {
			for j := 0; j < m.Cols; j++ {
				result.Set(i, j, m.At(i, j)-other.At(i, 0))
			}
		}
		return result
	}
	
	panic(fmt.Sprintf("broadcast sub not implemented for shapes %v and %v", m.Shape, other.Shape))
}

// BroadcastMul 执行矩阵的广播乘法运算（元素级乘法）
// 支持标量、行向量、列向量与矩阵的广播乘法
func (m *Matrix) BroadcastMul(other *Matrix) *Matrix {
	if !canBroadcast(m.Shape, other.Shape) {
		panic(fmt.Sprintf("cannot broadcast shapes %v and %v", m.Shape, other.Shape))
	}
	
	if m.Rows == other.Rows && m.Cols == other.Cols {
		return m.HadamardProduct(other)
	}
	
	if other.Rows == 1 && other.Cols == 1 {
		return m.Scale(other.Data[0])
	}
	
	if other.Rows == 1 && other.Cols == m.Cols {
		result := m.Copy()
		for i := 0; i < m.Rows; i++ {
			for j := 0; j < m.Cols; j++ {
				result.Set(i, j, m.At(i, j)*other.At(0, j))
			}
		}
		return result
	}
	
	if other.Rows == m.Rows && other.Cols == 1 {
		result := m.Copy()
		for i := 0; i < m.Rows; i++ {
			for j := 0; j < m.Cols; j++ {
				result.Set(i, j, m.At(i, j)*other.At(i, 0))
			}
		}
		return result
	}
	
	panic(fmt.Sprintf("broadcast mul not implemented for shapes %v and %v", m.Shape, other.Shape))
}

// BroadcastDiv 执行矩阵的广播除法运算（元素级除法）
// 支持标量、行向量、列向量与矩阵的广播除法，会检查除零错误
func (m *Matrix) BroadcastDiv(other *Matrix) *Matrix {
	if !canBroadcast(m.Shape, other.Shape) {
		panic(fmt.Sprintf("cannot broadcast shapes %v and %v", m.Shape, other.Shape))
	}
	
	if m.Rows == other.Rows && m.Cols == other.Cols {
		result := NewMatrix(m.Rows, m.Cols)
		for i := 0; i < len(m.Data); i++ {
			if other.Data[i] == 0 {
				panic("division by zero")
			}
			result.Data[i] = m.Data[i] / other.Data[i]
		}
		return result
	}
	
	if other.Rows == 1 && other.Cols == 1 {
		if other.Data[0] == 0 {
			panic("division by zero")
		}
		return m.Scale(1.0 / other.Data[0])
	}
	
	if other.Rows == 1 && other.Cols == m.Cols {
		result := m.Copy()
		for i := 0; i < m.Rows; i++ {
			for j := 0; j < m.Cols; j++ {
				divisor := other.At(0, j)
				if divisor == 0 {
					panic("division by zero")
				}
				result.Set(i, j, m.At(i, j)/divisor)
			}
		}
		return result
	}
	
	if other.Rows == m.Rows && other.Cols == 1 {
		result := m.Copy()
		for i := 0; i < m.Rows; i++ {
			for j := 0; j < m.Cols; j++ {
				divisor := other.At(i, 0)
				if divisor == 0 {
					panic("division by zero")
				}
				result.Set(i, j, m.At(i, j)/divisor)
			}
		}
		return result
	}
	
	panic(fmt.Sprintf("broadcast div not implemented for shapes %v and %v", m.Shape, other.Shape))
}

// SumAxis 沿指定轴对矩阵元素求和
// axis: 0表示沿行求和（按列汇总），1表示沿列求和（按行汇总）
// keepDims: 是否保持原有维度数量
func (m *Matrix) SumAxis(axis int, keepDims bool) *Matrix {
	if axis != 0 && axis != 1 {
		panic(fmt.Sprintf("axis must be 0 or 1, got %d", axis))
	}
	
	if axis == 0 {
		var result *Matrix
		if keepDims {
			result = NewMatrix(1, m.Cols)
		} else {
			result = NewMatrix(1, m.Cols)
			result.Shape = []int{m.Cols}
		}
		
		for j := 0; j < m.Cols; j++ {
			sum := 0.0
			for i := 0; i < m.Rows; i++ {
				sum += m.At(i, j)
			}
			result.Set(0, j, sum)
		}
		return result
	} else {
		var result *Matrix
		if keepDims {
			result = NewMatrix(m.Rows, 1)
		} else {
			result = NewMatrix(m.Rows, 1)
			result.Shape = []int{m.Rows}
		}
		
		for i := 0; i < m.Rows; i++ {
			sum := 0.0
			for j := 0; j < m.Cols; j++ {
				sum += m.At(i, j)
			}
			result.Set(i, 0, sum)
		}
		return result
	}
}

// MeanAxis 沿指定轴计算矩阵元素的平均值
// axis: 0表示沿行计算均值（按列平均），1表示沿列计算均值（按行平均）
// keepDims: 是否保持原有维度数量
func (m *Matrix) MeanAxis(axis int, keepDims bool) *Matrix {
	sumResult := m.SumAxis(axis, keepDims)
	
	if axis == 0 {
		sumResult.ScaleInPlace(1.0 / float64(m.Rows))
	} else {
		sumResult.ScaleInPlace(1.0 / float64(m.Cols))
	}
	
	return sumResult
}

// CanBroadcast 导出版本的canBroadcast函数，供测试使用
func CanBroadcast(shape1, shape2 []int) bool {
	return canBroadcast(shape1, shape2)
}

// GetBroadcastShape 导出版本的getBroadcastShape函数，供测试使用
func GetBroadcastShape(shape1, shape2 []int) []int {
	return getBroadcastShape(shape1, shape2)
}