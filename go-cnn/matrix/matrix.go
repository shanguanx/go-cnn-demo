package matrix

import (
	"fmt"
	"math"
	"math/rand"
)

type Matrix struct {
	Data   []float64
	Rows   int
	Cols   int
	Shape  []int
	Stride []int
}

// NewMatrix 创建一个新的矩阵，使用指定的行数和列数
func NewMatrix(rows, cols int) *Matrix {
	return &Matrix{
		Data:   make([]float64, rows*cols),
		Rows:   rows,
		Cols:   cols,
		Shape:  []int{rows, cols},
		Stride: []int{cols, 1},
	}
}

// NewMatrixFromData 使用给定的数据切片创建矩阵，验证数据长度是否匹配维度
func NewMatrixFromData(data []float64, rows, cols int) *Matrix {
	if len(data) != rows*cols {
		panic(fmt.Sprintf("data length %d doesn't match dimensions %dx%d", len(data), rows, cols))
	}
	
	m := NewMatrix(rows, cols)
	copy(m.Data, data)
	return m
}

// NewMatrixFrom2D 使用二维切片创建矩阵
func NewMatrixFrom2D(data [][]float64) *Matrix {
	if len(data) == 0 {
		panic("cannot create matrix from empty 2D slice")
	}
	
	rows := len(data)
	cols := len(data[0])
	
	// 验证所有行的长度一致
	for i, row := range data {
		if len(row) != cols {
			panic(fmt.Sprintf("inconsistent row length at row %d: expected %d, got %d", i, cols, len(row)))
		}
	}
	
	m := NewMatrix(rows, cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			m.Set(i, j, data[i][j])
		}
	}
	return m
}

// At 获取矩阵在位置(i,j)的元素值，会检查边界
func (m *Matrix) At(i, j int) float64 {
	if i < 0 || i >= m.Rows || j < 0 || j >= m.Cols {
		panic(fmt.Sprintf("index out of bounds: (%d, %d) for matrix %dx%d", i, j, m.Rows, m.Cols))
	}
	return m.Data[i*m.Stride[0]+j*m.Stride[1]]
}

// Set 设置矩阵在位置(i,j)的元素值，会检查边界  
func (m *Matrix) Set(i, j int, val float64) {
	if i < 0 || i >= m.Rows || j < 0 || j >= m.Cols {
		panic(fmt.Sprintf("index out of bounds: (%d, %d) for matrix %dx%d", i, j, m.Rows, m.Cols))
	}
	m.Data[i*m.Stride[0]+j*m.Stride[1]] = val
}

// Copy 创建矩阵的深拷贝
func (m *Matrix) Copy() *Matrix {
	result := NewMatrix(m.Rows, m.Cols)
	copy(result.Data, m.Data)
	return result
}

// Reshape 重新塑形矩阵为新的维度，不改变元素总数
func (m *Matrix) Reshape(rows, cols int) *Matrix {
	if rows*cols != m.Rows*m.Cols {
		panic(fmt.Sprintf("cannot reshape %dx%d matrix to %dx%d", m.Rows, m.Cols, rows, cols))
	}
	
	return &Matrix{
		Data:   m.Data,
		Rows:   rows,
		Cols:   cols,
		Shape:  []int{rows, cols},
		Stride: []int{cols, 1},
	}
}

// Add 矩阵加法，返回新矩阵
func (m *Matrix) Add(other *Matrix) *Matrix {
	if m.Rows != other.Rows || m.Cols != other.Cols {
		panic(fmt.Sprintf("matrix dimensions don't match: %dx%d vs %dx%d", m.Rows, m.Cols, other.Rows, other.Cols))
	}
	
	result := NewMatrix(m.Rows, m.Cols)
	for i := 0; i < len(m.Data); i++ {
		result.Data[i] = m.Data[i] + other.Data[i]
	}
	return result
}

// AddInPlace 就地矩阵加法，修改当前矩阵
func (m *Matrix) AddInPlace(other *Matrix) {
	if m.Rows != other.Rows || m.Cols != other.Cols {
		panic(fmt.Sprintf("matrix dimensions don't match: %dx%d vs %dx%d", m.Rows, m.Cols, other.Rows, other.Cols))
	}
	
	for i := 0; i < len(m.Data); i++ {
		m.Data[i] += other.Data[i]
	}
}

// Sub 矩阵减法，返回新矩阵
func (m *Matrix) Sub(other *Matrix) *Matrix {
	if m.Rows != other.Rows || m.Cols != other.Cols {
		panic(fmt.Sprintf("matrix dimensions don't match: %dx%d vs %dx%d", m.Rows, m.Cols, other.Rows, other.Cols))
	}
	
	result := NewMatrix(m.Rows, m.Cols)
	for i := 0; i < len(m.Data); i++ {
		result.Data[i] = m.Data[i] - other.Data[i]
	}
	return result
}

// SubInPlace 就地矩阵减法，修改当前矩阵
func (m *Matrix) SubInPlace(other *Matrix) {
	if m.Rows != other.Rows || m.Cols != other.Cols {
		panic(fmt.Sprintf("matrix dimensions don't match: %dx%d vs %dx%d", m.Rows, m.Cols, other.Rows, other.Cols))
	}
	
	for i := 0; i < len(m.Data); i++ {
		m.Data[i] -= other.Data[i]
	}
}

// Mul 矩阵乘法，返回新矩阵
func (m *Matrix) Mul(other *Matrix) *Matrix {
	if m.Cols != other.Rows {
		panic(fmt.Sprintf("incompatible dimensions for matrix multiplication: %dx%d and %dx%d", 
			m.Rows, m.Cols, other.Rows, other.Cols))
	}
	
	result := NewMatrix(m.Rows, other.Cols)
	
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < other.Cols; j++ {
			sum := 0.0
			for k := 0; k < m.Cols; k++ {
				sum += m.At(i, k) * other.At(k, j)
			}
			result.Set(i, j, sum)
		}
	}
	
	return result
}

// HadamardProduct 逐元素乘法（哈达玛积），返回新矩阵
func (m *Matrix) HadamardProduct(other *Matrix) *Matrix {
	if m.Rows != other.Rows || m.Cols != other.Cols {
		panic(fmt.Sprintf("matrix dimensions don't match: %dx%d vs %dx%d", m.Rows, m.Cols, other.Rows, other.Cols))
	}
	
	result := NewMatrix(m.Rows, m.Cols)
	for i := 0; i < len(m.Data); i++ {
		result.Data[i] = m.Data[i] * other.Data[i]
	}
	return result
}

// HadamardProductInPlace 就地逐元素乘法，修改当前矩阵
func (m *Matrix) HadamardProductInPlace(other *Matrix) {
	if m.Rows != other.Rows || m.Cols != other.Cols {
		panic(fmt.Sprintf("matrix dimensions don't match: %dx%d vs %dx%d", m.Rows, m.Cols, other.Rows, other.Cols))
	}
	
	for i := 0; i < len(m.Data); i++ {
		m.Data[i] *= other.Data[i]
	}
}

// T 矩阵转置，返回新矩阵
func (m *Matrix) T() *Matrix {
	result := NewMatrix(m.Cols, m.Rows)
	
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			result.Set(j, i, m.At(i, j))
		}
	}
	
	return result
}

// Scale 矩阵标量乘法，返回新矩阵
func (m *Matrix) Scale(scalar float64) *Matrix {
	result := NewMatrix(m.Rows, m.Cols)
	for i := 0; i < len(m.Data); i++ {
		result.Data[i] = m.Data[i] * scalar
	}
	return result
}

// ScaleInPlace 就地矩阵标量乘法，修改当前矩阵
func (m *Matrix) ScaleInPlace(scalar float64) {
	for i := 0; i < len(m.Data); i++ {
		m.Data[i] *= scalar
	}
}

// AddScalar 矩阵加标量，返回新矩阵
func (m *Matrix) AddScalar(scalar float64) *Matrix {
	result := NewMatrix(m.Rows, m.Cols)
	for i := 0; i < len(m.Data); i++ {
		result.Data[i] = m.Data[i] + scalar
	}
	return result
}

// AddScalarInPlace 就地矩阵加标量，修改当前矩阵
func (m *Matrix) AddScalarInPlace(scalar float64) {
	for i := 0; i < len(m.Data); i++ {
		m.Data[i] += scalar
	}
}

// Sum 计算矩阵所有元素的和
func (m *Matrix) Sum() float64 {
	sum := 0.0
	for _, v := range m.Data {
		sum += v
	}
	return sum
}

// Mean 计算矩阵所有元素的平均值
func (m *Matrix) Mean() float64 {
	return m.Sum() / float64(len(m.Data))
}

// Max 找到矩阵中的最大值
func (m *Matrix) Max() float64 {
	if len(m.Data) == 0 {
		panic("cannot find max of empty matrix")
	}
	
	max := m.Data[0]
	for _, v := range m.Data[1:] {
		if v > max {
			max = v
		}
	}
	return max
}

// Min 找到矩阵中的最小值
func (m *Matrix) Min() float64 {
	if len(m.Data) == 0 {
		panic("cannot find min of empty matrix")
	}
	
	min := m.Data[0]
	for _, v := range m.Data[1:] {
		if v < min {
			min = v
		}
	}
	return min
}

// Apply 对矩阵每个元素应用函数，返回新矩阵
func (m *Matrix) Apply(fn func(float64) float64) *Matrix {
	result := NewMatrix(m.Rows, m.Cols)
	for i := 0; i < len(m.Data); i++ {
		result.Data[i] = fn(m.Data[i])
	}
	return result
}

// ApplyInPlace 就地对矩阵每个元素应用函数，修改当前矩阵
func (m *Matrix) ApplyInPlace(fn func(float64) float64) {
	for i := 0; i < len(m.Data); i++ {
		m.Data[i] = fn(m.Data[i])
	}
}

// Zeros 创建全零矩阵
func Zeros(rows, cols int) *Matrix {
	return NewMatrix(rows, cols)
}

// Ones 创建全一矩阵
func Ones(rows, cols int) *Matrix {
	m := NewMatrix(rows, cols)
	for i := 0; i < len(m.Data); i++ {
		m.Data[i] = 1.0
	}
	return m
}

// Eye 创建单位矩阵
func Eye(size int) *Matrix {
	m := NewMatrix(size, size)
	for i := 0; i < size; i++ {
		m.Set(i, i, 1.0)
	}
	return m
}

// Random 创建随机矩阵，元素在[min,max)范围内均匀分布
func Random(rows, cols int, min, max float64) *Matrix {
	m := NewMatrix(rows, cols)
	scale := max - min
	for i := 0; i < len(m.Data); i++ {
		m.Data[i] = rand.Float64()*scale + min
	}
	return m
}

// Randn 创建正态分布随机矩阵，指定均值和标准差
func Randn(rows, cols int, mean, stddev float64) *Matrix {
	m := NewMatrix(rows, cols)
	for i := 0; i < len(m.Data); i++ {
		m.Data[i] = rand.NormFloat64()*stddev + mean
	}
	return m
}

// Equals 比较两个矩阵是否相等，允许指定容差
func (m *Matrix) Equals(other *Matrix, tolerance float64) bool {
	if m.Rows != other.Rows || m.Cols != other.Cols {
		return false
	}
	
	for i := 0; i < len(m.Data); i++ {
		if math.Abs(m.Data[i]-other.Data[i]) > tolerance {
			return false
		}
	}
	return true
}

// String 返回矩阵的字符串表示，用于打印和调试
func (m *Matrix) String() string {
	s := fmt.Sprintf("Matrix(%dx%d):\n", m.Rows, m.Cols)
	for i := 0; i < m.Rows; i++ {
		s += "["
		for j := 0; j < m.Cols; j++ {
			if j > 0 {
				s += " "
			}
			s += fmt.Sprintf("%8.4f", m.At(i, j))
		}
		s += "]\n"
	}
	return s
}