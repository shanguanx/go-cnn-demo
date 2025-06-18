package tests

import (
	"math"
	"testing"

	"github.com/user/go-cnn/matrix"
)

func TestMatrixCreation(t *testing.T) {
	m := matrix.NewMatrix(3, 4)
	if m.Rows != 3 || m.Cols != 4 {
		t.Errorf("Expected 3x4 matrix, got %dx%d", m.Rows, m.Cols)
	}
	if len(m.Data) != 12 {
		t.Errorf("Expected data length 12, got %d", len(m.Data))
	}
}

func TestMatrixFromData(t *testing.T) {
	data := []float64{1, 2, 3, 4, 5, 6}
	m := matrix.NewMatrixFromData(data, 2, 3)
	
	if m.At(0, 0) != 1 || m.At(0, 1) != 2 || m.At(0, 2) != 3 {
		t.Errorf("First row incorrect")
	}
	if m.At(1, 0) != 4 || m.At(1, 1) != 5 || m.At(1, 2) != 6 {
		t.Errorf("Second row incorrect")
	}
}

func TestMatrixSetGet(t *testing.T) {
	m := matrix.NewMatrix(2, 2)
	m.Set(0, 0, 1.5)
	m.Set(1, 1, 2.5)
	
	if m.At(0, 0) != 1.5 {
		t.Errorf("Expected 1.5, got %f", m.At(0, 0))
	}
	if m.At(1, 1) != 2.5 {
		t.Errorf("Expected 2.5, got %f", m.At(1, 1))
	}
}

func TestMatrixAdd(t *testing.T) {
	a := matrix.NewMatrixFromData([]float64{1, 2, 3, 4}, 2, 2)
	b := matrix.NewMatrixFromData([]float64{5, 6, 7, 8}, 2, 2)
	
	c := a.Add(b)
	
	expected := []float64{6, 8, 10, 12}
	for i, v := range expected {
		if c.Data[i] != v {
			t.Errorf("Expected %f, got %f at index %d", v, c.Data[i], i)
		}
	}
}

func TestMatrixSub(t *testing.T) {
	a := matrix.NewMatrixFromData([]float64{5, 6, 7, 8}, 2, 2)
	b := matrix.NewMatrixFromData([]float64{1, 2, 3, 4}, 2, 2)
	
	c := a.Sub(b)
	
	expected := []float64{4, 4, 4, 4}
	for i, v := range expected {
		if c.Data[i] != v {
			t.Errorf("Expected %f, got %f at index %d", v, c.Data[i], i)
		}
	}
}

func TestMatrixMul(t *testing.T) {
	a := matrix.NewMatrixFromData([]float64{1, 2, 3, 4}, 2, 2)
	b := matrix.NewMatrixFromData([]float64{5, 6, 7, 8}, 2, 2)
	
	c := a.Mul(b)
	
	if c.At(0, 0) != 19 || c.At(0, 1) != 22 {
		t.Errorf("First row incorrect: [%f, %f]", c.At(0, 0), c.At(0, 1))
	}
	if c.At(1, 0) != 43 || c.At(1, 1) != 50 {
		t.Errorf("Second row incorrect: [%f, %f]", c.At(1, 0), c.At(1, 1))
	}
}

func TestMatrixHadamardProduct(t *testing.T) {
	a := matrix.NewMatrixFromData([]float64{1, 2, 3, 4}, 2, 2)
	b := matrix.NewMatrixFromData([]float64{5, 6, 7, 8}, 2, 2)
	
	c := a.HadamardProduct(b)
	
	expected := []float64{5, 12, 21, 32}
	for i, v := range expected {
		if c.Data[i] != v {
			t.Errorf("Expected %f, got %f at index %d", v, c.Data[i], i)
		}
	}
}

func TestMatrixTranspose(t *testing.T) {
	a := matrix.NewMatrixFromData([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	
	b := a.T()
	
	if b.Rows != 3 || b.Cols != 2 {
		t.Errorf("Expected 3x2 matrix, got %dx%d", b.Rows, b.Cols)
	}
	
	if b.At(0, 0) != 1 || b.At(0, 1) != 4 {
		t.Errorf("First row incorrect: [%f, %f]", b.At(0, 0), b.At(0, 1))
	}
	if b.At(1, 0) != 2 || b.At(1, 1) != 5 {
		t.Errorf("Second row incorrect: [%f, %f]", b.At(1, 0), b.At(1, 1))
	}
	if b.At(2, 0) != 3 || b.At(2, 1) != 6 {
		t.Errorf("Third row incorrect: [%f, %f]", b.At(2, 0), b.At(2, 1))
	}
}

func TestMatrixScale(t *testing.T) {
	a := matrix.NewMatrixFromData([]float64{1, 2, 3, 4}, 2, 2)
	
	b := a.Scale(2.5)
	
	expected := []float64{2.5, 5.0, 7.5, 10.0}
	for i, v := range expected {
		if b.Data[i] != v {
			t.Errorf("Expected %f, got %f at index %d", v, b.Data[i], i)
		}
	}
}

func TestMatrixUtilityFunctions(t *testing.T) {
	a := matrix.NewMatrixFromData([]float64{1, 2, 3, 4}, 2, 2)
	
	if a.Sum() != 10.0 {
		t.Errorf("Expected sum 10.0, got %f", a.Sum())
	}
	
	if a.Mean() != 2.5 {
		t.Errorf("Expected mean 2.5, got %f", a.Mean())
	}
	
	if a.Max() != 4.0 {
		t.Errorf("Expected max 4.0, got %f", a.Max())
	}
	
	if a.Min() != 1.0 {
		t.Errorf("Expected min 1.0, got %f", a.Min())
	}
}

func TestMatrixApply(t *testing.T) {
	a := matrix.NewMatrixFromData([]float64{1, 4, 9, 16}, 2, 2)
	
	b := a.Apply(math.Sqrt)
	
	expected := []float64{1, 2, 3, 4}
	for i, v := range expected {
		if math.Abs(b.Data[i]-v) > 1e-6 {
			t.Errorf("Expected %f, got %f at index %d", v, b.Data[i], i)
		}
	}
}

func TestMatrixEquals(t *testing.T) {
	a := matrix.NewMatrixFromData([]float64{1.0, 2.0, 3.0, 4.0}, 2, 2)
	b := matrix.NewMatrixFromData([]float64{1.0001, 1.9999, 3.0001, 3.9999}, 2, 2)
	
	if !a.Equals(b, 0.01) {
		t.Errorf("Matrices should be equal within tolerance")
	}
	
	if a.Equals(b, 0.0001) {
		t.Errorf("Matrices should not be equal with strict tolerance")
	}
}

func TestMatrixCreationFunctions(t *testing.T) {
	zeros := matrix.Zeros(2, 3)
	for _, v := range zeros.Data {
		if v != 0.0 {
			t.Errorf("Expected 0.0, got %f", v)
		}
	}
	
	ones := matrix.Ones(2, 3)
	for _, v := range ones.Data {
		if v != 1.0 {
			t.Errorf("Expected 1.0, got %f", v)
		}
	}
	
	eye := matrix.Eye(3)
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			expected := 0.0
			if i == j {
				expected = 1.0
			}
			if eye.At(i, j) != expected {
				t.Errorf("Expected %f at (%d,%d), got %f", expected, i, j, eye.At(i, j))
			}
		}
	}
}

func TestBroadcastAdd(t *testing.T) {
	a := matrix.NewMatrixFromData([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	
	scalar := matrix.NewMatrixFromData([]float64{10}, 1, 1)
	result := a.BroadcastAdd(scalar)
	expected := []float64{11, 12, 13, 14, 15, 16}
	for i, v := range expected {
		if result.Data[i] != v {
			t.Errorf("Expected %f, got %f at index %d", v, result.Data[i], i)
		}
	}
	
	row := matrix.NewMatrixFromData([]float64{10, 20, 30}, 1, 3)
	result2 := a.BroadcastAdd(row)
	expected2 := []float64{11, 22, 33, 14, 25, 36}
	for i, v := range expected2 {
		if result2.Data[i] != v {
			t.Errorf("Expected %f, got %f at index %d", v, result2.Data[i], i)
		}
	}
}

func TestBroadcastMul(t *testing.T) {
	a := matrix.NewMatrixFromData([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	
	scalar := matrix.NewMatrixFromData([]float64{2}, 1, 1)
	result := a.BroadcastMul(scalar)
	expected := []float64{2, 4, 6, 8, 10, 12}
	for i, v := range expected {
		if result.Data[i] != v {
			t.Errorf("Expected %f, got %f at index %d", v, result.Data[i], i)
		}
	}
}

func TestCanBroadcast(t *testing.T) {
	testCases := []struct {
		shape1   []int
		shape2   []int
		expected bool
	}{
		{[]int{3, 4}, []int{3, 4}, true},
		{[]int{3, 4}, []int{1, 4}, true},
		{[]int{3, 4}, []int{3, 1}, true},
		{[]int{3, 4}, []int{1, 1}, true},
		{[]int{3, 4}, []int{4}, true},
		{[]int{3, 4}, []int{1}, true},
		{[]int{3, 4}, []int{3, 5}, false},
		{[]int{3, 4}, []int{5, 4}, false},
		{[]int{2, 3}, []int{2, 4}, false},
		{[]int{}, []int{3, 4}, true},
		{[]int{3, 4}, []int{}, true},
	}
	
	for i, tc := range testCases {
		result := matrix.CanBroadcast(tc.shape1, tc.shape2)
		if result != tc.expected {
			t.Errorf("Test case %d: canBroadcast(%v, %v) = %v, expected %v", i, tc.shape1, tc.shape2, result, tc.expected)
		}
	}
}

func TestGetBroadcastShape(t *testing.T) {
	testCases := []struct {
		shape1   []int
		shape2   []int
		expected []int
	}{
		{[]int{3, 4}, []int{3, 4}, []int{3, 4}},
		{[]int{3, 4}, []int{1, 4}, []int{3, 4}},
		{[]int{3, 4}, []int{3, 1}, []int{3, 4}},
		{[]int{3, 4}, []int{1, 1}, []int{3, 4}},
		{[]int{3, 4}, []int{4}, []int{3, 4}},
		{[]int{3, 4}, []int{1}, []int{3, 4}},
		{[]int{2, 1}, []int{1, 3}, []int{2, 3}},
	}
	
	for i, tc := range testCases {
		result := matrix.GetBroadcastShape(tc.shape1, tc.shape2)
		if len(result) != len(tc.expected) {
			t.Errorf("Test case %d: getBroadcastShape(%v, %v) length = %d, expected %d", i, tc.shape1, tc.shape2, len(result), len(tc.expected))
			continue
		}
		for j, v := range result {
			if v != tc.expected[j] {
				t.Errorf("Test case %d: getBroadcastShape(%v, %v)[%d] = %d, expected %d", i, tc.shape1, tc.shape2, j, v, tc.expected[j])
			}
		}
	}
}

func TestGetBroadcastShapePanic(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Expected panic for incompatible shapes")
		}
	}()
	matrix.GetBroadcastShape([]int{3, 4}, []int{3, 5})
}

func TestBroadcastAddComprehensive(t *testing.T) {
	a := matrix.NewMatrixFromData([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	
	scalar := matrix.NewMatrixFromData([]float64{10}, 1, 1)
	result := a.BroadcastAdd(scalar)
	expected := []float64{11, 12, 13, 14, 15, 16}
	for i, v := range expected {
		if result.Data[i] != v {
			t.Errorf("Scalar add: Expected %f, got %f at index %d", v, result.Data[i], i)
		}
	}
	
	row := matrix.NewMatrixFromData([]float64{10, 20, 30}, 1, 3)
	result2 := a.BroadcastAdd(row)
	expected2 := []float64{11, 22, 33, 14, 25, 36}
	for i, v := range expected2 {
		if result2.Data[i] != v {
			t.Errorf("Row broadcast add: Expected %f, got %f at index %d", v, result2.Data[i], i)
		}
	}
	
	col := matrix.NewMatrixFromData([]float64{100, 200}, 2, 1)
	result3 := a.BroadcastAdd(col)
	expected3 := []float64{101, 102, 103, 204, 205, 206}
	for i, v := range expected3 {
		if result3.Data[i] != v {
			t.Errorf("Column broadcast add: Expected %f, got %f at index %d", v, result3.Data[i], i)
		}
	}
	
	same := matrix.NewMatrixFromData([]float64{1, 1, 1, 1, 1, 1}, 2, 3)
	result4 := a.BroadcastAdd(same)
	expected4 := []float64{2, 3, 4, 5, 6, 7}
	for i, v := range expected4 {
		if result4.Data[i] != v {
			t.Errorf("Same size add: Expected %f, got %f at index %d", v, result4.Data[i], i)
		}
	}
}

func TestBroadcastSub(t *testing.T) {
	a := matrix.NewMatrixFromData([]float64{10, 20, 30, 40, 50, 60}, 2, 3)
	
	scalar := matrix.NewMatrixFromData([]float64{5}, 1, 1)
	result := a.BroadcastSub(scalar)
	expected := []float64{5, 15, 25, 35, 45, 55}
	for i, v := range expected {
		if result.Data[i] != v {
			t.Errorf("Scalar sub: Expected %f, got %f at index %d", v, result.Data[i], i)
		}
	}
	
	row := matrix.NewMatrixFromData([]float64{1, 2, 3}, 1, 3)
	result2 := a.BroadcastSub(row)
	expected2 := []float64{9, 18, 27, 39, 48, 57}
	for i, v := range expected2 {
		if result2.Data[i] != v {
			t.Errorf("Row broadcast sub: Expected %f, got %f at index %d", v, result2.Data[i], i)
		}
	}
	
	col := matrix.NewMatrixFromData([]float64{10, 20}, 2, 1)
	result3 := a.BroadcastSub(col)
	expected3 := []float64{0, 10, 20, 20, 30, 40}
	for i, v := range expected3 {
		if result3.Data[i] != v {
			t.Errorf("Column broadcast sub: Expected %f, got %f at index %d", v, result3.Data[i], i)
		}
	}
	
	same := matrix.NewMatrixFromData([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	result4 := a.BroadcastSub(same)
	expected4 := []float64{9, 18, 27, 36, 45, 54}
	for i, v := range expected4 {
		if result4.Data[i] != v {
			t.Errorf("Same size sub: Expected %f, got %f at index %d", v, result4.Data[i], i)
		}
	}
}

func TestBroadcastMulComprehensive(t *testing.T) {
	a := matrix.NewMatrixFromData([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	
	scalar := matrix.NewMatrixFromData([]float64{2}, 1, 1)
	result := a.BroadcastMul(scalar)
	expected := []float64{2, 4, 6, 8, 10, 12}
	for i, v := range expected {
		if result.Data[i] != v {
			t.Errorf("Scalar mul: Expected %f, got %f at index %d", v, result.Data[i], i)
		}
	}
	
	row := matrix.NewMatrixFromData([]float64{2, 3, 4}, 1, 3)
	result2 := a.BroadcastMul(row)
	expected2 := []float64{2, 6, 12, 8, 15, 24}
	for i, v := range expected2 {
		if result2.Data[i] != v {
			t.Errorf("Row broadcast mul: Expected %f, got %f at index %d", v, result2.Data[i], i)
		}
	}
	
	col := matrix.NewMatrixFromData([]float64{10, 100}, 2, 1)
	result3 := a.BroadcastMul(col)
	expected3 := []float64{10, 20, 30, 400, 500, 600}
	for i, v := range expected3 {
		if result3.Data[i] != v {
			t.Errorf("Column broadcast mul: Expected %f, got %f at index %d", v, result3.Data[i], i)
		}
	}
	
	same := matrix.NewMatrixFromData([]float64{2, 2, 2, 2, 2, 2}, 2, 3)
	result4 := a.BroadcastMul(same)
	expected4 := []float64{2, 4, 6, 8, 10, 12}
	for i, v := range expected4 {
		if result4.Data[i] != v {
			t.Errorf("Same size mul: Expected %f, got %f at index %d", v, result4.Data[i], i)
		}
	}
}

func TestBroadcastDiv(t *testing.T) {
	a := matrix.NewMatrixFromData([]float64{10, 20, 30, 40, 50, 60}, 2, 3)
	
	scalar := matrix.NewMatrixFromData([]float64{2}, 1, 1)
	result := a.BroadcastDiv(scalar)
	expected := []float64{5, 10, 15, 20, 25, 30}
	for i, v := range expected {
		if result.Data[i] != v {
			t.Errorf("Scalar div: Expected %f, got %f at index %d", v, result.Data[i], i)
		}
	}
	
	row := matrix.NewMatrixFromData([]float64{1, 2, 3}, 1, 3)
	result2 := a.BroadcastDiv(row)
	expected2 := []float64{10, 10, 10, 40, 25, 20}
	for i, v := range expected2 {
		if result2.Data[i] != v {
			t.Errorf("Row broadcast div: Expected %f, got %f at index %d", v, result2.Data[i], i)
		}
	}
	
	col := matrix.NewMatrixFromData([]float64{10, 20}, 2, 1)
	result3 := a.BroadcastDiv(col)
	expected3 := []float64{1, 2, 3, 2, 2.5, 3}
	for i, v := range expected3 {
		if result3.Data[i] != v {
			t.Errorf("Column broadcast div: Expected %f, got %f at index %d", v, result3.Data[i], i)
		}
	}
	
	same := matrix.NewMatrixFromData([]float64{2, 4, 6, 8, 10, 12}, 2, 3)
	result4 := a.BroadcastDiv(same)
	expected4 := []float64{5, 5, 5, 5, 5, 5}
	for i, v := range expected4 {
		if result4.Data[i] != v {
			t.Errorf("Same size div: Expected %f, got %f at index %d", v, result4.Data[i], i)
		}
	}
}

func TestBroadcastDivByZero(t *testing.T) {
	a := matrix.NewMatrixFromData([]float64{1, 2, 3, 4}, 2, 2)
	
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Expected panic for division by zero (scalar)")
		}
	}()
	zero := matrix.NewMatrixFromData([]float64{0}, 1, 1)
	a.BroadcastDiv(zero)
}

func TestBroadcastDivByZeroRow(t *testing.T) {
	a := matrix.NewMatrixFromData([]float64{1, 2, 3, 4}, 2, 2)
	
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Expected panic for division by zero (row)")
		}
	}()
	zeroRow := matrix.NewMatrixFromData([]float64{1, 0}, 1, 2)
	a.BroadcastDiv(zeroRow)
}

func TestBroadcastDivByZeroCol(t *testing.T) {
	a := matrix.NewMatrixFromData([]float64{1, 2, 3, 4}, 2, 2)
	
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Expected panic for division by zero (column)")
		}
	}()
	zeroCol := matrix.NewMatrixFromData([]float64{1, 0}, 2, 1)
	a.BroadcastDiv(zeroCol)
}

func TestSumAxis(t *testing.T) {
	a := matrix.NewMatrixFromData([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	
	sumAxis0 := a.SumAxis(0, true)
	if sumAxis0.Rows != 1 || sumAxis0.Cols != 3 {
		t.Errorf("Expected 1x3 matrix, got %dx%d", sumAxis0.Rows, sumAxis0.Cols)
	}
	expected0 := []float64{5, 7, 9}
	for i, v := range expected0 {
		if sumAxis0.Data[i] != v {
			t.Errorf("Sum axis 0: Expected %f, got %f at index %d", v, sumAxis0.Data[i], i)
		}
	}
	
	sumAxis1 := a.SumAxis(1, true)
	if sumAxis1.Rows != 2 || sumAxis1.Cols != 1 {
		t.Errorf("Expected 2x1 matrix, got %dx%d", sumAxis1.Rows, sumAxis1.Cols)
	}
	expected1 := []float64{6, 15}
	for i, v := range expected1 {
		if sumAxis1.Data[i] != v {
			t.Errorf("Sum axis 1: Expected %f, got %f at index %d", v, sumAxis1.Data[i], i)
		}
	}
	
	sumAxis0False := a.SumAxis(0, false)
	if sumAxis0False.Rows != 1 || sumAxis0False.Cols != 3 {
		t.Errorf("Expected 1x3 matrix with keepDims=false, got %dx%d", sumAxis0False.Rows, sumAxis0False.Cols)
	}
	
	sumAxis1False := a.SumAxis(1, false)
	if sumAxis1False.Rows != 2 || sumAxis1False.Cols != 1 {
		t.Errorf("Expected 2x1 matrix with keepDims=false, got %dx%d", sumAxis1False.Rows, sumAxis1False.Cols)
	}
}

func TestSumAxisInvalidAxis(t *testing.T) {
	a := matrix.NewMatrixFromData([]float64{1, 2, 3, 4}, 2, 2)
	
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Expected panic for invalid axis")
		}
	}()
	a.SumAxis(2, true)
}

func TestMeanAxis(t *testing.T) {
	a := matrix.NewMatrixFromData([]float64{2, 4, 6, 8, 10, 12}, 2, 3)
	
	meanAxis0 := a.MeanAxis(0, true)
	if meanAxis0.Rows != 1 || meanAxis0.Cols != 3 {
		t.Errorf("Expected 1x3 matrix, got %dx%d", meanAxis0.Rows, meanAxis0.Cols)
	}
	expected0 := []float64{5, 7, 9}
	for i, v := range expected0 {
		if meanAxis0.Data[i] != v {
			t.Errorf("Mean axis 0: Expected %f, got %f at index %d", v, meanAxis0.Data[i], i)
		}
	}
	
	meanAxis1 := a.MeanAxis(1, true)
	if meanAxis1.Rows != 2 || meanAxis1.Cols != 1 {
		t.Errorf("Expected 2x1 matrix, got %dx%d", meanAxis1.Rows, meanAxis1.Cols)
	}
	expected1 := []float64{4, 10}
	for i, v := range expected1 {
		if meanAxis1.Data[i] != v {
			t.Errorf("Mean axis 1: Expected %f, got %f at index %d", v, meanAxis1.Data[i], i)
		}
	}
	
	meanAxis0False := a.MeanAxis(0, false)
	if meanAxis0False.Rows != 1 || meanAxis0False.Cols != 3 {
		t.Errorf("Expected 1x3 matrix with keepDims=false, got %dx%d", meanAxis0False.Rows, meanAxis0False.Cols)
	}
	
	meanAxis1False := a.MeanAxis(1, false)
	if meanAxis1False.Rows != 2 || meanAxis1False.Cols != 1 {
		t.Errorf("Expected 2x1 matrix with keepDims=false, got %dx%d", meanAxis1False.Rows, meanAxis1False.Cols)
	}
}