package tests

import (
	"fmt"
	"testing"

	"github.com/user/go-cnn/graph"
	"github.com/user/go-cnn/matrix"
	"github.com/user/go-cnn/optimizers"
)

// TestSimpleCNNComparison tests CNN with fixed weights for Python comparison
func TestSimpleCNNComparison(t *testing.T) {
	fmt.Println("=== Go CNN Test with Fixed Weights ===")
	fmt.Println(repeatString("=", 50))

	// 创建优化器和模型用于演示
	optimizer := optimizers.NewSGD(0.1)
	model := graph.NewModel(optimizer)

	// Input data (same as Python)
	inputData := []float64{

		0.1, 0.2, 0.3, 0.4,
		0.5, 0.6, 0.7, 0.8,
		0.9, 1.0, 0.1, 0.2,
		0.3, 0.4, 0.5, 0.6,
	}

	// Create input tensor: batch_size=1, features=16 (flattened 4x4x1)
	inputMatrix := matrix.NewMatrixFromData(inputData, 1, 16)
	input := graph.Input(inputMatrix, "input")

	fmt.Println("\n1. INPUT LAYER")
	fmt.Printf("   Shape: (%d, %d)\n", input.Value.Rows, input.Value.Cols)
	fmt.Printf("   Data: %v\n", flattenMatrix(input.Value))

	// Conv2D Layer: 1->2 channels, 3x3 kernel, stride=1, padding=0
	// Output: 4x4 -> 2x2 per channel, 2 channels -> 2x2x2 = 8 features
	conv1 := graph.Conv2dWithFixedWeights(input, 2, 3, 1, 0, 4, 4, 1)

	fmt.Println("\n2. CONV2D LAYER")
	fmt.Printf("   Parameters: in_channels=1, out_channels=2, kernel=3x3, stride=1, padding=0\n")
	if convOp, ok := conv1.Op.(*graph.ConvOp); ok {
		weights := convOp.GetWeights()
		bias := convOp.GetBiases()
		fmt.Printf("   Weights shape: (%d, %d)\n", weights.Rows, weights.Cols)
		fmt.Printf("   Weights: %v\n", flattenMatrix(weights))
		fmt.Printf("   Bias: %v\n", flattenMatrix(bias))
	}
	fmt.Printf("   Output shape: (%d, %d)\n", conv1.Value.Rows, conv1.Value.Cols)
	fmt.Printf("   Output: %v\n", flattenMatrix(conv1.Value))

	// ReLU Activation
	relu1 := graph.ReLU(conv1)
	fmt.Println("\n3. RELU ACTIVATION")
	fmt.Printf("   Output shape: (%d, %d)\n", relu1.Value.Rows, relu1.Value.Cols)
	fmt.Printf("   Output: %v\n", flattenMatrix(relu1.Value))

	// MaxPool2D: 2x2 window, stride=1
	// Input: 2x2x2, Output: 1x1x2 = 2 features
	pool1 := graph.MaxPool2d(relu1, 2, 1, 2, 2, 2)
	fmt.Println("\n4. MAXPOOL2D LAYER")
	fmt.Printf("   Parameters: kernel=2x2, stride=1\n")
	fmt.Printf("   Output shape: (%d, %d)\n", pool1.Value.Rows, pool1.Value.Cols)
	fmt.Printf("   Output: %v\n", flattenMatrix(pool1.Value))

	// Reshape/Flatten
	flatten := graph.Reshape(pool1, 1, 2)
	fmt.Println("\n5. FLATTEN LAYER")
	fmt.Printf("   Output shape: (%d, %d)\n", flatten.Value.Rows, flatten.Value.Cols)
	fmt.Printf("   Output: %v\n", flattenMatrix(flatten.Value))

	// Dense Layer: 2->2
	dense1 := graph.DenseWithFixedWeights(flatten, 2)

	fmt.Println("\n6. DENSE LAYER")
	fmt.Printf("   Parameters: in_features=2, out_features=2\n")
	if denseOp, ok := dense1.Op.(*graph.DenseOp); ok {
		weights := denseOp.GetWeights()
		bias := denseOp.GetBiases()
		fmt.Printf("   Weights shape: (%d, %d)\n", weights.Rows, weights.Cols)
		fmt.Printf("   Weights: %v\n", flattenMatrix(weights))
		fmt.Printf("   Bias: %v\n", flattenMatrix(bias))
	}
	fmt.Printf("   Output shape: (%d, %d)\n", dense1.Value.Rows, dense1.Value.Cols)
	fmt.Printf("   Output: %v\n", flattenMatrix(dense1.Value))

	// Loss calculation (target class = 1)
	labels := graph.NewConstant(matrix.NewMatrixFromData([]float64{1}, 1, 1), "label")
	loss := graph.SoftmaxCrossEntropyLoss(dense1, labels, true)

	fmt.Println("\n7. LOSS CALCULATION")
	fmt.Printf("   Target label: 1\n")
	fmt.Printf("   Loss value: %.6f\n", loss.Value.At(0, 0))

	// 收集参数用于优化器演示
	model.SetOutput(dense1)
	model.CollectParameters(dense1)

	// Backward pass
	fmt.Println("\n8. BACKWARD PASS")
	loss.Backward()
	fmt.Println("   Backward pass completed")

	// Print gradients
	fmt.Println("\n9. GRADIENTS")

	// Dense layer gradients
	if denseOp, ok := dense1.Op.(*graph.DenseOp); ok {
		weightGrads := denseOp.GetWeightGradients()
		biasGrads := denseOp.GetBiasGradients()
		fmt.Printf("   Dense weight gradients: %v\n", flattenMatrix(weightGrads))
		fmt.Printf("   Dense bias gradients: %v\n", flattenMatrix(biasGrads))
	}

	// Conv layer gradients
	if convOp, ok := conv1.Op.(*graph.ConvOp); ok {
		weightGrads := convOp.GetWeightGradients()
		biasGrads := convOp.GetBiasGradients()
		fmt.Printf("   Conv weight gradients: %v\n", flattenMatrix(weightGrads))
		fmt.Printf("   Conv bias gradients: %v\n", flattenMatrix(biasGrads))
	}

	// Input gradients
	if input.Gradient != nil {
		fmt.Printf("   Input gradients: %v\n", flattenMatrix(input.Gradient))
	}

	// Optimizer step
	fmt.Println("\n10. OPTIMIZER STEP")

	model.Step()
	fmt.Println("   Parameter update completed")

	// Updated weights
	fmt.Println("\n11. UPDATED WEIGHTS")
	if denseOp, ok := dense1.Op.(*graph.DenseOp); ok {
		weights := denseOp.GetWeights()
		bias := denseOp.GetBiases()
		fmt.Printf("   Dense weights: %v\n", flattenMatrix(weights))
		fmt.Printf("   Dense bias: %v\n", flattenMatrix(bias))
	}

	if convOp, ok := conv1.Op.(*graph.ConvOp); ok {
		weights := convOp.GetWeights()
		bias := convOp.GetBiases()
		fmt.Printf("   Conv weights: %v\n", flattenMatrix(weights))
		fmt.Printf("   Conv bias: %v\n", flattenMatrix(bias))
	}

	fmt.Println("\n" + repeatString("=", 50))
	fmt.Println("=== Test Complete ===")
}

// Helper function to flatten matrix for display
func flattenMatrix(m *matrix.Matrix) []float64 {
	if m == nil {
		return nil
	}
	result := make([]float64, m.Rows*m.Cols)
	idx := 0
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			result[idx] = m.At(i, j)
			idx++
		}
	}
	return result
}

// Helper function to repeat string
func repeatString(s string, count int) string {
	result := ""
	for i := 0; i < count; i++ {
		result += s
	}
	return result
}
