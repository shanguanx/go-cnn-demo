package tests

import (
	"github.com/user/go-cnn/graph"
	"github.com/user/go-cnn/matrix"
	"github.com/user/go-cnn/optimizers"
)

func TestModelBasicUsage(t *testing.T) {
	// 创建优化器和模型
	optimizer := optimizers.NewSGD(0.01)
	model := graph.NewModel(optimizer)

	// 创建简单的全连接网络
	inputData := matrix.Random(1, 10, -1.0, 1.0)
	input := graph.Input(inputData, "input")
	dense := graph.Dense(input, 5)
	output := graph.ReLU(dense)

	model.SetOutput(output)
	model.CollectParameters(output)

	// 验证参数收集
	if model.GetParameterCount() == 0 {
		t.Error("Expected parameters to be collected")
	}

	// 测试前向传播
	result := output.Value
	if result.Rows != 1 || result.Cols != 5 {
		t.Errorf("Expected output shape 1x5, got %dx%d", result.Rows, result.Cols)
	}

	// 测试训练步骤
	model.ZeroGrad()
	model.Step()

	t.Log("Model test passed")
}
