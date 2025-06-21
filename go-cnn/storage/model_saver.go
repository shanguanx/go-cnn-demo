package storage

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"time"

	"github.com/user/go-cnn/graph"
	"github.com/user/go-cnn/matrix"
)

// SerializedMatrix 序列化的矩阵
type SerializedMatrix struct {
	Rows int       `json:"rows"`
	Cols int       `json:"cols"`
	Data []float64 `json:"data"`
}

// LayerParams 单层的参数
type LayerParams struct {
	LayerType string            `json:"layer_type"`
	Weights   *SerializedMatrix `json:"weights,omitempty"`
	Biases    *SerializedMatrix `json:"biases,omitempty"`
}

// NetworkConfig 网络配置
type NetworkConfig struct {
	InputSize  int `json:"input_size"`
	OutputSize int `json:"output_size"`
	BatchSize  int `json:"batch_size"`
	NumLayers  int `json:"num_layers"`
}

// ModelState 模型状态，包含所有可训练参数
type ModelState struct {
	// 模型元信息
	ModelVersion string    `json:"model_version"`
	Architecture string    `json:"architecture"`
	CreatedAt    time.Time `json:"created_at"`

	// 网络结构信息
	NetworkConfig NetworkConfig `json:"network_config"`

	// 参数数据
	Parameters map[string]LayerParams `json:"parameters"`
}

// ModelSaver 模型保存器接口
type ModelSaver interface {
	// 保存模型参数
	SaveModel(model *graph.Model, filepath string) error

	// 加载模型参数
	LoadModel(model *graph.Model, filepath string) error
}

// JSONModelSaver JSON格式保存器
type JSONModelSaver struct{}

// NewJSONModelSaver 创建JSON模型保存器
func NewJSONModelSaver() *JSONModelSaver {
	return &JSONModelSaver{}
}

// SaveModel 保存模型到JSON文件
func (s *JSONModelSaver) SaveModel(model *graph.Model, filepath string) error {
	// 获取模型状态
	modelState := s.extractModelState(model)

	// 序列化为JSON
	jsonData, err := json.MarshalIndent(modelState, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal model state: %w", err)
	}

	// 写入文件
	err = ioutil.WriteFile(filepath, jsonData, 0644)
	if err != nil {
		return fmt.Errorf("failed to write model file: %w", err)
	}

	fmt.Printf("模型已保存到: %s\n", filepath)
	return nil
}

// LoadModel 从JSON文件加载模型
func (s *JSONModelSaver) LoadModel(model *graph.Model, filepath string) error {
	// 读取文件
	jsonData, err := ioutil.ReadFile(filepath)
	if err != nil {
		return fmt.Errorf("failed to read model file: %w", err)
	}

	// 反序列化JSON
	var modelState ModelState
	err = json.Unmarshal(jsonData, &modelState)
	if err != nil {
		return fmt.Errorf("failed to unmarshal model state: %w", err)
	}

	// 应用模型状态
	err = s.applyModelState(model, modelState)
	if err != nil {
		return fmt.Errorf("failed to apply model state: %w", err)
	}

	fmt.Printf("模型已从 %s 加载\n", filepath)
	return nil
}

// extractModelState 从模型提取状态
func (s *JSONModelSaver) extractModelState(model *graph.Model) ModelState {
	parameters := make(map[string]LayerParams)

	// 遍历所有可训练层
	for i, layerOp := range model.GetLayerOps() {
		layerName := fmt.Sprintf("layer_%d", i)

		var layerParams LayerParams

		// 根据层类型提取参数
		switch op := layerOp.(type) {
		case *graph.ConvOp:
			layerParams = LayerParams{
				LayerType: "conv",
				Weights:   SerializeMatrix(op.GetWeights()),
				Biases:    SerializeMatrix(op.GetBiases()),
			}
		case *graph.DenseOp:
			layerParams = LayerParams{
				LayerType: "dense",
				Weights:   SerializeMatrix(op.GetWeights()),
				Biases:    SerializeMatrix(op.GetBiases()),
			}
		default:
			// 跳过不支持的层类型
			continue
		}

		parameters[layerName] = layerParams
	}

	return ModelState{
		ModelVersion: "1.0",
		Architecture: "LeNet-5",
		CreatedAt:    time.Now(),
		NetworkConfig: NetworkConfig{
			InputSize:  784,
			OutputSize: 10,
			BatchSize:  32,
			NumLayers:  len(parameters),
		},
		Parameters: parameters,
	}
}

// applyModelState 应用模型状态到模型
func (s *JSONModelSaver) applyModelState(model *graph.Model, state ModelState) error {
	layerOps := model.GetLayerOps()

	// 检查层数是否匹配
	if len(layerOps) != state.NetworkConfig.NumLayers {
		return fmt.Errorf("layer count mismatch: model has %d layers, saved state has %d layers",
			len(layerOps), state.NetworkConfig.NumLayers)
	}

	// 应用每层的参数
	for i, layerOp := range layerOps {
		layerName := fmt.Sprintf("layer_%d", i)

		if params, exists := state.Parameters[layerName]; exists {
			err := s.applyLayerParams(layerOp, params)
			if err != nil {
				return fmt.Errorf("failed to apply parameters for %s: %w", layerName, err)
			}
		} else {
			return fmt.Errorf("missing parameters for %s", layerName)
		}
	}

	return nil
}

// applyLayerParams 应用层参数
func (s *JSONModelSaver) applyLayerParams(layerOp graph.LayerOperation, params LayerParams) error {
	switch op := layerOp.(type) {
	case *graph.ConvOp:
		if params.LayerType != "conv" {
			return fmt.Errorf("layer type mismatch: expected conv, got %s", params.LayerType)
		}

		if params.Weights != nil {
			weights := DeserializeMatrix(params.Weights)
			err := op.SetWeights(weights)
			if err != nil {
				return fmt.Errorf("failed to set conv weights: %w", err)
			}
		}

		if params.Biases != nil {
			biases := DeserializeMatrix(params.Biases)
			err := op.SetBiases(biases)
			if err != nil {
				return fmt.Errorf("failed to set conv biases: %w", err)
			}
		}

	case *graph.DenseOp:
		if params.LayerType != "dense" {
			return fmt.Errorf("layer type mismatch: expected dense, got %s", params.LayerType)
		}

		if params.Weights != nil {
			weights := DeserializeMatrix(params.Weights)
			err := op.SetWeights(weights)
			if err != nil {
				return fmt.Errorf("failed to set dense weights: %w", err)
			}
		}

		if params.Biases != nil {
			biases := DeserializeMatrix(params.Biases)
			err := op.SetBiases(biases)
			if err != nil {
				return fmt.Errorf("failed to set dense biases: %w", err)
			}
		}

	default:
		return fmt.Errorf("unsupported layer type")
	}

	return nil
}

// SerializeMatrix 序列化矩阵
func SerializeMatrix(m *matrix.Matrix) *SerializedMatrix {
	if m == nil {
		return nil
	}

	return &SerializedMatrix{
		Rows: m.Rows,
		Cols: m.Cols,
		Data: append([]float64(nil), m.Data...), // 复制数据
	}
}

// DeserializeMatrix 反序列化矩阵
func DeserializeMatrix(sm *SerializedMatrix) *matrix.Matrix {
	if sm == nil {
		return nil
	}

	return matrix.NewMatrixFromData(sm.Data, sm.Rows, sm.Cols)
}
