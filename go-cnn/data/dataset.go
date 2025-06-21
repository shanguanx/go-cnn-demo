package data

import (
	"encoding/csv"
	"fmt"
	"os"
	"strconv"

	"github.com/user/go-cnn/matrix"
)

// Dataset 定义数据集接口
type Dataset interface {
	// Len 返回数据集的样本数量
	Len() int

	// GetItem 获取指定索引的样本
	// 返回: data - 图像数据, label - 标签数据, err - 错误信息
	GetItem(idx int) (data, label *matrix.Matrix, err error)
}

// MNISTDataset MNIST数据集实现
type MNISTDataset struct {
	images     [][]float64 // [num_samples][784]
	labels     []int       // [num_samples]
	numSamples int
	isTraining bool // 是否是训练集（有标签）
}

// LoadMNISTFromCSV 从CSV文件加载MNIST数据
// filepath: CSV文件路径
// isTraining: true表示训练集（有标签），false表示测试集（无标签）
func LoadMNISTFromCSV(filepath string, isTraining bool) (*MNISTDataset, error) {
	// 打开文件
	file, err := os.Open(filepath)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %w", err)
	}
	defer file.Close()

	// 创建CSV读取器
	reader := csv.NewReader(file)

	// 读取所有记录
	records, err := reader.ReadAll()
	if err != nil {
		return nil, fmt.Errorf("failed to read CSV: %w", err)
	}

	// 跳过表头（如果有的话）
	startIdx := 0
	if len(records) > 0 && records[0][0] == "label" {
		startIdx = 1
	}

	numSamples := len(records) - startIdx
	if numSamples == 0 {
		return nil, fmt.Errorf("no data found in CSV file")
	}

	// 初始化数据集
	dataset := &MNISTDataset{
		images:     make([][]float64, numSamples),
		labels:     make([]int, numSamples),
		numSamples: numSamples,
		isTraining: isTraining,
	}

	// 解析数据
	for i := 0; i < numSamples; i++ {
		record := records[i+startIdx]

		if isTraining {
			// 训练集：第一列是标签
			if len(record) != 785 {
				return nil, fmt.Errorf("invalid record length at row %d: expected 785, got %d", i+startIdx, len(record))
			}

			// 解析标签
			label, err := strconv.Atoi(record[0])
			if err != nil {
				return nil, fmt.Errorf("failed to parse label at row %d: %w", i+startIdx, err)
			}
			dataset.labels[i] = label

			// 解析像素值
			dataset.images[i] = make([]float64, 784)
			for j := 0; j < 784; j++ {
				pixel, err := strconv.ParseFloat(record[j+1], 64)
				if err != nil {
					return nil, fmt.Errorf("failed to parse pixel at row %d, col %d: %w", i+startIdx, j+1, err)
				}
				// 归一化到[0,1]
				dataset.images[i][j] = pixel / 255.0
			}
		} else {
			// 测试集：没有标签列
			if len(record) != 784 {
				return nil, fmt.Errorf("invalid record length at row %d: expected 784, got %d", i+startIdx, len(record))
			}

			// 解析像素值
			dataset.images[i] = make([]float64, 784)
			for j := 0; j < 784; j++ {
				pixel, err := strconv.ParseFloat(record[j], 64)
				if err != nil {
					return nil, fmt.Errorf("failed to parse pixel at row %d, col %d: %w", i+startIdx, j, err)
				}
				// 归一化到[0,1]
				dataset.images[i][j] = pixel / 255.0
			}
			// 测试集标签设置为-1
			dataset.labels[i] = -1
		}
	}

	return dataset, nil
}

// Len 实现Dataset接口，返回数据集大小
func (d *MNISTDataset) Len() int {
	return d.numSamples
}

// GetItem 实现Dataset接口，获取指定索引的样本
func (d *MNISTDataset) GetItem(idx int) (data, label *matrix.Matrix, err error) {
	if idx < 0 || idx >= d.numSamples {
		return nil, nil, fmt.Errorf("index out of range: %d", idx)
	}

	// 创建数据矩阵 [1, 784]
	data = matrix.NewMatrix(1, 784)
	for j := 0; j < 784; j++ {
		data.Set(0, j, d.images[idx][j])
	}

	// 创建标签矩阵 [1, 1]
	label = matrix.NewMatrix(1, 1)
	label.Set(0, 0, float64(d.labels[idx]))

	return data, label, nil
}
