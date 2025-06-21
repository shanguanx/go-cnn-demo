package data

import (
	"fmt"

	"github.com/user/go-cnn/matrix"
)

// DataLoader 数据加载器，用于批量加载数据
type DataLoader struct {
	dataset    Dataset
	batchSize  int
	currentIdx int // 当前读取位置
}

// NewDataLoader 创建新的数据加载器
func NewDataLoader(dataset Dataset, batchSize int) *DataLoader {
	if batchSize <= 0 {
		batchSize = 1
	}

	return &DataLoader{
		dataset:    dataset,
		batchSize:  batchSize,
		currentIdx: 0,
	}
}

// Next 获取下一批数据
// 返回: data - 批量图像数据 [batch_size, 784]
//
//	labels - 批量标签数据 [batch_size, 1]
//	hasMore - 是否还有更多数据
func (dl *DataLoader) Next() (data, labels *matrix.Matrix, hasMore bool) {
	// 检查是否还有数据
	if dl.currentIdx >= dl.dataset.Len() {
		return nil, nil, false
	}

	// 计算实际批大小（最后一批可能不足batchSize）
	remainingSamples := dl.dataset.Len() - dl.currentIdx
	actualBatchSize := dl.batchSize
	if remainingSamples < dl.batchSize {
		actualBatchSize = remainingSamples
	}

	// 创建批量数据矩阵
	data = matrix.NewMatrix(actualBatchSize, 784)
	labels = matrix.NewMatrix(actualBatchSize, 1)

	// 填充批量数据
	for i := 0; i < actualBatchSize; i++ {
		sampleData, sampleLabel, err := dl.dataset.GetItem(dl.currentIdx + i)
		if err != nil {
			// 处理错误，这里简单地跳过
			fmt.Printf("Error loading sample %d: %v\n", dl.currentIdx+i, err)
			continue
		}

		// 复制数据到批量矩阵
		for j := 0; j < 784; j++ {
			data.Set(i, j, sampleData.At(0, j))
		}
		labels.Set(i, 0, sampleLabel.At(0, 0))
	}

	// 更新当前索引
	dl.currentIdx += actualBatchSize

	// 检查是否还有更多数据
	hasMore = dl.currentIdx < dl.dataset.Len()

	return data, labels, true
}

// Reset 重置数据加载器到开始位置
func (dl *DataLoader) Reset() {
	dl.currentIdx = 0
}

// NumBatches 返回总批次数
func (dl *DataLoader) NumBatches() int {
	numSamples := dl.dataset.Len()
	numBatches := numSamples / dl.batchSize
	if numSamples%dl.batchSize != 0 {
		numBatches++
	}
	return numBatches
}

// CurrentBatch 返回当前批次索引（从0开始）
func (dl *DataLoader) CurrentBatch() int {
	if dl.currentIdx == 0 {
		return 0
	}
	return (dl.currentIdx - 1) / dl.batchSize
}
