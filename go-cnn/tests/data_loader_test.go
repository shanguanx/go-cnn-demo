package tests

import (
	"testing"

	"github.com/user/go-cnn/data"
)

// TestDataLoaderWithRealData 测试使用真实MNIST数据（如果存在）
func TestDataLoaderWithRealData(t *testing.T) {
	// 尝试加载真实的train.csv文件
	dataset, err := data.LoadMNISTFromCSV("../../train.csv", true)
	if err != nil {
		t.Skipf("Skipping real data test: %v", err)
		return
	}

	t.Logf("Loaded MNIST dataset with %d samples", dataset.Len())

	// 创建小批量加载器
	loader := data.NewDataLoader(dataset, 32)
	t.Logf("Created data loader with %d batches", loader.NumBatches())

	// 测试加载第一批
	batchData, batchLabels, hasMore := loader.Next()
	if !hasMore {
		t.Error("Expected to have data in first batch")
	}

	t.Logf("First batch: data shape [%d, %d], labels shape [%d, %d]",
		batchData.Rows, batchData.Cols, batchLabels.Rows, batchLabels.Cols)

	// 验证数据范围
	for i := 0; i < batchData.Rows; i++ {
		for j := 0; j < batchData.Cols; j++ {
			if batchData.At(i, j) < 0 || batchData.At(i, j) > 1 {
				t.Errorf("Pixel value out of range [0,1]: %f at [%d,%d]", batchData.At(i, j), i, j)
			}
		}
	}

	// 验证标签范围
	for i := 0; i < batchLabels.Rows; i++ {
		label := batchLabels.At(i, 0)
		if label < 0 || label > 9 {
			t.Errorf("Label out of range [0,9]: %f at row %d", label, i)
		}
	}
}
