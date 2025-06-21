# Go-CNN 数据加载模块设计（最小化实现）

## 概述
为Go-CNN项目提供最简单的MNIST数据加载功能，仅支持从Kaggle CSV格式文件加载数据。

## 核心组件

### 1. Dataset 接口
```go
type Dataset interface {
    // 获取数据集大小
    Len() int
    
    // 获取指定索引的样本
    GetItem(idx int) (data, label *matrix.Matrix, err error)
}
```

### 2. MNISTDataset 实现
```go
type MNISTDataset struct {
    // 数据存储
    images [][]float64  // [num_samples][784]
    labels []int        // [num_samples]
    
    // 数据集信息
    numSamples int
}
```

### 3. DataLoader 批处理器
```go
type DataLoader struct {
    dataset    Dataset
    batchSize  int
    
    // 内部状态
    currentIdx int      // 当前批次位置
}
```

## API 设计

### 数据集加载
```go
// 从CSV文件加载MNIST数据（Kaggle格式）
// isTraining: true表示训练集（有标签），false表示测试集（无标签）
func LoadMNISTFromCSV(filepath string, isTraining bool) (*MNISTDataset, error)
```

### 数据加载器
```go
// 创建数据加载器
func NewDataLoader(dataset Dataset, batchSize int) *DataLoader

// 获取下一批数据
func (dl *DataLoader) Next() (data, labels *matrix.Matrix, hasMore bool)

// 重置到开始位置
func (dl *DataLoader) Reset()
```

## 使用示例

```go
// 1. 加载训练数据
trainDataset, err := LoadMNISTFromCSV("train.csv", true)
if err != nil {
    log.Fatal(err)
}

// 2. 创建数据加载器
trainLoader := NewDataLoader(trainDataset, batchSize=32)

// 3. 训练循环
for epoch := 0; epoch < numEpochs; epoch++ {
    trainLoader.Reset()
    
    for {
        batchData, batchLabels, hasMore := trainLoader.Next()
        if !hasMore {
            break
        }
        
        // 使用batchData和batchLabels进行训练
        // ...
    }
}
```

## 实现细节

### CSV文件格式
- **训练集(train.csv)**: 
  - 第1列：标签(0-9)
  - 第2-785列：像素值(0-255)
  
- **测试集(test.csv)**:
  - 第1-784列：像素值(0-255)
  - 无标签列

### 数据预处理
- 像素值归一化：将0-255的整数值转换为0-1的浮点数（除以255.0）
- 数据格式：展平的784维向量

### 批处理返回格式
- data: `[batch_size, 784]` 的矩阵
- labels: `[batch_size, 1]` 的矩阵（one-hot编码在模型中处理）

## 文件结构
```
data/
├── dataset.go      # Dataset接口和MNISTDataset实现
└── loader.go       # DataLoader实现
```

## 错误处理
- 文件不存在
- CSV格式错误
- 内存分配失败

## 实现要点
1. 使用encoding/csv包读取CSV文件
2. 一次性加载所有数据到内存（MNIST数据集较小）
3. 批处理时创建新的矩阵返回
4. 简单的顺序读取，不实现shuffle功能