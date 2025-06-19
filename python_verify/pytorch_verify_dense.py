import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def print_matrix(name, matrix, is_2d=True):
    """打印矩阵，格式与Go代码保持一致"""
    print(f"=== {name} ===")
    if is_2d:
        if len(matrix.shape) == 1:
            matrix = matrix.unsqueeze(0)  # 添加batch维度
        rows, cols = matrix.shape
        print(f"Matrix({rows}x{cols}):")
        for i in range(rows):
            row_str = "["
            for j in range(cols):
                row_str += f"  {matrix[i, j]:8.4f}"
            row_str += "]"
            print(row_str)
    else:
        print(matrix)
    print()

def main():
    print("PyTorch密集层验证 - 与Go代码对比")
    print("=" * 50)
    
    # 设置随机种子确保可重复性
    torch.manual_seed(42)
    
    # 创建与Go代码相同的参数
    input_features = 4
    output_features = 3
    batch_size = 2
    learning_rate = 0.05
    
    # 创建线性层
    linear_layer = nn.Linear(input_features, output_features, bias=True)
    
    # 手动设置权重矩阵 (4x3) - 与Go代码TestDenseLayerWeightUpdate完全一致
    # Go代码中的权重设置：
    # weights.Set(0, 0, 1.5)  weights.Set(0, 1, -0.8)  weights.Set(0, 2, 2.3)
    # weights.Set(1, 0, -1.2) weights.Set(1, 1, 0.9)   weights.Set(1, 2, -0.5)
    # weights.Set(2, 0, 0.7)  weights.Set(2, 1, -1.8)  weights.Set(2, 2, 1.1)
    # weights.Set(3, 0, -0.3) weights.Set(3, 1, 2.1)   weights.Set(3, 2, -0.9)
    
    # 注意：PyTorch的Linear层权重是(output_features, input_features)格式
    # 所以需要转置我们的权重矩阵
    weights = torch.tensor([
        [1.5, -0.8, 2.3],      # 第0行：weights[0,0], weights[0,1], weights[0,2]
        [-1.2, 0.9, -0.5],     # 第1行：weights[1,0], weights[1,1], weights[1,2]
        [0.7, -1.8, 1.1],      # 第2行：weights[2,0], weights[2,1], weights[2,2]
        [-0.3, 2.1, -0.9]      # 第3行：weights[3,0], weights[3,1], weights[3,2]
    ], dtype=torch.float32)
    
    # PyTorch需要转置权重矩阵
    weights_torch = weights.t()  # 转置为(3, 4)格式
    
    # 手动设置偏置向量 (3,) - 与Go代码完全一致
    # Go代码中的偏置设置：
    # biases.Set(0, 0, 0.25) biases.Set(0, 1, -0.75) biases.Set(0, 2, 1.5)
    biases = torch.tensor([0.25, -0.75, 1.5], dtype=torch.float32)
    
    # 设置权重和偏置
    with torch.no_grad():
        linear_layer.weight.copy_(weights_torch)
        linear_layer.bias.copy_(biases)
    
    print_matrix("初始权重矩阵 (4x3)", linear_layer.weight)
    print_matrix("初始偏置向量 (1x3)", linear_layer.bias)
    
    # 创建输入数据 (2x4) - 与Go代码完全一致
    # Go代码中的输入设置：
    # input := matrix.NewMatrixFromData([]float64{
    #     1.0, 2.0, 3.0, 4.0, // 第一个样本
    #     0.5, 1.5, 2.5, 3.5, // 第二个样本
    # }, 2, 4)
    input_data = torch.tensor([
        [1.0, 2.0, 3.0, 4.0],    # 第一个样本
        [0.5, 1.5, 2.5, 3.5]     # 第二个样本
    ], dtype=torch.float32)
    
    print_matrix("输入矩阵 (2x4)", input_data)
    
    # 前向传播
    print("=== 前向传播 ===")
    output = linear_layer(input_data)
    print_matrix("前向传播输出 (2x3)", output)
    
    # 创建输出梯度 (2x3) - 与Go代码完全一致
    # Go代码中的梯度设置：
    # gradOutput := matrix.NewMatrixFromData([]float64{
    #     0.1, 0.2, 0.3, // 第一个样本的梯度
    #     0.4, 0.5, 0.6, // 第二个样本的梯度
    # }, 2, 3)
    grad_output = torch.tensor([
        [0.1, 0.2, 0.3],    # 第一个样本的梯度
        [0.4, 0.5, 0.6]     # 第二个样本的梯度
    ], dtype=torch.float32)
    
    print("=== 反向传播 ===")
    print_matrix("输出梯度矩阵 (2x3)", grad_output)
    
    # 清零梯度
    linear_layer.zero_grad()
    
    # 反向传播
    output.backward(grad_output)
    
    # 获取梯度
    weight_grads = linear_layer.weight.grad.clone()
    bias_grads = linear_layer.bias.grad.clone()
    
    # 计算输入梯度 - 与Go代码保持一致
    # 在PyTorch中，我们需要手动计算输入梯度
    # grad_input = grad_output @ linear_layer.weight
    grad_input = grad_output @ linear_layer.weight
    
    print_matrix("输入梯度矩阵 (2x4)", grad_input)
    print_matrix("计算出的权重梯度 (4x3)", weight_grads)
    print_matrix("计算出的偏置梯度 (1x3)", bias_grads.unsqueeze(0))
    
    # 权重更新 - 与Go代码完全一致
    # Go代码中的学习率：learningRate := 0.05
    print("=== 权重更新 ===")
    print(f"学习率: {learning_rate}")
    
    print("更新前权重矩阵:")
    print_matrix("更新前权重", linear_layer.weight)
    print("更新前偏置向量:")
    print_matrix("更新前偏置", linear_layer.bias)
    
    # 手动更新权重和偏置
    with torch.no_grad():
        linear_layer.weight -= learning_rate * weight_grads
        linear_layer.bias -= learning_rate * bias_grads
    
    print("更新后权重矩阵:")
    print_matrix("更新后权重", linear_layer.weight)
    print("更新后偏置向量:")
    print_matrix("更新后偏置", linear_layer.bias)
    
    # 梯度清零验证 - 与Go代码完全一致
    print("=== 梯度清零验证 ===")
    linear_layer.zero_grad()
    
    print("梯度清零后权重梯度:")
    if linear_layer.weight.grad is not None:
        print_matrix("清零后权重梯度", linear_layer.weight.grad)
    else:
        print("清零后权重梯度: None")
    
    print("梯度清零后偏置梯度:")
    if linear_layer.bias.grad is not None:
        print_matrix("清零后偏置梯度", linear_layer.bias.grad.unsqueeze(0))
    else:
        print("清零后偏置梯度: None")
    
    # 检查梯度是否被清零
    weight_grad_cleared = linear_layer.weight.grad is None or torch.all(linear_layer.weight.grad == 0)
    bias_grad_cleared = linear_layer.bias.grad is None or torch.all(linear_layer.bias.grad == 0)
    
    print(f"权重梯度是否清零: {weight_grad_cleared}")
    print(f"偏置梯度是否清零: {bias_grad_cleared}")
    
    print("\n" + "=" * 50)
    print("PyTorch密集层验证完成")
    print("与Go代码TestDenseLayerWeightUpdate测试参数完全一致")
    print("=" * 50)

if __name__ == "__main__":
    main() 