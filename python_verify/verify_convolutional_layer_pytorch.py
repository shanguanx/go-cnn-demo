#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用PyTorch卷积层验证Go代码实现的Python程序
使用与Go测试相同的输入和参数
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def print_matrix(name, matrix, format_spec=".6f"):
    """打印矩阵，格式与Go代码类似"""
    print(f"\n{name}:")
    if isinstance(matrix, np.ndarray):
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        elif matrix.ndim > 2:
            # 对于高维数组，展平为2D
            original_shape = matrix.shape
            matrix = matrix.reshape(matrix.shape[0], -1)
            print(f"原始形状: {original_shape}")
        rows, cols = matrix.shape
        print(f"形状: ({rows}, {cols})")
        for i in range(rows):
            row_str = " ".join([f"{matrix[i, j]:{format_spec}}" for j in range(cols)])
            print(f"[{i}]: {row_str}")
    elif isinstance(matrix, torch.Tensor):
        if matrix.dim() == 1:
            matrix = matrix.unsqueeze(0)
        elif matrix.dim() > 2:
            # 对于高维张量，展平为2D
            original_shape = matrix.shape
            matrix = matrix.reshape(matrix.shape[0], -1)
            print(f"原始形状: {original_shape}")
        
        rows, cols = matrix.shape
        print(f"形状: ({rows}, {cols})")
        for i in range(rows):
            row_str = " ".join([f"{matrix[i, j].item():{format_spec}}" for j in range(cols)])
            print(f"[{i}]: {row_str}")

def create_fixed_weights(in_channels, out_channels, kernel_size):
    """创建与Go代码相同的固定权重"""
    weight_cols = in_channels * kernel_size * kernel_size
    weights = np.zeros((out_channels, weight_cols))
    
    # 使用与Go代码相同的初始化模式：float64(i*10+j)*0.1
    for i in range(out_channels):
        for j in range(weight_cols):
            weights[i, j] = (i * 10 + j) * 0.1
    
    return weights

def create_fixed_biases(out_channels):
    """创建与Go代码相同的固定偏置"""
    biases = np.zeros((out_channels, 1))
    
    # 使用与Go代码相同的初始化模式：float64(i)*0.5
    for i in range(out_channels):
        biases[i, 0] = i * 0.5
    
    return biases

def numpy_to_torch_conv_weights(weights, in_channels, out_channels, kernel_size):
    """将numpy权重转换为PyTorch卷积层权重格式"""
    # PyTorch卷积层权重格式: (out_channels, in_channels, kernel_height, kernel_width)
    torch_weights = weights.reshape(out_channels, in_channels, kernel_size, kernel_size)
    return torch.tensor(torch_weights, dtype=torch.float64)

def numpy_to_torch_conv_biases(biases):
    """将numpy偏置转换为PyTorch卷积层偏置格式"""
    # PyTorch卷积层偏置格式: (out_channels,)
    torch_biases = biases.flatten()
    return torch.tensor(torch_biases, dtype=torch.float64)

def forward_convolution_pytorch(input_data, weights, biases, kernel_size, stride, padding):
    """使用PyTorch卷积层进行前向传播"""
    # 转换为PyTorch张量
    input_tensor = torch.tensor(input_data, dtype=torch.float64, requires_grad=True)
    
    # 创建卷积层并设置权重
    conv_layer = nn.Conv2d(
        in_channels=input_data.shape[1],
        out_channels=weights.shape[0],
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=True,
        dtype=torch.float64
    )
    
    # 设置权重和偏置
    conv_layer.weight.data = weights
    conv_layer.bias.data = biases
    
    # 前向传播
    output = conv_layer(input_tensor)
    
    return output, input_tensor, conv_layer

def backward_convolution_pytorch(grad_output, input_tensor, conv_layer):
    """使用PyTorch自动微分进行反向传播"""
    # 设置梯度输出
    grad_output_tensor = torch.tensor(grad_output, dtype=torch.float64)
    
    # 反向传播
    output = conv_layer(input_tensor)
    output.backward(grad_output_tensor)
    
    # 获取梯度
    grad_input = input_tensor.grad
    weight_gradients = conv_layer.weight.grad
    bias_gradients = conv_layer.bias.grad
    
    return grad_input, weight_gradients, bias_gradients

def main():
    print("=== PyTorch卷积层验证程序 ===")
    print("使用PyTorch内置卷积层实现，与Go测试相同的参数和输入")
    
    # 参数设置（与Go测试相同）
    in_channels = 1
    out_channels = 1
    kernel_size = 3
    stride = 1
    padding = 1
    input_height = 3
    input_width = 3
    batch_size = 1
    
    print(f"\n卷积层参数:")
    print(f"输入通道数: {in_channels}")
    print(f"输出通道数: {out_channels}")
    print(f"卷积核大小: {kernel_size}")
    print(f"步长: {stride}")
    print(f"填充: {padding}")
    print(f"输入尺寸: {input_height}x{input_width}")
    print(f"批量大小: {batch_size}")
    
    # 创建固定权重和偏置（与Go代码相同）
    numpy_weights = create_fixed_weights(in_channels, out_channels, kernel_size)
    numpy_biases = create_fixed_biases(out_channels)
    
    # 转换为PyTorch格式
    torch_weights = numpy_to_torch_conv_weights(numpy_weights, in_channels, out_channels, kernel_size)
    torch_biases = numpy_to_torch_conv_biases(numpy_biases)
    
    print_matrix("权重矩阵 (numpy)", numpy_weights)
    print_matrix("权重张量 (PyTorch)", torch_weights)
    print_matrix("偏置矩阵 (numpy)", numpy_biases)
    print_matrix("偏置张量 (PyTorch)", torch_biases)
    
    # 创建输入数据（与Go测试相同）
    input_values = np.array([0.73, 0.29, 0.85, 0.41, 0.67, 0.93, 0.18, 0.54, 0.76])
    input_data = input_values.reshape(batch_size, in_channels, input_height, input_width)
    
    print_matrix("输入数据", input_data.reshape(batch_size, -1))
    
    # 前向传播
    print("\n=== 前向传播 (PyTorch) ===")
    output, input_tensor, conv_layer = forward_convolution_pytorch(
        input_data, torch_weights, torch_biases, kernel_size, stride, padding
    )
    
    print_matrix("前向传播输出", output.detach().numpy().reshape(batch_size, -1))
    
    # 创建梯度输出（与Go测试相同）
    grad_values = np.array([0.31, 0.89, 0.47, 0.62, 0.15, 0.83, 0.39, 0.71, 0.24])
    grad_output = grad_values.reshape(batch_size, out_channels, input_height, input_width)
    
    print_matrix("梯度输出", grad_output.reshape(batch_size, -1))
    
    # 反向传播
    print("\n=== 反向传播 (PyTorch) ===")
    grad_input, weight_gradients, bias_gradients = backward_convolution_pytorch(
        grad_output, input_tensor, conv_layer
    )
    
    print_matrix("输入梯度", grad_input.detach().numpy().reshape(batch_size, -1))
    print_matrix("权重梯度", weight_gradients.detach().numpy())
    print_matrix("偏置梯度", bias_gradients.detach().numpy().reshape(-1, 1))
    
    # 权重更新
    print("\n=== 权重更新 ===")
    learning_rate = 0.01
    
    # 保存原始权重和偏置
    original_weights = torch_weights.clone()
    original_biases = torch_biases.clone()
    
    print_matrix("原始权重", original_weights.detach().numpy())
    print_matrix("原始偏置", original_biases.detach().numpy().reshape(-1, 1))
    
    # 更新权重和偏置
    updated_weights = torch_weights - learning_rate * weight_gradients
    updated_biases = torch_biases - learning_rate * bias_gradients
    
    print_matrix("更新后的权重", updated_weights.detach().numpy())
    print_matrix("更新后的偏置", updated_biases.detach().numpy().reshape(-1, 1))
    
    # 检查权重是否发生变化
    weight_changed = not torch.allclose(torch_weights, original_weights, atol=1e-10)
    bias_changed = not torch.allclose(torch_biases, original_biases, atol=1e-10)
    
    print(f"\n权重是否发生变化: {weight_changed}")
    print(f"偏置是否发生变化: {bias_changed}")
    
    print("\n=== PyTorch验证完成 ===")

if __name__ == "__main__":
    main() 