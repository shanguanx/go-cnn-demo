#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证Go代码卷积层实现的Python程序
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
        rows, cols = matrix.shape
        print(f"形状: ({rows}, {cols})")
        for i in range(rows):
            row_str = " ".join([f"{matrix[i, j]:{format_spec}}" for j in range(cols)])
            print(f"[{i}]: {row_str}")
    elif isinstance(matrix, torch.Tensor):
        if matrix.dim() == 1:
            matrix = matrix.unsqueeze(0)
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

def im2col(input_data, kernel_size, stride, padding):
    """将输入数据转换为列格式（im2col操作）"""
    batch_size, in_channels, height, width = input_data.shape
    
    # 计算输出尺寸
    output_height = (height + 2 * padding - kernel_size) // stride + 1
    output_width = (width + 2 * padding - kernel_size) // stride + 1
    
    # 添加padding
    if padding > 0:
        padded_input = np.pad(input_data, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')
    else:
        padded_input = input_data
    
    # 计算每个输出位置对应的输入窗口
    col_data = []
    for b in range(batch_size):
        for h in range(output_height):
            for w in range(output_width):
                h_start = h * stride
                h_end = h_start + kernel_size
                w_start = w * stride
                w_end = w_start + kernel_size
                
                window = padded_input[b, :, h_start:h_end, w_start:w_end]
                col_data.append(window.flatten())
    
    return np.array(col_data)

def col2im(col_data, input_shape, kernel_size, stride, padding):
    """将列格式数据转换回原始格式（col2im操作）"""
    batch_size, in_channels, height, width = input_shape
    output_height = (height + 2 * padding - kernel_size) // stride + 1
    output_width = (width + 2 * padding - kernel_size) // stride + 1
    
    # 初始化输出
    if padding > 0:
        padded_output = np.zeros((batch_size, in_channels, height + 2 * padding, width + 2 * padding))
    else:
        padded_output = np.zeros((batch_size, in_channels, height, width))
    
    # 将列数据映射回原始位置
    col_idx = 0
    for b in range(batch_size):
        for h in range(output_height):
            for w in range(output_width):
                h_start = h * stride
                h_end = h_start + kernel_size
                w_start = w * stride
                w_end = w_start + kernel_size
                
                window_data = col_data[col_idx].reshape(in_channels, kernel_size, kernel_size)
                padded_output[b, :, h_start:h_end, w_start:w_end] += window_data
                col_idx += 1
    
    # 移除padding
    if padding > 0:
        return padded_output[:, :, padding:-padding, padding:-padding]
    else:
        return padded_output

def forward_convolution(input_data, weights, biases, kernel_size, stride, padding):
    """前向传播"""
    batch_size, in_channels, height, width = input_data.shape
    out_channels = weights.shape[0]
    
    # 计算输出尺寸
    output_height = (height + 2 * padding - kernel_size) // stride + 1
    output_width = (width + 2 * padding - kernel_size) // stride + 1
    
    # im2col操作
    input_cols = im2col(input_data, kernel_size, stride, padding)
    print_matrix("输入列格式 (im2col结果)", input_cols)
    
    # 矩阵乘法：output = weights @ input_cols.T + biases
    output_cols = weights @ input_cols.T + biases
    
    # 重塑为输出格式
    output = output_cols.T.reshape(batch_size, out_channels, output_height, output_width)
    
    return output, input_cols

def backward_convolution(grad_output, input_cols, weights, input_shape, kernel_size, stride, padding):
    """反向传播"""
    batch_size, in_channels, height, width = input_shape
    out_channels = weights.shape[0]
    
    # 计算输出尺寸
    output_height = (height + 2 * padding - kernel_size) // stride + 1
    output_width = (width + 2 * padding - kernel_size) // stride + 1
    
    # 重塑梯度输出为列格式
    grad_output_cols = grad_output.reshape(batch_size, out_channels, -1).transpose(0, 2, 1).reshape(-1, out_channels)
    
    # 计算权重梯度：dW = grad_output_cols.T @ input_cols
    weight_gradients = grad_output_cols.T @ input_cols
    
    # 计算偏置梯度：db = sum(grad_output_cols, axis=0)
    bias_gradients = np.sum(grad_output_cols, axis=0, keepdims=True).T
    
    # 计算输入梯度：dX = weights.T @ grad_output_cols
    grad_input_cols = weights.T @ grad_output_cols.T
    
    # 将输入梯度转换回原始格式
    grad_input = col2im(grad_input_cols.T, input_shape, kernel_size, stride, padding)
    
    return grad_input, weight_gradients, bias_gradients

def main():
    print("=== Python卷积层验证程序 ===")
    print("使用与Go测试相同的参数和输入")
    
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
    weights = create_fixed_weights(in_channels, out_channels, kernel_size)
    biases = create_fixed_biases(out_channels)
    
    print_matrix("权重矩阵", weights)
    print_matrix("偏置矩阵", biases)
    
    # 创建输入数据（与Go测试相同）
    input_values = np.array([0.73, 0.29, 0.85, 0.41, 0.67, 0.93, 0.18, 0.54, 0.76])
    input_data = input_values.reshape(batch_size, in_channels, input_height, input_width)
    
    print_matrix("输入数据", input_data.reshape(batch_size, -1))
    
    # 前向传播
    print("\n=== 前向传播 ===")
    output, input_cols = forward_convolution(input_data, weights, biases, kernel_size, stride, padding)
    
    print_matrix("前向传播输出", output.reshape(batch_size, -1))
    
    # 创建梯度输出（与Go测试相同）
    grad_values = np.array([0.31, 0.89, 0.47, 0.62, 0.15, 0.83, 0.39, 0.71, 0.24])
    grad_output = grad_values.reshape(batch_size, out_channels, input_height, input_width)
    
    print_matrix("梯度输出", grad_output.reshape(batch_size, -1))
    
    # 反向传播
    print("\n=== 反向传播 ===")
    grad_input, weight_gradients, bias_gradients = backward_convolution(
        grad_output, input_cols, weights, input_data.shape, kernel_size, stride, padding
    )
    
    print_matrix("输入梯度", grad_input.reshape(batch_size, -1))
    print_matrix("权重梯度", weight_gradients)
    print_matrix("偏置梯度", bias_gradients)
    
    # 权重更新
    print("\n=== 权重更新 ===")
    learning_rate = 0.01
    
    # 保存原始权重和偏置
    original_weights = weights.copy()
    original_biases = biases.copy()
    
    print_matrix("原始权重", original_weights)
    print_matrix("原始偏置", original_biases)
    
    # 更新权重和偏置
    updated_weights = weights - learning_rate * weight_gradients
    updated_biases = biases - learning_rate * bias_gradients
    
    print_matrix("更新后的权重", updated_weights)
    print_matrix("更新后的偏置", updated_biases)
    
    # 检查权重是否发生变化
    weight_changed = not np.allclose(weights, original_weights, atol=1e-10)
    bias_changed = not np.allclose(biases, original_biases, atol=1e-10)
    
    print(f"\n权重是否发生变化: {weight_changed}")
    print(f"偏置是否发生变化: {bias_changed}")
    
    print("\n=== Python验证完成 ===")

if __name__ == "__main__":
    main() 