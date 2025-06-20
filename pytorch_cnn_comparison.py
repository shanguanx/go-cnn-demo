# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def test_simple_cnn():
    print("=== 简单CNN测试 (PyTorch版本) ===")
    
    # 设置随机种子以确保可重现性
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 创建输入数据：4x4图像，1个通道
    # 与Go代码相同的输入数据
    input_data = [
        0.1, 0.2, 0.3, 0.4,
        0.5, 0.6, 0.7, 0.8,
        0.9, 1.0, 0.1, 0.2,
        0.3, 0.4, 0.5, 0.6,
    ]
    
    # 重塑为 (batch_size=1, channels=1, height=4, width=4)
    input_tensor = torch.tensor(input_data, dtype=torch.float32).reshape(1, 1, 4, 4)
    input_tensor.requires_grad_(True)
    
    print(f"输入形状: {input_tensor.shape}")
    print("输入数据:")
    print(input_tensor)
    print(f"输入数据扁平化 (与Go对比): {input_tensor.flatten()}")
    
    # 1. 卷积层：1输入通道 -> 2输出通道，3x3卷积核，stride=1，padding=0
    conv1 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0, bias=True)
    
    # 打印卷积层参数
    print("\n=== 卷积层参数 ===")
    print(f"卷积层权重形状: {conv1.weight.shape}")
    print("卷积层权重:")
    print(conv1.weight.data)
    print(f"卷积层偏置形状: {conv1.bias.shape}")
    print("卷积层偏置:")
    print(conv1.bias.data)
    
    # 前向传播
    conv1_output = conv1(input_tensor)
    print(f"\n卷积层1输出形状: {conv1_output.shape}")
    print("卷积层1输出:")
    print(conv1_output)
    print(f"卷积层1输出扁平化 (与Go对比): {conv1_output.flatten()}")
    
    # 2. ReLU激活
    relu1_output = F.relu(conv1_output)
    print(f"\nReLU1输出形状: {relu1_output.shape}")
    print("ReLU1输出:")
    print(relu1_output)
    print(f"ReLU1输出扁平化 (与Go对比): {relu1_output.flatten()}")
    
    # 扁平化以准备全连接层
    flattened = relu1_output.flatten(start_dim=1)  # 保持batch维度
    print(f"\n扁平化后形状: {flattened.shape}")
    print("扁平化后数据:")
    print(flattened)
    
    # 3. 全连接层：8 -> 2（分类为2类）
    dense1 = nn.Linear(in_features=8, out_features=2)
    
    # 打印全连接层参数
    print("\n=== 全连接层参数 ===")
    print(f"全连接层权重形状: {dense1.weight.shape}")
    print("全连接层权重:")
    print(dense1.weight.data)
    print(f"全连接层偏置形状: {dense1.bias.shape}")
    print("全连接层偏置:")
    print(dense1.bias.data)
    
    dense1_output = dense1(flattened)
    print(f"\n全连接层输出形状: {dense1_output.shape}")
    print("全连接层输出:")
    print(dense1_output)
    
    # 4. 计算损失（假设标签是类别1）
    target = torch.tensor([1], dtype=torch.long)  # 类别1
    loss = F.cross_entropy(dense1_output, target)
    
    print("\n=== 损失计算 ===")
    print(f"损失值: {loss.item()}")
    print(f"目标标签: {target}")
    
    # 应用softmax看概率分布
    probabilities = F.softmax(dense1_output, dim=1)
    print("Softmax概率分布:")
    print(probabilities)
    
    # 5. 反向传播
    print("\n=== 开始反向传播 ===")
    loss.backward()
    print("反向传播完成")
    
    # 6. 检查梯度
    print("\n=== 梯度检查 ===")
    
    # 全连接层梯度
    print("全连接层权重梯度:")
    print(dense1.weight.grad)
    print("全连接层偏置梯度:")
    print(dense1.bias.grad)
    
    # 卷积层梯度
    print("\n卷积层权重梯度:")
    print(conv1.weight.grad)
    print("卷积层偏置梯度:")
    print(conv1.bias.grad)
    
    # 输入梯度
    print("\n输入梯度:")
    print(input_tensor.grad)
    print(f"输入梯度扁平化 (与Go对比): {input_tensor.grad.flatten()}")
    
    print("\n=== 简单CNN测试完成 (PyTorch版本) ===")
    
    return {
        'input': input_tensor,
        'conv1_output': conv1_output,
        'relu1_output': relu1_output,
        'dense1_output': dense1_output,
        'loss': loss,
        'conv1_layer': conv1,
        'dense1_layer': dense1
    }

def test_with_same_weights():
    """使用固定权重进行测试，以便与Go实现精确对比"""
    print("\n" + "="*50)
    print("=== 使用固定权重的测试 (与Go代码相同) ===")
    
    # 创建输入
    input_data = [
        0.1, 0.2, 0.3, 0.4,
        0.5, 0.6, 0.7, 0.8,
        0.9, 1.0, 0.1, 0.2,
        0.3, 0.4, 0.5, 0.6,
    ]
    input_tensor = torch.tensor(input_data, dtype=torch.float32).reshape(1, 1, 4, 4)
    input_tensor.requires_grad_(True)
    
    print(f"输入: {input_tensor.flatten()}")
    
    # 创建卷积层并设置与Go代码相同的固定权重
    conv1 = nn.Conv2d(1, 2, 3, stride=1, padding=0, bias=True)
    
    # Go代码中的固定权重模式：float64(i*10+j)*0.1
    # 卷积层权重矩阵形状：(out_channels=2, in_channels*kernel_size*kernel_size=1*3*3=9)
    # 权重值：
    # 第0输出通道: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    # 第1输出通道: [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8]
    with torch.no_grad():
        # 设置第0个输出通道的权重 (reshape to 3x3)
        conv1.weight[0, 0] = torch.tensor([
            [0.0, 0.1, 0.2],
            [0.3, 0.4, 0.5],
            [0.6, 0.7, 0.8]
        ], dtype=torch.float32)
        
        # 设置第1个输出通道的权重 (reshape to 3x3)
        conv1.weight[1, 0] = torch.tensor([
            [1.0, 1.1, 1.2],
            [1.3, 1.4, 1.5],
            [1.6, 1.7, 1.8]
        ], dtype=torch.float32)
        
        # Go代码中的偏置模式：float64(i)*0.5
        # 第0输出通道偏置: 0*0.5 = 0.0
        # 第1输出通道偏置: 1*0.5 = 0.5
        conv1.bias[0] = 0.0
        conv1.bias[1] = 0.5
    
    print("\n设置的卷积层权重 (与Go代码相同):")
    print("第0输出通道权重:", conv1.weight[0, 0].flatten())
    print("第1输出通道权重:", conv1.weight[1, 0].flatten())
    print("卷积层偏置:", conv1.bias.data)
    
    # 前向传播
    conv1_output = conv1(input_tensor)
    relu1_output = F.relu(conv1_output)
    flattened = relu1_output.flatten(start_dim=1)
    
    print(f"卷积层输出: {conv1_output.flatten()}")
    print(f"ReLU输出: {relu1_output.flatten()}")
    
    # 创建全连接层并设置与Go代码相同的固定权重
    dense1 = nn.Linear(8, 2)
    
    # Go代码中全连接层权重模式：float64(i*cols+j)*0.1
    # 权重矩阵形状：(input_features=8, output_features=2)
    # PyTorch格式是 (out_features, in_features)，所以需要转置
    go_weights = []
    for i in range(8):  # input_features
        row = []
        for j in range(2):  # output_features
            weight_val = (i * 2 + j) * 0.1
            row.append(weight_val)
        go_weights.append(row)
    
    # 转换为PyTorch格式 (out_features, in_features)
    pytorch_weights = []
    for j in range(2):  # output_features
        row = []
        for i in range(8):  # input_features
            row.append(go_weights[i][j])
        pytorch_weights.append(row)
    
    with torch.no_grad():
        dense1.weight.data = torch.tensor(pytorch_weights, dtype=torch.float32)
        
        # Go代码中的偏置模式：float64(j)*0.5
        # 第0输出: 0*0.5 = 0.0
        # 第1输出: 1*0.5 = 0.5
        dense1.bias.data = torch.tensor([0.0, 0.5], dtype=torch.float32)
    
    print("\n设置的全连接层权重 (与Go代码相同):")
    print("全连接层权重:")
    print(dense1.weight.data)
    print("全连接层偏置:", dense1.bias.data)
    
    dense1_output = dense1(flattened)
    
    # 计算损失
    target = torch.tensor([1])
    loss = F.cross_entropy(dense1_output, target)
    
    print(f"\n各层输出:")
    print(f"全连接层输出: {dense1_output.flatten()}")
    print(f"损失: {loss.item()}")
    
    # 反向传播
    loss.backward()
    
    print(f"\n梯度:")
    print(f"输入梯度: {input_tensor.grad.flatten()}")
    print(f"卷积层权重梯度: {conv1.weight.grad.flatten()}")
    print(f"卷积层偏置梯度: {conv1.bias.grad.flatten()}")
    print(f"全连接层权重梯度: {dense1.weight.grad.flatten()}")
    print(f"全连接层偏置梯度: {dense1.bias.grad.flatten()}")
    
    print("=== 固定权重测试完成 ===")
    print("\n与Go代码对比指南:")
    print("1. 卷积层输出应该匹配Go代码中的 'Matrix(1x8)' 输出")
    print("2. 全连接层输出应该匹配Go代码中的 'Matrix(1x2)' 输出")
    print("3. 损失值应该匹配Go代码中的损失值")
    print("4. 各层梯度应该与Go代码中的梯度值匹配")

if __name__ == "__main__":
    # 运行随机权重测试
    result = test_simple_cnn()
    
    # 运行固定权重测试
    test_with_same_weights() 