#!/usr/bin/env python3
"""
Python代码验证Go卷积层前向传播结果
对拍测试用例：2输入通道，3输出通道，3x3卷积核，步长1，填充1
使用固定权重进行精确对拍
"""

import numpy as np
import torch
import torch.nn as nn

def create_fixed_weights(in_channels, out_channels, kernel_size):
    """创建与Go代码相同的固定权重"""
    # 权重矩阵形状：(out_channels, in_channels * kernel_size * kernel_size)
    weight_cols = in_channels * kernel_size * kernel_size
    weights = np.zeros((out_channels, weight_cols), dtype=np.float32)
    
    # 使用与Go代码相同的模式：float64(i*10+j)*0.1
    for i in range(out_channels):
        for j in range(weight_cols):
            weights[i, j] = float(i * 10 + j) * 0.1
    
    return weights

def create_fixed_biases(out_channels):
    """创建与Go代码相同的固定偏置"""
    # 偏置向量形状：(out_channels,)
    biases = np.zeros(out_channels, dtype=np.float32)
    
    # 使用与Go代码相同的模式：float64(i)*0.5
    for i in range(out_channels):
        biases[i] = float(i) * 0.5
    
    return biases

def verify_convolution():
    print("=== Python卷积层验证（固定权重）===")
    
    # 卷积层参数
    in_channels = 2
    out_channels = 3
    kernel_size = 3
    stride = 1
    padding = 1
    input_height = 4
    input_width = 4
    
    # 创建输入数据
    # 第一个样本：递增值 (0, 1, 2, ..., 31)
    sample1 = np.arange(32, dtype=np.float32)
    # 第二个样本：递减值 (31, 30, 29, ..., 0)
    sample2 = np.arange(31, -1, -1, dtype=np.float32)
    
    # 重塑为 (batch_size, channels, height, width)
    # 从 (32,) 重塑为 (2, 4, 4) 然后添加通道维度
    sample1_reshaped = sample1.reshape(2, 4, 4)  # (2, 4, 4)
    sample2_reshaped = sample2.reshape(2, 4, 4)  # (2, 4, 4)
    
    # 创建批次输入 (batch_size, channels, height, width)
    input_data = np.stack([sample1_reshaped, sample2_reshaped], axis=0)  # (2, 2, 4, 4)
    
    print("输入数据形状:", input_data.shape)
    print("第一个样本 (通道0):")
    print(input_data[0, 0])
    print("第一个样本 (通道1):")
    print(input_data[0, 1])
    print("第二个样本 (通道0):")
    print(input_data[1, 0])
    print("第二个样本 (通道1):")
    print(input_data[1, 1])
    
    # 创建PyTorch卷积层
    conv_layer = nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=True
    )
    
    # 设置固定权重和偏置（与Go代码相同）
    fixed_weights = create_fixed_weights(in_channels, out_channels, kernel_size)
    fixed_biases = create_fixed_biases(out_channels)
    
    # 重塑权重为PyTorch格式：(out_channels, in_channels, kernel_height, kernel_width)
    weight_torch = fixed_weights.reshape(out_channels, in_channels, kernel_size, kernel_size)
    
    # 设置权重和偏置
    conv_layer.weight.data = torch.from_numpy(weight_torch)
    conv_layer.bias.data = torch.from_numpy(fixed_biases)
    
    print(f"\n卷积层权重形状: {conv_layer.weight.shape}")
    print(f"卷积层偏置形状: {conv_layer.bias.shape}")
    
    print("\n权重矩阵 (展平):")
    print(fixed_weights)
    print("\n偏置向量:")
    print(fixed_biases)
    
    # 转换为PyTorch张量
    input_tensor = torch.from_numpy(input_data)
    
    # 前向传播
    with torch.no_grad():
        output_tensor = conv_layer(input_tensor)
    
    print(f"\n输出形状: {output_tensor.shape}")
    print(f"期望输出形状: (2, 3, 4, 4)")
    
    # 将输出展平为 (batch_size, out_channels * height * width)
    output_flat = output_tensor.reshape(2, -1)
    
    print("\n=== 输出结果 ===")
    print("第一个样本输出:")
    for i, val in enumerate(output_flat[0].numpy()):
        print(f"{val:8.4f}", end="")
        if (i + 1) % 16 == 0:
            print()
    
    print("\n第二个样本输出:")
    for i, val in enumerate(output_flat[1].numpy()):
        print(f"{val:8.4f}", end="")
        if (i + 1) % 16 == 0:
            print()
    
    # 验证两个样本的输出确实不同
    sample1_output = output_flat[0].numpy()
    sample2_output = output_flat[1].numpy()
    
    if np.array_equal(sample1_output, sample2_output):
        print("\n❌ 错误：两个样本的输出完全相同！")
    else:
        print("\n✅ 验证通过：两个样本的输出不同，批量处理正常工作")
        print(f"输出差异统计：")
        print(f"  最大值差异: {np.max(np.abs(sample1_output - sample2_output)):.6f}")
        print(f"  平均差异: {np.mean(np.abs(sample1_output - sample2_output)):.6f}")
    
    return output_flat.numpy()

def compare_with_go_output():
    """比较Python输出与Go输出的差异"""
    print("\n=== 与Go输出比较（固定权重）===")
    
    # Go代码的实际输出（从测试结果中获取）
    go_output_sample1 = [
        119.6000, 179.6000, 191.6000, 126.4000, 188.1000, 279.3000, 294.6000, 192.3000,
        231.3000, 340.5000, 355.8000, 230.7000, 142.4000, 207.2000, 215.6000, 138.0000,
        204.1000, 312.1000, 336.1000, 226.9000, 338.6000, 513.8000, 547.1000, 366.8000,
        429.8000, 647.0000, 680.3000, 453.2000, 290.9000, 435.7000, 456.1000, 302.5000,
        288.6000, 444.6000, 480.6000, 327.4000, 489.1000, 748.3000, 799.6000, 541.3000,
        628.3000, 953.5000, 1004.8000, 675.7000, 439.4000, 664.2000, 696.6000, 467.0000
    ]
    
    go_output_sample2 = [
        140.8000, 192.4000, 180.4000, 109.2000, 146.7000, 195.0000, 179.7000, 105.3000,
        103.5000, 133.8000, 118.5000, 66.9000, 43.6000, 53.2000, 44.8000, 23.2000,
        305.3000, 432.9000, 408.9000, 257.7000, 369.2000, 519.5000, 486.2000, 303.8000,
        278.0000, 386.3000, 353.0000, 217.4000, 144.1000, 197.7000, 177.3000, 107.7000,
        469.8000, 673.4000, 637.4000, 406.2000, 591.7000, 844.0000, 792.7000, 502.3000,
        452.5000, 638.8000, 587.5000, 367.9000, 244.6000, 342.2000, 309.8000, 192.2000
    ]
    
    go_output = np.array([go_output_sample1, go_output_sample2])
    
    print("Go代码输出形状:", go_output.shape)
    print("Go代码第一个样本输出:")
    for i, val in enumerate(go_output[0]):
        print(f"{val:8.4f}", end="")
        if (i + 1) % 16 == 0:
            print()
    
    print("\nGo代码第二个样本输出:")
    for i, val in enumerate(go_output[1]):
        print(f"{val:8.4f}", end="")
        if (i + 1) % 16 == 0:
            print()
    
    # 获取Python输出
    python_output = verify_convolution()
    
    print("\n=== 精确比较结果 ===")
    
    # 检查输出形状是否一致
    if go_output.shape == python_output.shape:
        print("✅ 输出形状一致")
    else:
        print(f"❌ 输出形状不一致: Go {go_output.shape}, Python {python_output.shape}")
    
    # 逐元素比较
    max_diff = np.max(np.abs(go_output - python_output))
    mean_diff = np.mean(np.abs(go_output - python_output))
    
    print(f"\n数值精度比较:")
    print(f"  最大差异: {max_diff:.10f}")
    print(f"  平均差异: {mean_diff:.10f}")
    
    if max_diff < 1e-6:
        print("✅ 完美匹配！Go和Python输出完全一致")
    elif max_diff < 1e-3:
        print("✅ 高度匹配！差异在可接受范围内")
    else:
        print("❌ 存在显著差异，需要检查实现")
    
    # 检查两个样本的输出是否都不同
    go_samples_different = not np.array_equal(go_output[0], go_output[1])
    python_samples_different = not np.array_equal(python_output[0], python_output[1])
    
    print(f"\n批量处理验证:")
    print(f"Go代码两个样本不同: {go_samples_different}")
    print(f"Python代码两个样本不同: {python_samples_different}")
    
    if go_samples_different and python_samples_different:
        print("✅ 两个实现都正确处理了批量数据")
    else:
        print("❌ 至少有一个实现没有正确处理批量数据")
    
    # 显示具体的差异位置（如果有的话）
    if max_diff > 1e-6:
        print(f"\n差异详情:")
        diff_positions = np.where(np.abs(go_output - python_output) > 1e-6)
        for i, j in zip(diff_positions[0], diff_positions[1]):
            go_val = go_output[i, j]
            py_val = python_output[i, j]
            print(f"  位置({i},{j}): Go={go_val:.6f}, Python={py_val:.6f}, 差异={abs(go_val-py_val):.10f}")
    
    print(f"\n🎉 对拍测试完成！")
    if max_diff < 1e-6:
        print("🎯 Go卷积层实现完全正确！")
    else:
        print("⚠️  需要进一步检查实现细节")

if __name__ == "__main__":
    compare_with_go_output() 