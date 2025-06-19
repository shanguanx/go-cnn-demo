import numpy as np

def im2col(input_data, kernel_size, stride=1, padding=1):
    """
    将输入数据转换为列矩阵形式，用于卷积运算
    """
    batch_size, channels, height, width = input_data.shape
    out_height = (height + 2 * padding - kernel_size) // stride + 1
    out_width = (width + 2 * padding - kernel_size) // stride + 1
    
    # 添加padding
    padded_input = np.pad(input_data, ((0, 0), (0, 0), (padding, padding), (padding, padding)), 'constant')
    
    # 创建输出矩阵
    col_matrix = np.zeros((batch_size, channels * kernel_size * kernel_size, out_height * out_width))
    
    for i in range(out_height):
        for j in range(out_width):
            h_start = i * stride
            h_end = h_start + kernel_size
            w_start = j * stride
            w_end = w_start + kernel_size
            
            # 提取当前窗口的数据
            window = padded_input[:, :, h_start:h_end, w_start:w_end]
            # 展平并存储
            col_matrix[:, :, i * out_width + j] = window.reshape(batch_size, -1)
    
    return col_matrix

def col2im(col_matrix, input_shape, kernel_size, stride=1, padding=1):
    """
    将列矩阵转换回图像形式，用于反向传播
    """
    batch_size, channels, height, width = input_shape
    out_height = (height + 2 * padding - kernel_size) // stride + 1
    out_width = (width + 2 * padding - kernel_size) // stride + 1
    
    # 创建带padding的输入梯度
    padded_grad = np.zeros((batch_size, channels, height + 2 * padding, width + 2 * padding))
    
    for i in range(out_height):
        for j in range(out_width):
            h_start = i * stride
            h_end = h_start + kernel_size
            w_start = j * stride
            w_end = w_start + kernel_size
            
            # 获取当前位置的梯度
            grad_window = col_matrix[:, :, i * out_width + j].reshape(batch_size, channels, kernel_size, kernel_size)
            # 累加到对应位置
            padded_grad[:, :, h_start:h_end, w_start:w_end] += grad_window
    
    # 移除padding
    if padding > 0:
        grad_input = padded_grad[:, :, padding:-padding, padding:-padding]
    else:
        grad_input = padded_grad
    
    return grad_input

def convolution_backward_verify():
    """
    验证卷积层反向传播，使用与Go测试相同的数据
    """
    print("=== 卷积层反向传播验证 ===\n")
    
    # 设置参数（与Go测试相同）
    batch_size = 1
    in_channels = 1
    out_channels = 1
    kernel_size = 3
    stride = 1
    padding = 1
    input_height = 3
    input_width = 3
    
    # 输入数据：[1, 2, 3, 4, 5, 6, 7, 8, 9]
    input_data = np.array([[[[1, 2, 3],
                             [4, 5, 6],
                             [7, 8, 9]]]], dtype=np.float32)  # (1, 1, 3, 3)
    
    # 权重：[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    weights = np.array([[[[0.0, 0.1, 0.2],
                          [0.3, 0.4, 0.5],
                          [0.6, 0.7, 0.8]]]], dtype=np.float32)  # (1, 1, 3, 3)
    
    # 偏置：[0.0]
    biases = np.array([0.0], dtype=np.float32)
    
    # 梯度输出：全1矩阵
    grad_output = np.ones((batch_size, out_channels, input_height, input_width), dtype=np.float32)
    
    print("输入矩阵 (1x9):")
    print(input_data.flatten())
    print()
    
    print("初始权重矩阵 (1x9):")
    print(weights.flatten())
    print()
    
    print("初始偏置矩阵 (1x1):")
    print(biases)
    print()
    
    print("梯度输出矩阵 (1x9):")
    print(grad_output.flatten())
    print()
    
    # 前向传播（简化版本）
    # 添加padding
    padded_input = np.pad(input_data, ((0, 0), (0, 0), (padding, padding), (padding, padding)), 'constant')
    
    # 手动计算卷积输出
    output = np.zeros((batch_size, out_channels, input_height, input_width))
    
    for i in range(input_height):
        for j in range(input_width):
            h_start = i
            h_end = h_start + kernel_size
            w_start = j
            w_end = w_start + kernel_size
            
            # 提取输入窗口
            input_window = padded_input[0, 0, h_start:h_end, w_start:w_end]
            # 卷积运算
            output[0, 0, i, j] = np.sum(input_window * weights[0, 0]) + biases[0]
    
    print("前向传播输出 (1x9):")
    print(output.flatten())
    print()
    
    # 反向传播
    # 1. 计算偏置梯度：所有输出位置的梯度之和
    grad_bias = np.sum(grad_output)
    
    # 2. 计算权重梯度：输入窗口与输出梯度的外积
    grad_weights = np.zeros_like(weights[0, 0])
    
    for i in range(input_height):
        for j in range(input_width):
            h_start = i
            h_end = h_start + kernel_size
            w_start = j
            w_end = w_start + kernel_size
            
            # 提取输入窗口
            input_window = padded_input[0, 0, h_start:h_end, w_start:w_end]
            # 权重梯度累加
            grad_weights += input_window * grad_output[0, 0, i, j]
    
    # 3. 计算输入梯度：权重与输出梯度的卷积
    grad_input = np.zeros_like(input_data[0, 0])
    
    # 创建带padding的梯度输出
    padded_grad_output = np.pad(grad_output[0, 0], ((padding, padding), (padding, padding)), 'constant')
    
    # 翻转权重（卷积的转置操作）
    flipped_weights = np.flip(weights[0, 0])
    
    for i in range(input_height):
        for j in range(input_width):
            h_start = i
            h_end = h_start + kernel_size
            w_start = j
            w_end = w_start + kernel_size
            
            # 提取梯度输出窗口
            grad_window = padded_grad_output[h_start:h_end, w_start:w_end]
            # 输入梯度累加
            grad_input[i, j] = np.sum(grad_window * flipped_weights)
    
    print("=== 反向传播结果 ===")
    print()
    
    print("输入梯度矩阵 (1x9):")
    print(grad_input.flatten())
    print()
    
    print("权重梯度矩阵 (1x9):")
    print(grad_weights.flatten())
    print()
    
    print("偏置梯度矩阵 (1x1):")
    print(np.array([grad_bias]))
    print()
    
    # 与Go测试结果对比
    print("=== 与Go测试结果对比 ===")
    print()
    
    go_input_grad = np.array([0.8, 1.5, 1.2, 2.1, 3.6, 2.7, 2.0, 3.3, 2.4])
    go_weight_grad = np.array([12.0, 21.0, 16.0, 27.0, 45.0, 33.0, 24.0, 39.0, 28.0])
    go_bias_grad = np.array([9.0])
    
    print("Python输入梯度:", grad_input.flatten())
    print("Go输入梯度:    ", go_input_grad)
    print("输入梯度差异:   ", np.abs(grad_input.flatten() - go_input_grad))
    print()
    
    print("Python权重梯度:", grad_weights.flatten())
    print("Go权重梯度:    ", go_weight_grad)
    print("权重梯度差异:   ", np.abs(grad_weights.flatten() - go_weight_grad))
    print()
    
    print("Python偏置梯度:", np.array([grad_bias]))
    print("Go偏置梯度:    ", go_bias_grad)
    print("偏置梯度差异:   ", np.abs(np.array([grad_bias]) - go_bias_grad))
    print()
    
    # 检查是否匹配
    input_match = np.allclose(grad_input.flatten(), go_input_grad, atol=1e-6)
    weight_match = np.allclose(grad_weights.flatten(), go_weight_grad, atol=1e-6)
    bias_match = np.allclose(np.array([grad_bias]), go_bias_grad, atol=1e-6)
    
    print("=== 验证结果 ===")
    print(f"输入梯度匹配: {'✅' if input_match else '❌'}")
    print(f"权重梯度匹配: {'✅' if weight_match else '❌'}")
    print(f"偏置梯度匹配: {'✅' if bias_match else '❌'}")
    
    if input_match and weight_match and bias_match:
        print("\n🎉 所有梯度计算都匹配Go实现！")
    else:
        print("\n⚠️  存在不匹配的梯度计算")

if __name__ == "__main__":
    convolution_backward_verify() 