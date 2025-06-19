import numpy as np

def verify_convolution():
    """
    验证卷积层前向传播结果
    特别是第一个样本（全1输入）的卷积计算
    """
    
    # 输入数据：3个样本，每个样本是3x3矩阵展平
    # 样本0: 全1输入 [1,1,1,1,1,1,1,1,1]
    # 样本1: 递增输入 [1,2,3,4,5,6,7,8,9]  
    # 样本2: 交替输入 [1,0,1,0,1,0,1,0,1]
    
    # 重塑为3D格式：(batch_size, channels, height, width)
    input_3d = np.array([
        [[[1, 1, 1],
          [1, 1, 1],
          [1, 1, 1]]],  # 样本0
        
        [[[1, 2, 3],
          [4, 5, 6],
          [7, 8, 9]]],  # 样本1
        
        [[[1, 0, 1],
          [0, 1, 0],
          [1, 0, 1]]]   # 样本2
    ])
    
    # 卷积核：3x3，全1权重
    kernel = np.ones((3, 3))
    
    # 偏置：0
    bias = 0
    
    # 步长：1
    stride = 1
    
    # 填充：1
    padding = 1
    
    print("=== 验证卷积层前向传播 ===")
    print(f"输入形状: {input_3d.shape}")
    print(f"卷积核形状: {kernel.shape}")
    print(f"步长: {stride}, 填充: {padding}")
    print()
    
    # 验证每个样本
    for sample_idx in range(3):
        print(f"--- 样本 {sample_idx} ---")
        
        # 获取当前样本
        sample = input_3d[sample_idx, 0]  # 去掉channel维度
        print(f"输入矩阵 ({sample.shape[0]}x{sample.shape[1]}):")
        print(sample)
        
        # 添加填充
        padded = np.pad(sample, padding, mode='constant', constant_values=0)
        print(f"\n填充后矩阵 ({padded.shape[0]}x{padded.shape[1]}):")
        print(padded)
        
        # 计算输出尺寸
        output_h = (sample.shape[0] + 2*padding - kernel.shape[0]) // stride + 1
        output_w = (sample.shape[1] + 2*padding - kernel.shape[1]) // stride + 1
        output = np.zeros((output_h, output_w))
        
        print(f"\n输出矩阵 ({output_h}x{output_w}):")
        
        # 执行卷积
        for i in range(output_h):
            for j in range(output_w):
                # 计算在填充后矩阵中的起始位置
                start_i = i * stride
                start_j = j * stride
                
                # 提取卷积窗口
                window = padded[start_i:start_i+kernel.shape[0], 
                               start_j:start_j+kernel.shape[1]]
                
                # 计算卷积结果
                conv_result = np.sum(window * kernel) + bias
                output[i, j] = conv_result
                
                # 打印详细信息（仅对样本0）
                if sample_idx == 0:
                    print(f"位置({i},{j}): 窗口=\n{window}, 结果={conv_result}")
        
        print(f"\n最终输出:")
        print(output)
        
        # 展平输出
        output_flat = output.flatten()
        print(f"展平输出: {output_flat}")
        print()

def verify_go_output():
    """
    验证Go代码的输出结果
    """
    print("=== Go代码输出验证 ===")
    
    # Go代码的实际输出
    go_outputs = [
        [4, 6, 4, 6, 9, 6, 4, 6, 4],  # 样本0
        [12, 21, 16, 27, 45, 33, 24, 39, 28],  # 样本1
        [2, 3, 2, 3, 5, 3, 2, 3, 2]   # 样本2
    ]
    
    for i, output in enumerate(go_outputs):
        print(f"样本{i}的Go输出: {output}")
        print(f"重塑为3x3:\n{np.array(output).reshape(3, 3)}")
        print()

if __name__ == "__main__":
    verify_convolution()
    print("\n" + "="*50 + "\n")
    verify_go_output() 