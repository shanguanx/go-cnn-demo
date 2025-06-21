# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def flatten_tensor(tensor):
    """Helper function to flatten tensor for display"""
    return tensor.detach().numpy().flatten().tolist()

def test_cnn_comparison():
    """Test CNN with fixed weights matching Go implementation"""
    print("=== Python CNN Test with Fixed Weights ===")
    print("=" * 50)
    
    # Disable gradient computation initially for cleaner output
    torch.set_grad_enabled(False)
    
    # Input data (same as Go)
    input_data = [
        0.1, 0.2, 0.3, 0.4,
        0.5, 0.6, 0.7, 0.8,
        0.9, 1.0, 0.1, 0.2,
        0.3, 0.4, 0.5, 0.6,
    ]
    
    # Create input tensor: (batch=1, channels=1, height=4, width=4)
    input_tensor = torch.tensor(input_data, dtype=torch.float32).reshape(1, 1, 4, 4)
    
    print("\n1. INPUT LAYER")
    print(f"   Shape: {tuple(input_tensor.shape)}")
    print(f"   Data: {flatten_tensor(input_tensor)}")
    
    # Conv2D Layer: 1->2 channels, 3x3 kernel, stride=1, padding=0
    conv1 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0, bias=True)
    
    # Set fixed weights matching Go implementation
    # Go pattern: float64(i*10+j)*0.1 for weights, float64(i)*0.5 for bias
    with torch.no_grad():
        # Channel 0 weights: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        conv1.weight[0, 0] = torch.tensor([
            [0.0, 0.1, 0.2],
            [0.3, 0.4, 0.5],
            [0.6, 0.7, 0.8]
        ])
        
        # Channel 1 weights: [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8]
        conv1.weight[1, 0] = torch.tensor([
            [1.0, 1.1, 1.2],
            [1.3, 1.4, 1.5],
            [1.6, 1.7, 1.8]
        ])
        
        # Bias: [0.0, 0.5]
        conv1.bias[0] = 0.0
        conv1.bias[1] = 0.5
    
    print("\n2. CONV2D LAYER")
    print(f"   Parameters: in_channels=1, out_channels=2, kernel=3x3, stride=1, padding=0")
    print(f"   Weights shape: {tuple(conv1.weight.shape)}")
    print(f"   Weights: {flatten_tensor(conv1.weight)}")
    print(f"   Bias: {flatten_tensor(conv1.bias)}")
    
    conv1_output = conv1(input_tensor)
    print(f"   Output shape: {tuple(conv1_output.shape)}")
    print(f"   Output: {flatten_tensor(conv1_output)}")
    
    # ReLU Activation
    relu1_output = F.relu(conv1_output)
    print("\n3. RELU ACTIVATION")
    print(f"   Output shape: {tuple(relu1_output.shape)}")
    print(f"   Output: {flatten_tensor(relu1_output)}")
    
    # MaxPool2D: 2x2 window, stride=1
    pool1 = nn.MaxPool2d(kernel_size=2, stride=1)
    pool1_output = pool1(relu1_output)
    print("\n4. MAXPOOL2D LAYER")
    print(f"   Parameters: kernel=2x2, stride=1")
    print(f"   Output shape: {tuple(pool1_output.shape)}")
    print(f"   Output: {flatten_tensor(pool1_output)}")
    
    # Flatten
    flattened = pool1_output.flatten(start_dim=1)
    print("\n5. FLATTEN LAYER")
    print(f"   Output shape: {tuple(flattened.shape)}")
    print(f"   Output: {flatten_tensor(flattened)}")
    
    # Dense Layer: 2->2
    dense1 = nn.Linear(in_features=2, out_features=2)
    
    # Set fixed weights matching Go implementation
    # Go pattern: float64(i*cols+j)*0.1 for weights, float64(j)*0.5 for bias
    with torch.no_grad():
        # Go uses row-major (input x output), PyTorch uses (output x input)
        # Go weights matrix: [[0.0, 0.1], [0.2, 0.3]]
        # PyTorch needs: [[0.0, 0.2], [0.1, 0.3]]
        dense1.weight[0, 0] = 0.0
        dense1.weight[0, 1] = 0.2
        dense1.weight[1, 0] = 0.1
        dense1.weight[1, 1] = 0.3
        
        # Bias: [0.0, 0.5]
        dense1.bias[0] = 0.0
        dense1.bias[1] = 0.5
    
    print("\n6. DENSE LAYER")
    print(f"   Parameters: in_features=2, out_features=2")
    print(f"   Weights shape: {tuple(dense1.weight.shape)}")
    print(f"   Weights: {flatten_tensor(dense1.weight)}")
    print(f"   Bias: {flatten_tensor(dense1.bias)}")
    
    dense1_output = dense1(flattened)
    print(f"   Output shape: {tuple(dense1_output.shape)}")
    print(f"   Output: {flatten_tensor(dense1_output)}")
    
    # Loss calculation (target class = 1)
    target = torch.tensor([1], dtype=torch.long)
    
    # Enable gradients for backward pass
    torch.set_grad_enabled(True)
    
    # Recompute forward pass with gradients
    input_tensor.requires_grad_(True)
    x = conv1(input_tensor)
    x = F.relu(x)
    x = pool1(x)
    x = x.flatten(start_dim=1)
    x = dense1(x)
    
    loss = F.cross_entropy(x, target)
    
    print("\n7. LOSS CALCULATION")
    print(f"   Target label: 1")
    print(f"   Loss value: {loss.item():.6f}")
  
    
    # Backward pass
    print("\n8. BACKWARD PASS")
    loss.backward()
    print("   Backward pass completed")
    
    # Print gradients
    print("\n9. GRADIENTS")
    print(f"   Dense weight gradients: {flatten_tensor(dense1.weight.grad)}")
    print(f"   Dense bias gradients: {flatten_tensor(dense1.bias.grad)}")
    print(f"   Conv weight gradients: {flatten_tensor(conv1.weight.grad)}")
    print(f"   Conv bias gradients: {flatten_tensor(conv1.bias.grad)}")
    print(f"   Input gradients: {flatten_tensor(input_tensor.grad)}")
    
    # 创建优化器并执行参数更新
    optimizer = torch.optim.SGD([conv1.weight, conv1.bias, dense1.weight, dense1.bias], lr=0.1)
    
    # Optimizer step
    print("\n10. OPTIMIZER STEP")
    optimizer.step()
    print("   Parameter update completed")
    
    # Updated weights
    print("\n11. UPDATED WEIGHTS")
    print(f"   Dense weights: {flatten_tensor(dense1.weight)}")
    print(f"   Dense bias: {flatten_tensor(dense1.bias)}")
    print(f"   Conv weights: {flatten_tensor(conv1.weight)}")
    print(f"   Conv bias: {flatten_tensor(conv1.bias)}")
    
    print("\n" + "=" * 50)
    print("=== Test Complete ===")


def test_cnn_all_layers():
    """Test CNN with all layers using fixed weights"""
    print("\n=== CNN Test with All Layers ===")
    print("=" * 50)
    
    torch.set_grad_enabled(False)
    
    # Larger input: 8x8 image, 1 channel
    input_size = 64
    input_data = [i * 0.01 for i in range(input_size)]
    input_tensor = torch.tensor(input_data, dtype=torch.float32).reshape(1, 1, 8, 8)
    
    print(f"\n1. Input: shape={tuple(input_tensor.shape)}")
    
    # Conv2D: 1->4 channels, 3x3 kernel
    conv1 = nn.Conv2d(1, 4, 3, stride=1, padding=0)
    # Initialize with fixed pattern
    with torch.no_grad():
        for out_ch in range(4):
            for i in range(9):
                conv1.weight[out_ch, 0].view(-1)[i] = (out_ch * 10 + i) * 0.1
            conv1.bias[out_ch] = out_ch * 0.5
    
    x = conv1(input_tensor)
    print(f"2. Conv2D: shape={tuple(x.shape)}, out_channels=4, kernel=3x3")
    
    # ReLU
    x = F.relu(x)
    print(f"3. ReLU: shape={tuple(x.shape)}")
    
    # MaxPool: 2x2, stride=2
    pool1 = nn.MaxPool2d(2, stride=2)
    x = pool1(x)
    print(f"4. MaxPool2D: shape={tuple(x.shape)}, kernel=2x2, stride=2")
    
    # Conv2D: 4->8 channels, 3x3 kernel
    conv2 = nn.Conv2d(4, 8, 3, stride=1, padding=0)
    # Initialize with fixed pattern
    with torch.no_grad():
        for out_ch in range(8):
            for in_ch in range(4):
                for i in range(9):
                    idx = in_ch * 9 + i
                    conv2.weight[out_ch, in_ch].view(-1)[i] = (out_ch * 40 + idx) * 0.1
            conv2.bias[out_ch] = out_ch * 0.5
    
    x = conv2(x)
    print(f"5. Conv2D: shape={tuple(x.shape)}, out_channels=8, kernel=3x3")
    
    # ReLU
    x = F.relu(x)
    print(f"6. ReLU: shape={tuple(x.shape)}")
    
    # Flatten
    x = x.flatten(start_dim=1)
    print(f"7. Flatten: shape={tuple(x.shape)}")
    
    # Dense: 8->16
    dense1 = nn.Linear(8, 16)
    with torch.no_grad():
        for i in range(8):
            for j in range(16):
                dense1.weight[j, i] = (i * 16 + j) * 0.1
        for j in range(16):
            dense1.bias[j] = j * 0.5
    
    x = dense1(x)
    print(f"8. Dense: shape={tuple(x.shape)}, out_features=16")
    
    # ReLU
    x = F.relu(x)
    print(f"9. ReLU: shape={tuple(x.shape)}")
    
    # Dense: 16->10
    dense2 = nn.Linear(16, 10)
    with torch.no_grad():
        for i in range(16):
            for j in range(10):
                dense2.weight[j, i] = (i * 10 + j) * 0.1
        for j in range(10):
            dense2.bias[j] = j * 0.5
    
    x = dense2(x)
    print(f"10. Dense: shape={tuple(x.shape)}, out_features=10")
    
    # Loss
    target = torch.tensor([3], dtype=torch.long)
    
    # Enable gradients and recompute
    torch.set_grad_enabled(True)
    input_tensor.requires_grad_(True)
    
    # Forward pass with gradients
    x = input_tensor
    x = conv1(x)
    x = F.relu(x)
    x = pool1(x)
    x = conv2(x)
    x = F.relu(x)
    x = x.flatten(start_dim=1)
    x = dense1(x)
    x = F.relu(x)
    x = dense2(x)
    
    loss = F.cross_entropy(x, target)
    print(f"\n11. Loss: {loss.item():.6f} (target class=3)")
    
    # Backward
    loss.backward()
    print("\nBackward pass completed successfully!")
    
    print("\n" + "=" * 50)
    print("=== All Layers Test Complete ===")


if __name__ == "__main__":
    # Run comparison test
    test_cnn_comparison()
    
    # Run all layers test
    test_cnn_all_layers()