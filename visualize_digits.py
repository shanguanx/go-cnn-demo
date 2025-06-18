# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# 读取训练数据
df = pd.read_csv('train.csv')

# 创建保存图片的目录
output_dir = 'digit_images'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 可视化前100个数字（或更少）
num_samples = min(100, len(df))

for idx in range(num_samples):
    # 获取标签和像素值
    label = df.iloc[idx, 0]
    pixels = df.iloc[idx, 1:].values
    
    # 将784个像素值重塑为28x28的图像
    image_array = pixels.reshape(28, 28)
    
    # 创建图像
    img = Image.fromarray(image_array.astype(np.uint8), 'L')
    
    # 保存图像
    filename = os.path.join(output_dir, f'digit_{idx:04d}_label_{label}.png')
    img.save(filename)
    
    # 每10个图像打印一次进度
    if (idx + 1) % 10 == 0:
        print(f'已处理 {idx + 1}/{num_samples} 个图像')

print(f'\n完成！已将前 {num_samples} 个数字图像保存到 {output_dir} 目录中')

# 可选：显示前9个数字的预览
fig, axes = plt.subplots(3, 3, figsize=(6, 6))
axes = axes.ravel()

for i in range(9):
    label = df.iloc[i, 0]
    pixels = df.iloc[i, 1:].values
    image_array = pixels.reshape(28, 28)
    
    axes[i].imshow(image_array, cmap='gray')
    axes[i].set_title(f'Label: {label}')
    axes[i].axis('off')

plt.tight_layout()
plt.savefig('digit_preview.png')
plt.show()