# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# 设置随机种子
np.random.seed(42)
tf.random.set_seed(42)

print("加载数据...")
# 加载训练数据
train_df = pd.read_csv('train.csv')

# 分离特征和标签
X = train_df.drop('label', axis=1).values
y = train_df['label'].values

# 数据归一化（将像素值从0-255缩放到0-1）
X = X / 255.0

# 将数据重塑为28x28x1的格式（用于CNN）
X = X.reshape(-1, 28, 28, 1)

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"训练集大小: {X_train.shape[0]}")
print(f"验证集大小: {X_val.shape[0]}")

# 构建CNN模型
def create_cnn_model():
    model = keras.Sequential([
        # 第一个卷积块
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        
        # 第二个卷积块
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # 第三个卷积块
        layers.Conv2D(64, (3, 3), activation='relu'),
        
        # 展平层
        layers.Flatten(),
        
        # 全连接层
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        
        # 输出层（10个类别）
        layers.Dense(10, activation='softmax')
    ])
    
    return model

# 创建模型
model = create_cnn_model()
model.summary()


# 编译模型
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 设置回调函数
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

model_checkpoint = keras.callbacks.ModelCheckpoint(
    'best_model.h5',
    monitor='val_accuracy',
    save_best_only=True
)

# 训练模型
print("\n开始训练模型...")
history = model.fit(
    X_train, y_train,
    batch_size=128,
    epochs=20,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping, model_checkpoint],
    verbose=1
)

# 评估模型
print("\n评估模型...")
val_loss, val_accuracy = model.evaluate(X_val, y_val)
print(f"验证集准确率: {val_accuracy:.4f}")

# 绘制训练历史
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='训练损失')
plt.plot(history.history['val_loss'], label='验证损失')
plt.title('模型损失')
plt.xlabel('Epoch')
plt.ylabel('损失')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='训练准确率')
plt.plot(history.history['val_accuracy'], label='验证准确率')
plt.title('模型准确率')
plt.xlabel('Epoch')
plt.ylabel('准确率')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png')
plt.show()

# 保存最终模型
model.save('digit_recognition_model.h5')
print("\n模型已保存为 'digit_recognition_model.h5'")

# 创建一个简单的预测函数
def predict_digit(model, image_array):
    """
    预测单个数字图像
    image_array: 28x28的numpy数组
    """
    # 确保输入格式正确
    if image_array.shape != (28, 28):
        raise ValueError("图像必须是28x28的数组")
    
    # 预处理
    image = image_array.reshape(1, 28, 28, 1) / 255.0
    
    # 预测
    predictions = model.predict(image)
    predicted_digit = np.argmax(predictions[0])
    confidence = predictions[0][predicted_digit]
    
    return predicted_digit, confidence

# 测试几个样本
print("\n测试几个样本:")
for i in range(5):
    idx = np.random.randint(0, len(X_val))
    image = X_val[idx].reshape(28, 28)
    true_label = y_val[idx]
    
    pred_digit, confidence = predict_digit(model, image * 255)  # 还原到0-255范围
    
    print(f"样本 {i+1}: 真实标签={true_label}, 预测={pred_digit}, 置信度={confidence:.3f}")