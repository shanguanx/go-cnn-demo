# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras

print("加载测试数据...")
test_df = pd.read_csv('test.csv')

# 预处理测试数据
X_test = test_df.values / 255.0
X_test = X_test.reshape(-1, 28, 28, 1)

print(f"测试集大小: {X_test.shape[0]}")

# 加载训练好的模型
print("\n加载模型...")
model = keras.models.load_model('best_model.h5')

# 进行预测
print("开始预测...")
predictions = model.predict(X_test, batch_size=128)
predicted_labels = np.argmax(predictions, axis=1)

# 创建提交文件
submission = pd.DataFrame({
    'ImageId': range(1, len(predicted_labels) + 1),
    'Label': predicted_labels
})

# 保存提交文件
submission.to_csv('submission.csv', index=False)
print("\n预测完成！结果已保存到 'submission.csv'")

# 显示前10个预测结果
print("\n前10个预测结果:")
print(submission.head(10))