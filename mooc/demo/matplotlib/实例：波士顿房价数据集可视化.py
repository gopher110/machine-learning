"""
keras
是一个高层的神经网络和深度学习库
可以快速搭建神经网络模型，非常易于调试与扩展，
TensorFlow 的官方 API
内置了一些常用的公共数据集，可以通过 keras.datasets 模块访问加载

Keras 中集成的数据集
1 boston_housing 波士顿房价数据集
2 CIFAR10 10种类别的图片集
3 CIFAR100 100种类别的图片集
4 MNIST  手写数字图片集
5 Fashion-MNIST 10种时尚类别的图片集
6 IMDB 电影点评数据集
7 reuters 路透社新闻数据集
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
boston_housing = tf.keras.datasets.boston_housing

(X_train, y_train), (_, _) = boston_housing.load_data(test_split=0)
print("Training set:", len(X_train))
titles = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
          "DIS", "RAD", "TAX", "PTRATIO", "B-1000", "LASTAT", "MEDV"]

plt.figure(figsize=(12, 12))
for i in range(13):
    plt.subplot(4, 4, (i+1))
    plt.scatter(X_train[:, i], y_train)

    plt.xlabel(titles[i])
    plt.ylabel("Price($1000's)")
    plt.title(str(i+1)+"."+titles[i]+" - Price")

plt.show()

