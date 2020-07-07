"""
MNIST 手写数据集
28x28像素 灰度图片
存储在28x28 的二维数组中


"""
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print(f"Training set:{len(X_train)}\nTesting set:{len(X_test)}")

print(f"X_train:{X_train.shape} {X_train.dtype}")
print(f"y_train:{y_train.shape} {y_train.dtype}")

print(X_train[0])

for i in range(4):
    num = np.random.randint(1, 6000)
    plt.subplot(1, 4, i+1)
    plt.axis('off')
    plt.imshow(X_train[num], cmap='gray')
    plt.title(y_train[num])

plt.show()