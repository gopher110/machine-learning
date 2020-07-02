import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print(tf.__version__)

np.random.seed(42)
tf.random.set_seed(42)

# random linear data : 100 between 0 and 50
n = 100
X = np.linspace(0, 50, n)
y = np.linspace(0, 50, n)

# Adding noise to the random linear data
# 从一个均匀分布[low,high)中随机采样，注意定义域是左闭右开，即包含low，不包含high.
X += np.random.uniform(-10, 10, n)
y += np.random.uniform(-10, 10, n)

# plot of traning data
plt.scatter(X, y)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Training Data')
plt.show()


class LinearModel:
    def __init__(self):
        # y_pred = W*x + b
        self.W = tf.Variable(13.0)
        self.b = tf.Variable(4.0)

    def loss(self, y, y_pred):
        return tf.reduce_mean(tf.square(y-y_pred))

    def train(self, X, y, lr=0.0001, epochs=20, verbose=True):
        def train_step():
            with tf.GradientTape() as t:
                current_loss = self.loss(y, self.predict(X))

            dw, db = t.gradient(current_loss, [self.W, self.b])
            self.W.assign_sub(lr * dw)   # W -=lr * dw
            self.b.assign_sub(lr * db)

            return current_loss

        for epoch in range(epochs):
            current_loss = train_step()
            if verbose:
                print(f'Epoch:{epoch} \nLoss:{current_loss.numpy()}')

    def predict(self, X):
        return self.W * X + self.b


model = LinearModel()
model.train(X, y, epochs=50)
plt.scatter(X, y, label='data')
plt.plot(X, model.predict(X), 'r-', label='predicted')
plt.legend()
plt.savefig("./img/get_started.png", dpi=180)
plt.show()