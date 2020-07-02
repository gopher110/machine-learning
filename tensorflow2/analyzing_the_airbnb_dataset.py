import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')   # 忽略匹配的警告
print(tf.__version__)


# load data and take a look at 48895 16
data = pd.read_csv('./dataset/AB_NYC_2019.csv').sample(frac=1)
features = data[['neighbourhood_group', 'room_type', 'minimum_nights', 'number_of_reviews', 'reviews_per_month', 'calculated_host_listings_count', 'availability_365']]

# print(features.isna().sum())
features.loc[features['reviews_per_month'].isna(), 'reviews_per_month'] = 0
# print(features.isna().sum())
onehot_neighborhood_group = pd.get_dummies(features['neighbourhood_group'])
onehot_room_type = pd.get_dummies(features['room_type'])

features = features.drop(columns=['neighbourhood_group', 'room_type'])
features = pd.concat([features, onehot_neighborhood_group, onehot_room_type], axis=1)  # 横向表拼接（行对齐）
# print(features.head())
targets = data['price']

train_size = int(0.7 * len(data))
X_train, X_test = features.values[:train_size, :], features.values[train_size:, :]
y_train, y_test = targets.values[:train_size], targets.values[train_size:]

print(len(X_train[0]))


# data visualization and analysis


# the Tensorflow2 machine learning approaches

# Linear Regrasseion
class LinerModel:
    def __init__(self):
        # y_pred = W*x + b
        self.initializer = tf.keras.initializers.GlorotUniform()

    def loss(self, y, y_pred):
        return tf.reduce_mean(tf.abs(y-y_pred))   #

    def train(self, X, y, lr=0.00001, epochs=20, verbose=True):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape((-1, 1))  # 自动计算行数 列数为1

        self.W = tf.Variable(
            initial_value=self.initializer(shape=(len(X[0]), 1), dtype='float32'))
        self.b = tf.Variable(
            initial_value=self.initializer(shape=(1,), dtype='float32'))

        def train_step():
            with tf.GradientTape() as t:
                current_loss = self.loss(y, self.predict(X))

            dW, db = t.gradient(current_loss, [self.W, self.b])
            self.W.assign_sub(lr * dW)  # W -= lr* dw
            self.b.assign_sub(lr * db)
            return current_loss

        for epoch in range(epochs):
            current_loss = train_step()
            if verbose:
                print(f'Epoch:{epoch} \nLoss:{current_loss.numpy()}')

    def predict(self, X):
        # [a, b]* [b, c]
        # x -> [n_instances, n_features] [n_feature, 1]
        return tf.matmul(X, self.W)+self.b


model = LinerModel()
model.train(X_train, y_train, epochs=100)
# conclusions

