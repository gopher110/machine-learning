import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
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
# sns.pairplot(data)
corr = data.corr()
cmap =sns.diverging_palette(250, 10, as_cmap=True)
plt.figure(figsize=(8, 8))
sns.heatmap(corr, square=True, cmap=cmap, annot=True)
plt.show()
# the Tensorflow2 machine learning approaches
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),  # 0 or x:_____/
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1)
])


def R_squared(y_true,y_pred):
    residual = tf.reduce_sum(tf.square(y_true - y_pred))
    total = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    r2 = 1.0 - residual / total
    return r2


adam_optimizer = tf.keras.optimizers.Adam()
loss_fn = tf.keras.losses.MAE
model.compile(
    optimizer=adam_optimizer,
    loss=loss_fn,
    metrics=[tf.keras.metrics.MAE,
             tf.keras.metrics.MSE,
             R_squared,  # -1 and 1 /  <0 ==> useless 0 and 1 ==> better close to 1
             ]
)
model.fit(X_train, y_train, epochs=10)
model.save('./my ann.h5')
loaded_model = tf.keras.models.load_model('./my ann.h5', custom_objects={"R_squared": R_squared})
print(loaded_model.summary())
print(loaded_model.evaluate(X_test, y_test))
print(loaded_model(X_test[:2]))
print(y_test[:2])

# conclusions
"""
Sometimes the data set limits us WRT to results
Data preprocessing and analysis is important Tensorflow does not live in a bubble,it's a tool
TF 2 simplifies many things: no more placeholders,eager execution,.numpy(),no more sessions,Keras has a bigger role
ANNs are very sensitive to hyperparameter choice
running on GPU can help,but it's not a must
"""


