import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from IPython import get_ipython
from sklearn.preprocessing.data import StandardScaler
import warnings
import os
import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

data = pd.read_csv('./dataset/UFC_1993_2019.csv')  # content/drive/My Drive/tensorflow/dataset
data = data.sample(frac=1)

# print(list(data.columns))

data['Winner'] = data['Winner'].map(lambda x: 1 if x == 'Red' else 0)
data['title_bout'] = data['title_bout'].map(lambda x: 1 if x == 'True' else 0)

train_size = int(0.8*len(data))
features = data.drop(columns=['Winner'])
targets = data['Winner']
X_train, X_test = features.values[:train_size, :], features.values[train_size:, :]
y_train, y_test = targets.values[:train_size], targets.values[train_size:]

# sns.pairplot(data)
corr = data.corr()
cmap = sns.diverging_palette(250, 10, as_cmap=True)
sns.heatmap(corr[['Winner']].head(), cmap=cmap, vmax=.3,
            square=True, linewidths=.5, cbar_kws={"shrink": .5},
            annot=True)

plt.show()
plt.close()

# Tensorflow ANNs
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(1024, activation=tf.nn.leaky_relu),  # if x >0 else alpha&x
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(512, activation=tf.nn.leaky_relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')  # [0,1]
])

red = len(y_train[y_train > 0])
blue = len(y_train)-red
total = len(y_train)
weight_for_red = total / (2 * red)
weight_for_blue = total / (2 * blue)
class_weight = {0: weight_for_blue, 1: weight_for_red}
print(class_weight)


adam_optimizer = tf.keras.optimizers.Adam()
model.compile(
    optimizer=adam_optimizer,
    loss='binary_crossentropy',
    metrics=[
        tf.keras.metrics.TruePositives(name='tp'),
        tf.keras.metrics.FalsePositives(name='fp'),
        tf.keras.metrics.TrueNegatives(name='tn'),
        tf.keras.metrics.FalseNegatives(name='fn'),
        tf.keras.metrics.BinaryAccuracy(name='acc'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc')
    ]
)
save_best_callback = tf.keras.callbacks.ModelCheckpoint(
    './model-{epoch:02d}-{acc:.2f}.hdf5',
    monitor='acc',
    verbose=1,
    save_best_only=True,
    save_weights_only=False,
    save_freq=1
)
logdir = os.path.join('tflogs', datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
tb_train_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1, profile_batch=0)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model.fit(X_train_scaled,
          y_train,
          class_weight=class_weight,
          # batch_size=64,
          validation_split=0.1,
          callbacks=[save_best_callback, tb_train_callback],
          epochs=50)


# model = tf.keras.models.load_model('./model-35-0.88.hdf5')
X_test_scaled = scaler.transform(X_test)
model.evaluate(X_test_scaled, y_test)
# print(np.round(model.predict(X_test)))