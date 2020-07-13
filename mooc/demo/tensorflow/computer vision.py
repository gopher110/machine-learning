import tensorflow as tf
import tensorflow.keras as keras
import ssl
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
ssl._create_default_https_context = ssl._create_unverified_context

class myCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.95:
            print("\nReached 95% accuracy so cancelling training!")
            self.model.stop_training = True


callbacks = myCallback()
data = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = data.load_data()
training_images = training_images / 255.0
test_images = test_images / 255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=50, callbacks=[callbacks])

print('++++++++++++++++++++++++++++++++++++')
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'test_loss:{test_loss} test_acc:{test_acc}')
classifications = model.predict(test_images)
print(classifications[0])
print(test_labels[0])
