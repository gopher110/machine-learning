import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


data = pd.read_csv('./dataset/sentiment_tweets.csv', encoding='latin-1', header=None)
data = data.sample(frac=1)
print(data.head())

# traing-test split
train_size = int(0.7*len(data))
features = data[5]
targets = data[0]
X_train, X_test = features.values[:train_size], features.values[train_size:]
y_train, y_test = targets.values[:train_size], targets.values[train_size:]

y_train[y_train==2] = 1
y_train[y_train==4] = 2
y_test[y_test==2] = 1
y_test[y_test==4] = 2

# count vectorize them
from sklearn.feature_extraction.text import CountVectorizer

count_vectorizer = CountVectorizer(max_features=100)
X_train_num = count_vectorizer.fit_transform(X_train).toarray()
X_test_num = count_vectorizer.transform(X_test).toarray()

instance = 45542
print(X_train[instance])
print(X_train_num[instance])

# concatenate the text
text = ' '.join(data[5])
text[:300]

import  numpy as np
vocab = sorted(set(text))
print(f'{len(vocab)} unique characters')

char2idx = {u: i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

text_as_int = np.array([char2idx[c] for c in text])

import tensorflow as tf
# the maximum length sentence we want for a single input in characters
seq_length = 128
examples_per_epoch = len(text) // (seq_length+1)

# create training examples / targets
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

for i in char_dataset.take(5):
    print(idx2char[i.numpy()])


sequences = char_dataset.batch(seq_length+1, drop_remainder=True)
for item in sequences.take(5):
    print(repr(''.join(idx2char[item.numpy()])))

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text,target_text


dataset = sequences.map(split_input_target)

for input_example, target_example in dataset.take(1):
    print('Input data: ', repr(''.join(idx2char[input_example.numpy()])))
    print('Target data: ', repr(''.join(idx2char[target_example.numpy()])))

for i, (input_idx, target_idx) in enumerate(zip(input_example[:5], target_example[:5])):
    print('Step {:4d}'.format(i))
    print('input:{} ({:s})'.format(input_idx, repr(idx2char[input_idx])))
    print(" expected output: {} (){:s}".format(target_idx, repr(idx2char[target_idx])))

BATCH_SIZE = 64

BUFFER_SIZE = 10000
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE,drop_remainder=True)
dataset

# text generation
def get_model(batch_size, vocab, embedding_dim=256, rnn_units=512):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(len(vocab), embedding_dim,
                             batch_input_shape=[batch_size, None]),
        tf.keras.layers.LSTM(rnn_units,
                             return_sequences=True,
                             stateful=True,
                             recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(len(vocab))
    ])
    return model


model = get_model(BATCH_SIZE, vocab)
for input_example_batch, target_example_batch in dataset.take(1):
    example_batch_predictions = model(input_example_batch)

sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy()
print('*******sampled_indices********')
print(sampled_indices)

print("Input: \n", repr("".join(idx2char[input_example_batch[0]])))
print()
print('Next char predictions: \n', repr("".join(idx2char[sampled_indices])))

EPOCHS = 20
checkpoint_dir= './training_checkpoints'
# name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath= checkpoint_dir,
    save_weights_only=True
)
model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback], verbose=1)

def generate_text(model, start_string):
    num_generate = 1000
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)
    text_generated = []
    temperature = 1.0

    model.reset_states()
    for i in range(num_generate):
        prdictions = model(input_eval)
        # remove the batch dimension
        predictions = tf.squeeze(predictions, 0)

        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(idx2char[predicted_id])
    return start_string + ''.join(text_generated)

# rebuild model with batch size=1  for generating
# TODO make this a function
generating_model = get_model(1, vocab)
generating_model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
generating_model.build(tf.TensorShape([1, None]))

print(generate_text(generating_model, start_string=u"Well, "))

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation=tf.nn.leaky_relu),  # x if x>0 else alpha*x
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(3)
])

adam_optimizer = tf.keras.optimizers.Adam()
model.compile(
    optimizer=adam_optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

print(y_train)

model.fit(X_train_num,
          y_train,
          batch_size=64,
          epochs=2,
          validation_split=0.1,
          verbose=1)