import os
from keras import layers
from keras import models
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences 
import numpy as np
import matplotlib.pyplot as plt

thirty_two_kb = 2 * 1024 * 32
#two hex digits per byte * 1024 bytes per kb * 32 kb

ape_escape_dir = '/home/joshua/machine_learning/ape_escape'
training_data_dir = os.path.join(ape_escape_dir, 'training_data')

train_data = []
train_labels = []

for fname in os.listdir(training_data_dir):
    if fname[-4:] == '.ggr':
        f = open(os.path.join(training_data_dir, fname), mode='rb')
        hexed_file = f.read().hex(' ')
        train_data.append(hexed_file[408:])
        f.close()

labels_path = os.path.join(ape_escape_dir, 'training_labels.txt')
labels = open(labels_path)
for line in labels:
    train_labels.append(float(line))

tkn = Tokenizer(10000)
sequences = tkn.texts_to_sequences(train_data)
x_train = pad_sequences(sequences, maxlen = thirty_two_kb)
y_train = np.asarray(train_labels)

#new code, randomizes order of training data + labels
np.random.shuffle(x_train)
np.random.shuffle(y_train)
'''
placeholder = 1

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape = (placeholder,)))
model.add(layers.Dense(1, activation='sigmoid'))
'''

partitions = 4
epochs = 50
train_len = len(x_train)
partition_len = train_len // partitions
assert train_len == len(y_train)
partitions_range = range(partitions)
x_train_partitions = [x_train[n*partition_len:(n+1)*partition_len] for n in partitions_range]
y_train_partitions = [y_train[n*partition_len:(n+1)*partition_len] for n in partitions_range]
validation_scores = []

def construct_model():
    max_words = 256
    embedding_dim = 100
    
    model = models.Sequential()
    model.add(layers.Embedding(max_words, 32))
    model.add(layers.LSTM(32))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop',
                  loss='mse',
                  metrics='mae')
    return model


for n in partitions_range:
    model = construct_model()

    x_partition_train = np.concatenate([x_train_partitions[m] for m in partitions_range if m != n])
    y_partition_train = np.concatenate([y_train_partitions[m] for m in partitions_range if m != n])
    history = model.fit(x_partition_train, y_partition_train,
                        batch_size = partition_len, epochs=epochs,
                        validation_data=(x_train_partitions[n], y_train_partitions[n]))
    validation_scores.append(history.history['mae'])
avg_validations = [sum([partition[n] for partition in validation_scores]) / partitions for n in range(epochs)]

print(avg_validations)


plt.plot(range(1, len(avg_validations)+1), avg_validations)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

