#IMDB Reviews dataset
from tensorflow.python.keras.datasets import imdb
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words = 10000)
print(x_train[0])
print(y_train[0])
class_names = ['Negative', 'Positive']
word_index = imdb.get_word_index()
print(word_index['hello'])

#Decoding Reviews
reverse_word_index = dict((value, key) for key, value in word_index.items())

def decode(review):
    text = ' '
    for i in review:
        text += reverse_word_index[i]
        text += ' '
    return text

decode(x_train[0])

def show_len():
    print('Len of 1st training example: ', len(x_train[0]))
    print('Len of 2st training example: ', len(x_train[1]))
    print('Len of 1st test example: ', len(x_test[0]))
    print('Len of 1st test example: ', len(x_test[0]))
    
show_len()

#Padding examples
word_index['the']

from tensorflow.python.keras.preprocessing.sequence import pad_sequences
x_train = pad_sequences(x_train, value = 1, padding = 'post', maxlen = 256)
x_test = pad_sequences(x_test, value = 1, padding = 'post', maxlen = 256)

show_len()

decode(x_train[0])

#Creating and Training model
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Embedding, GlobalAveragePooling1D

model = Sequential([
    Embedding(10000, 16),
    GlobalAveragePooling1D(),
    Dense(16, activation = 'relu'),
    Dense(1, activation = 'sigmoid')
])

model.compile(
    loss = 'binary_crossentropy',
    optimizer = 'adam',
    metrics = ['accuracy']
)

model.summary()

from tensorflow.python.keras.callbacks import LambdaCallback
simple_log = LambdaCallback(on_epoch_end = lambda e, l: print(e, end = '.'))

E = 20

h = model.fit(
    x_train, y_train,
    validation_split = 0.2,
    epochs = E,
    callbacks = [simple_log],
    verbose = False
)


#Predictions and Evaluation
import matplotlib.pyplot as plt
%matplotlib inline

plt.plot(range(E), h.history['acc'], label = 'Training')
plt.plot(range(E), h.history['val_acc'], label = 'Validation')
plt.legend()
plt.show()

loss, acc = model.evaluate(x_test, y_test)
print('Test set accuracy:', acc * 100)

import numpy as np


p = model.predict(np.expand_dims(x_test[0], axis = 0))
print(class_names[np.argmax(p[0])])
