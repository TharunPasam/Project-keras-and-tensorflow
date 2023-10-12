#Importing tensorflow
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)
print('Using TensorFlow version', tf.__version__)

from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)
print('x_test shape:', x_test.shape)
print('y_test shape:', y_test.shape)

from matplotlib import pyplot as plt
%matplotlib inline

plt.imshow(x_train[0], cmap='binary')
plt.show()

y_train[0]

print(set(y_train))

#One hot encoding
from tensorflow.keras.utils import to_categorical

y_train_encoded = to_categorical(y_train)
y_test_encoded = to_categorical(y_test)

print('y_train_encoded shape:', y_train_encoded.shape)
print('y_test_encoded shape:', y_test_encoded.shape)

y_train_encoded[5]

import numpy as np

x_train_modify = np.reshape(x_train, (60000, 784))
x_test_modify = np.reshape(x_test, (10000, 784))

print('x_train_modify shape:', x_train_modify.shape)
print('x_test_modify shape:',x_test_modify.shape)

print(set(x_test_modify[0]))

x_mean = np.mean(x_train_modify)
x_std = np.std(x_train_modify)

epsilon = 1e-10

x_train_norm = (x_train_modify - x_mean) / (x_std + epsilon)
x_test_norm = (x_test_modify - x_mean) / (x_std + epsilon)

print(set(x_train_norm[0]))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(64, activation='relu', input_shape=(784,)),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])                                                            #Creating model

model.compile(
    optimizer='sgd',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()                                              #Compiling Model

#Training Model
model.fit(x_train_norm, y_train_encoded, epochs=3)

_, accuracy = model.evaluate(x_test_norm, y_test_encoded)
print('Test set accuracy:', accuracy * 100)

#Predictions
preds = model.predict(x_test_norm)
print('Shape of preds:', preds.shape)

#Plotting Results
plt.figure(figsize=(12, 12))

start_index = 0

for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    
    pred = np.argmax(preds[start_index+i])
    gt = y_test[start_index+i]
    
    col = 'b'
    
    if pred != gt:
        col = 'g'
        
    plt.xlabel('i={}, pred={}, gt={}'. format(start_index+i, pred, gt), color=col)
    plt.imshow(x_test[start_index+i], cmap='binary')
plt.show()

plt.plot(preds[8])
plt.show()