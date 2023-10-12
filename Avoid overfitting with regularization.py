#Importing data
from tensorflow.python.keras.datasets import fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
import matplotlib.pyplot as plt
%matplotlib inline

plt.figure(figsize = (10, 10))
for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.imshow(x_train[i], cmap = 'binary')
    
plt.show()

#Processing data
 from tensorflow.python.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
    
print(y_train.shape, y_test.shape)
print(y_train[0])

import numpy as np
x_train = np.reshape(x_train, (60000, 28 * 28))
x_test = np.reshape(x_test, (10000, 28 * 28))

x_train = x_train/255.
x_test = x_test/255.

#Creating experiment
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.regularizers import l2

nodes = 64

def create_model(w_reg = False, d_reg = False):
    model = Sequential()
    if w_reg:
        model.add(Dense(nodes, activation = 'relu', input_shape = (784,), kernel_regularizer = l2(0.009)))
        model.add(Dense(nodes, activation = 'relu', kernel_regularizer = 12(0.009)))
    else:
        model.add(Dense(nodes, activation = 'relu', input_shape = (784,)))
        model.add(Dense(nodes, activation = 'relu'))
    if d_reg:
        model.add(Dropout(0.2))
        
    model.add(Dense(10, activation = 'softmax'))
    model.compile(
        loss = 'categorical_crossentropy',
        optimizer = 'adam',
        metrics = ['accuracy']
    )
    model.summary()
    return model


def show_acc(h, E):
    plt.plot(range(E), h.history['acc'], label = 'Training')
    plt.plot(range(E), h.history['val_acc'], label = 'Validation')
    plt.ylim([0.7, 1.0])
    plt.legend()
    plt.show()
    return

from tensorflow.python.keras.callbacks import LambdaCallback

simple_log = LambdaCallback(
     on_epoch_end = lambda e, l: print(e, end = '.')
)

def run_experiment(E = 20, w_reg = False, d_reg = False):
    m = create_model(w_reg, d_reg)
    h = m.fit(
         x_train, y_train,
         epochs = E, verbose = False,
         validation_data = (x_test, y_test),
         callbacks = [simple_log]
    )
    
    show_acc(h, E)
    return


#Result
run_experiment()
