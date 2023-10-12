#Importing libraries 
import numpy as np

from tensorflow.keras.datasets import mnist
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping, LambdaCallback
from tensorflow.keras.utils import to_categorical

%matplotlib inline

#Data preprocessing
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float')/255.
x_test = x_test.astype('float')/255.
x_train = np.reshape(x_train, (60000, 784))
x_test = np.reshape(x_test, (10000, 784))

#Adding noise to training and test sets
x_train_noisy = x_train + np.random.rand(60000, 784)*0.9
x_test_noisy = x_test + np.random.rand(10000, 784)*0.9
x_train_noisy = np.clip(x_train_noisy, 0.,1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

def plot(x, p, labels=False):
    plt.figure(figsize=(20, 2))
    for i in range(10):
        plt.subplot(1, 10, i+1)
        plt.imshow(x[i].reshape(28, 28), cmap='binary')
        plt.xticks([])
        plt.yticks([])
        if labels:
            plt.xlabel(np.argmax(p[i]))
plt.show()


plot(x_train, None)

plot(x_train_noisy, None)

#Building and Training a Classifier
classifier = Sequential([
    Dense(256, activation='relu', input_shape=(784,)),
    Dense(256, activation='relu'),
    Dense(10, activation='softmax'),
])

classifier.compile(
   optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
classifier.fit(x_train, y_train, batch_size=512, epochs=3)

loss, acc = classifier.evaluate(x_test, y_test)
print(acc)

loss, acc = classifier.evaluate(x_test_noisy, y_test)
print(acc)

#Building autoencoder
input_image = Input(shape=(784,))
encoded = Dense(64, activation='relu')(input_image)
decoded = Dense(784, activation='sigmoid')(encoded)

autoencoder = Model(input_image, decoded)
autoencoder.compile(loss='binary_crossentropy', optimizer='adam')

#Training autoencoder
autoencoder.fit(
    x_train_noisy, x_train, epochs = 100,
    batch_size=512, validation_split=0.2, verbose=False,
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=5),
        LambdaCallback(on_epoch_end=lambda e, l: print('{:.3f}'.format(l['val_loss']), end='_'))
    ]
)

print(' _ ')
print('Training is completed')

#Denoised Images
predictions = autoencoder.predict(x_test_noisy)
plot(x_test_noisy, None)

plot(x_test_noisy, None)

loss, acc = classifier.evaluate(predictions, y_test)
print(acc)

#Composite Model
input_image = Input(shape=(784,))
x = autoencoder(input_image)
y = classifier(x)

denoise_and_classify = Model(input_image, y)

predictions = denoise_and_classify.predict(x_test_noisy)

plot(x_test_noisy, predictions, True)
plot(x_test, to_categorical(y_test), True)