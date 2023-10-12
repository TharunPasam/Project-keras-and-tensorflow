#Import Libraries
import tensorflow as tf
import numpy as np
import os

print('This notebook works with TensorFlow version:', tf.__version__)

folders = ['tmp', 'models', 'model_name', 'weights']
for folder in folders:
    if not os.path.isdir(folder):
        os.mkdir(folder)

print(os.listdir('.'))

#Creating Model
def create_cmodel():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, activation ='relu', input_shape = (784,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation = 'softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    return model

model = create_model()
model.summary()

#Data Preprocessing
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train = np.reshape(x_train, (x_train.shape[0], 784))/255.
x_test = np.reshape(x_test, (x_test.shape[0], 784))/255.

y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

#Model Checkpoint during training
checkpoint_dir = 'weights/'

_ = model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=2, batch_size=512,
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint(
           os.path.join(checkpoint_dir, 'epoch_{epoch:02d}_acc_{val_acc:3f}'),
            monitor='val_acc', save_weights_only = True, save_best_only=True
        )
    ]
)

os.listdir(checkpoint_dir)

#Load Weights
model = create_model()
print(model.evaluate(x_test, y_test, verbose=False))

#Saving complete model during training
models_dir = 'models/'

_ = model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=2, batch_size=512,
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint(
           os.path.join(models_dir, 'epoch_{epoch:02d}_acc_{val_acc:3f}.h5'),
            monitor='val_acc', save_weights_only = False, save_best_only=False        )
    ]
)

os.listdir(models_dir)

#Load Models
from file_with_methods import create_model
model = create_model()
print(model.evaluate(x_test, y_test, verbose=False))

model = tf.keras.models.load_model('models/epoch_02_acc_0.852500.h5')
print(model.evaluate(x_test, y_test, verbose=False))

#Manually Saving Weights and Models
model.save_weights('tmp/manual_saved.w')
os.listdir('tmp')

model.save('tmp/manually_saved.h5')
os.listdir('tmp')

#Exporting and Restoring SavedModel Format
model.save('model_name')
os.listdir('model_name/')

model = tf.keras.models.load_model('model_name')