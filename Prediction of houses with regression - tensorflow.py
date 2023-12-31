import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from utils import *
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, LambdaCallback

%matplotlib inline
tf.logging.set_verbosity(tf.logging.ERROR)

print('Libraries imported.')




d0p00000f = pd.read_csv('data.csv', names = column_names)
df.head(5)

df.isna().sum()             #check missing data


df =df.iloc[:,1:]
df_norm = (df - df.mean())/df.std()
df_norm.head()                 #Data Normalisation

y_mean = df['price'].mean()
y_std = df['price'].std()

def convert_label_value(pred):
    return int(pred * y_std + y_mean)

print(convert_label_value(0.354146))         #Convert label value


x = df_norm.iloc[:, :6]
x.head()           #Selecting features


y = df_norm.iloc[:,-1]
y.head()           #Selecting labels



x_arr = x.values
y_arr = y.values

print('features array shape:', x_arr.shape)
print('labels array shape:', y_arr.shape)              #Feature and label values



x_train, x_test, y_train, y_test = train_test_split(x_arr, y_arr, test_size=0.05, random_state=0)
print('Training set:', x_train.shape, y_train.shape)
print('Test set:', x_test.shape, y_test.shape)



def get_model():
    model = Sequential([
        Dense(10, input_shape = (6,), activation='relu'),
        Dense(20, activation='relu'),
        Dense(5,activation='relu'),
        Dense(1)
    ])
    model.compile(
    loss='mse',
    optimizer='adam'
    )
    return model

get_model().summary()                            #Creating model


es_cb = EarlyStopping(monitor='val_loss', patience=5)

model = get_model()
preds_on_untrained = model.predict(x_test)

history = model.fit(
    x_train, y_train,
    validation_data = (x_test, y_test),
    epochs = 100,
    callbacks = [es_cb] 
)                                               #Model Training


plot_loss(history)


preds_on_trained = model.predict(x_test)
compare_predictions(preds_on_untrained, preds_on_trained, y_test)

price_untrained = [convert_label_value(y) for y in preds_on_untrained]
price_trained = [convert_label_value(y) for y in preds_on_trained]
price_test = [convert_label_value(y) for  y in y_test]

compare_predictions(price_untrained, price_trained, price_test)