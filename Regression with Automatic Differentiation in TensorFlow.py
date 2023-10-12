import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

print('Using TensorFlow version:', tf.__version__)
print('Devices available:', tf.config.list_physical_devices())

##Constants
tf.constant([1, 2, 3])
tf.convert_to_tensor ([1, 2, 3])
tf.convert_to_tensor ([1, 2, 3,], dtype = tf.float32)
tf.convert_to_tensor ([1, 2, 3]).numpy()

##Variables
tf.Variable([[1, 2, 3]])

v = tf.Variable(1)
print('Init value:', v.numpy())
v.assign(2)
print('Init value:', v.numpy())

c = tf.convert_to_tensor (np.random.randn(2, 3))                 #Converting to tensor
v = tf.Variable(np.random.randn(3, 1))
print(tf.matmul(c, v)) 

#Automatic Differentiation

y = tf.Variable(4.0)

with tf.GradientTape() as tape:
    x = y**3  
dx_dy = tape.gradient (x, y)
print('gradient at y={} is {}'.format(y.numpy(), dx_dy.numpy()))

#For higher gradient

y = tf.Variable(6.0)
with tf.GradientTape() as t1:
    with tf.GradientTape() as t2:
        x = y**3
    dx_dy = t2.gradient (x, y)
d2x_dy2 = t1.gradient (dx_dy, y)
print('2nd order gradient at y={} is {}'.format(y.numpy(), d2x_dy2.numpy()))

#Using constant in place of variable gives us output as none
x = tf.constant(4.0)
with tf.GradientTape() as tape:
    y = x**3  
dy_dx = tape.gradient (y, x)
print(dy_dx)


#we use the command tape.watch() to get the output
x = tf.constant(4.0)
with tf.GradientTape() as tape:
    tape.watch(x)
    y = x ** 3
dy_dx = tape.gradient(y, x)
print (dy_dx)

#Use of Persistent Tape
y= tf.Variable (3.0)
with tf.GradientTape(persistent=True) as tape :
    x = y ** 3
    z = 2 * x    
dz_dx = tape.gradient(z, x)
dx_dy = tape.gradient(x, y)
dz_dy = tape.gradient(z, y)
del tape
print('dz_dx', dz_dx.numpy())
print('dy_dx', dy_dx.numpy())
print('dz_dy', dz_dy.numpy())


#Generating data for linear Regression  (y = wx+b)
true_w, true_b = 7., 4.

def create_batch(batch_size = 64):
    x = np.random.randn(batch_size, 1)
    y = np.random.randn(batch_size, 1) + true_w * x + true_b
    return x, y

x, y = create_batch()
plt.plot(x,y, '.')            #Plotting the data


#Linear Regression
iterations = 100
lr = 0.03

w_history = []
b_history = []

w = tf.Variable(10.0)
b = tf.Variable(2.0)
 
for i in range(0, iterations):
    x_batch, y_batch = create_batch()
    x_batch = tf.convert_to_tensor(x_batch, dtype=tf.float32)
    y_batch = tf.convert_to_tensor(y_batch, dtype=tf.float32)
    with tf.GradientTape(persistent=True) as tape:
        y = b + w * x_batch
        loss = tf.reduce_mean(tf.square(y - y_batch))
    dw = tape.gradient(loss, w)
    db = tape.gradient(loss, b)
    
    del tape
    
    w.assign_sub(lr*dw)
    b.assign_sub(lr*db)
     
    w_history.append(w.numpy())
    b_history.append(b.numpy())
    
    if i%10 == 0:
        print('Iter {}, w={}, b={}'.format(i, w.numpy(), b.numpy()))
    



plt.plot(range(iterations), w_history, label='Learned w')
plt.plot(range(iterations), b_history, label='Learned b')
plt.plot(range(iterations), [true_w]*iterations, label='True w')
plt.plot(range(iterations), [true_b]*iterations, label='True b')
plt.legend()

