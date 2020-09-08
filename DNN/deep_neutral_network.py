import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

###Create Data
learning_rate = 0.1
training_steps = 1000
display_step = 100
n_samples = 50


X = np.random.rand(n_samples).astype(np.float32)
Y = X * 10 + 5
W = tf.Variable(tf.random.normal([1]))
b = tf.Variable(tf.zeros([1]))


def linear_regression(X):
    return W * X + b


def mean_square(y_pred, y_true):
    return tf.reduce_sum(tf.pow((y_pred-y_true),2)/(n_samples))

optimizer = tf.optimizers.SGD(learning_rate)


def run_optimization():
    with tf.GradientTape() as g:
        pred = linear_regression(X)
        loss = mean_square(pred , Y)
        gradients = g.gradient(loss, [W,b])
    
    optimizer.apply_gradients(zip(gradients, [W,b]))


for step in range(1, training_steps + 1):
    run_optimization()
    if step % display_step == 0:
         pred = linear_regression(X)
         loss = mean_square(pred , Y)
         print("step: %i, loss: %f, W: %f, b: %f" % (step, loss, W.numpy(), b.numpy()))



plt.plot(X, Y, 'ro', label='Original data')
plt.plot(X, np.array(W * X + b), label='Fitted line')
plt.legend()
plt.show()
