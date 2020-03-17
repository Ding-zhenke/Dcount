# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 21:20:59 2020

@author: 31214
"""
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import matplotlib.pyplot as plt
import numpy as np
from layer import add_layer,input_BN
tf.set_random_seed(1)
np.random.seed(1)
x_data = np.linspace(0,40,1)
y_data = np.linspace(0,40,1)

x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])

l1 = add_layer(input_BN(x), 1,3,activation_function=tf.nn.relu,norm=True)
prediction =add_layer(l1,3,1,norm=True)

loss = tf.losses.mean_squared_error(y,prediction)
train = tf.train.AdamOptimizer(0.1).minimize(loss) 
sess = tf.Session()
sess.run(tf.global_variables_initializer())  # initialize var in graph


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data, y_data)
plt.ion()
plt.show()

for i in range(1000):
    # training
    sess.run(train, feed_dict={x: x_data, y: y_data})
    if i % 20 == 0:
        # to visualize the result and improvement
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        prediction_value = sess.run(prediction, feed_dict={x: x_data})
        # plot the prediction
        lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
        plt.pause(0.1)