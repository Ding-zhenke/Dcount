# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 16:29:42 2020

@author: 31214
"""
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import matplotlib.pyplot as plt
import numpy as np
from layer import add_layer,input_BN
import saleh as sh
tf.set_random_seed(1)
np.random.seed(1)


def creat_signal(altitude,f,phase=0,paint=False):
    x_data = np.linspace(-1,1,300)[:, np.newaxis] 
    noise = np.random.normal(0, 0.5, x_data.shape)
    signal = altitude*np.sin(f*x_data+phase)+noise
    if paint == True:      
        plt.title("initial signal and process signal")
        plt.xlabel("x axis:")
        plt.ylabel("y axis:")
        plt.scatter(x_data,signal)
        plt.show()
    return signal

# create study data
x_data = np.linspace(-2,2,900)[:, np.newaxis]
sig = 2*np.sin(10*x_data)+np.random.normal(0, 0.5, x_data.shape)
tf_x = tf.placeholder(tf.float32, [None, 1])
tf_y = tf.placeholder(tf.float32, [None, 1])

#create neuron network with BN
l1 = add_layer(input_BN(tf_x), 1,5,activation_function=tf.nn.relu,norm=True)
l2 = add_layer(l1, 5,7,activation_function=tf.nn.relu,norm=True)
l3 = add_layer(l2, 7,9,activation_function=tf.nn.relu,norm=True)
l4 = add_layer(l3, 9,7,activation_function=tf.nn.relu,norm=True)
l5 = add_layer(l4, 7,5,activation_function=tf.nn.relu,norm=True)
prediction =add_layer(l5,5,1,norm=True)
#prediction = sh.saleh_model(prediction)
#loss = tf.reduce_mean(tf.reduce_sum(tf.square(tf_y-prediction), reduction_indices=[1]))
loss = tf.losses.mean_squared_error(tf_y,prediction) 
train = tf.train.AdamOptimizer(0.1).minimize(loss)
saver = tf.train.Saver()  # define a saver for saving and restoring
sess = tf.Session()
sess.run(tf.global_variables_initializer())  # initialize var in graph
#saver.restore(sess, './datak')

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data, sig)
plt.ion()
plt.show()

for i in range(1000):
    # training
    sess.run(train, feed_dict={tf_x: x_data, tf_y: sig})
    if i % 20 == 0:
        # to visualize the result and improvement
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        prediction_value = sess.run(prediction, feed_dict={tf_x: x_data})
        # plot the prediction
        lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
        plt.pause(0.1)
#saver.save(sess, './datak', write_meta_graph=False)  # meta_graph is not recommended

    