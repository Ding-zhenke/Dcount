# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 13:35:24 2020

@author: 31214
"""
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.set_random_seed(1)


def add_layer(inputs, in_size, out_size, activation_function=None, norm=False):
    global fc_mean,fc_var
    # weights and biases (bad initialization for this case)
    Weights = tf.Variable(tf.random_normal([in_size, out_size], mean=0., stddev=1.))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    scale = tf.Variable(tf.ones([1]))
    shift = tf.Variable(tf.zeros([1]))
    epsilon = 0.001
    # fully connected product
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    # normalize fully connected product
    if norm:
        # Batch Normalize
        fc_mean, fc_var = tf.nn.moments(
            Wx_plus_b,
            axes=[0],   # the dimension you wanna normalize, here [0] for batch
            # for image, you wanna do [0, 1, 2] for [batch, height, width] but not channel
        )
        scale = tf.Variable(tf.ones([out_size]))
        shift = tf.Variable(tf.zeros([out_size]))
        epsilon = 0.001

    # apply moving average for mean and var when train on batch
        ema = tf.train.ExponentialMovingAverage(decay=0.5)
        def mean_var_with_update():
            ema_apply_op = ema.apply([fc_mean, fc_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(fc_mean), tf.identity(fc_var)
        mean, var = mean_var_with_update()
    
        Wx_plus_b = tf.nn.batch_normalization(Wx_plus_b, mean, var, shift, scale, epsilon)
        # similar with this two steps:
        # Wx_plus_b = (Wx_plus_b - fc_mean) / tf.sqrt(fc_var + 0.001)
        # Wx_plus_b = Wx_plus_b * scale + shift

    # activation
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)

    return outputs
def input_BN(xs):
    # BN for the first input
    fc_mean, fc_var = tf.nn.moments(
        xs,
        axes=[0],
    )
    scale = tf.Variable(tf.ones([1]))
    shift = tf.Variable(tf.zeros([1]))
    epsilon = 0.001
    # apply moving average for mean and var when train on batch
    ema = tf.train.ExponentialMovingAverage(decay=0.5)
    def mean_var_with_update():
        ema_apply_op = ema.apply([fc_mean, fc_var])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(fc_mean), tf.identity(fc_var)
    mean, var = mean_var_with_update()
    xs = tf.nn.batch_normalization(xs, mean, var, shift, scale, epsilon)
    return xs
