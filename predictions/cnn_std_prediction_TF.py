# -*- coding: utf-8 -*-
# @Time    : 28/1/2018 5:06 PM
# @Author  : Jason Lin
# @File    : cnn_std_prediction.py
# @Software: PyCharm Community Edition

import tensorflow as tf
import pickle as pkl
from sklearn.metrics import auc, roc_curve
import matplotlib.pylab as plt
from numpy.random import seed

seed(5)
from tensorflow import set_random_seed
set_random_seed(11)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 4, 1], padding='SAME')

def conv2d1(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 5, 1, 1],
                          strides=[1, 5, 1, 1], padding='SAME')


x = tf.placeholder(tf.float32, [None, 92])
y_ = tf.placeholder(tf.float32, [None, 2])

x_image = tf.reshape(x, [-1, 23, 4, 1])

W_filter_expan = weight_variable([1, 4, 1, 10])
b_filter_expan = bias_variable([10])

W_conv1_1 = weight_variable([1, 4, 1, 10])
b_conv1_1 = bias_variable([10])

W_conv1_2 = weight_variable([2, 4, 1, 10])
b_conv1_2 = weight_variable([10])

W_conv1_3 = weight_variable([3, 4, 1, 10])
b_conv1_3 = weight_variable([10])

W_conv1_5 = weight_variable([5, 4, 1, 10])
b_conv1_5 = weight_variable([10])


##### Batch Normalization #####
conv_layer_1 = conv2d(x_image, W_conv1_1) + b_conv1_1
conv_layer_2 = conv2d(x_image, W_conv1_2) + b_conv1_2
conv_layer_3 = conv2d(x_image, W_conv1_3) + b_conv1_3
conv_layer_5 = conv2d(x_image, W_conv1_5) + b_conv1_5

conv_layer = tf.concat([conv_layer_1, conv_layer_2, conv_layer_3, conv_layer_5], 3)

conv_mean, conv_var = tf.nn.moments(conv_layer,axes=[0, 1, 2])
scale2 = tf.Variable(tf.ones([23, 1, 40]))
beta2 = tf.Variable(tf.zeros([23, 1, 40]))
conv_BN = tf.nn.batch_normalization(conv_layer, conv_mean, conv_var, beta2, scale2, 1e-3)
h_conv1 = tf.nn.relu(conv_BN)

h_pool1 = max_pool_2x2(h_conv1)

W_fc1 = weight_variable([1 * 5 * 40, 100])
b_fc1 = bias_variable([100])

h_pool2_flat = tf.reshape(h_pool1, [-1, 1 * 5 * 40])

h_fc1 = tf.nn.softmax(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_dense_1 = weight_variable([100, 23])
b_dense_1 = bias_variable([23])
y_dense_1 = tf.nn.softmax(tf.matmul(h_fc1_drop, W_dense_1) + b_dense_1)

W_ouput = weight_variable([23, 2])
b_ouput = bias_variable([2])
y_conv = tf.nn.softmax(tf.matmul(y_dense_1, W_ouput) + b_ouput)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

min_batch = 100

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
saver = tf.train.Saver()

saver.restore(sess, "../CNN_std_model/cnn_all_train.ckpt")

# load test_data
# test_data = test[0]
# test_data_label = test[1][:, 0]

# predict off-target effect
# res = sess.run(y_conv, feed_dict={x: test_data, keep_prob: 1.0})
# result = res[:, 0]
