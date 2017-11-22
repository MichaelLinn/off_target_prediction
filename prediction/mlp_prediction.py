# -*- coding: utf-8 -*-
# @Time    : 16/11/2017 5:28 PM
# @Author  : Jason Lin
# @File    : mlp_prediction.py
# @Software: PyCharm Community Edition

import tensorflow as tf
import numpy as np
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle as pkl
from numpy.random import seed
seed(5)
from tensorflow import set_random_seed
set_random_seed(11)

# Parameters
learning_rate = 0.001
training_epochs = 100
batch_size = 400
display_step = 1

# training_epochs = 100, batch_size = 400, 2_layer(60,50)  seed(5,11)

# Network Parameters
n_hidden_1 = 60 # 1st layer number of neurons
n_hidden_2 = 50 # 2nd layer number of neurons
# n_hidden_3 = 30
n_input = 60 # Data input
n_classes = 2 # Total classes

# tf Graph input
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_classes])

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    # 'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    # 'b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

def load_train_data():
    mismatch_filename = "../data/mismatch_offtarget_noNC.csv"
    data = pd.read_csv(mismatch_filename)

    siteCode = {"A": 0, "G": 1, "C": 2, "T": 3}
    input = []
    input2dim = []
    Y = []
    for idx, row in data.iterrows():
        wt = row.WTSequence
        off = row.MutatedSequence
        wtl = list(wt)
        offl = list(off)
        code = np.zeros((8, 20))
        flag = 0
        if row.etp > np.log2(4.8):
            flag = 1
            Y.append([1., 0.])
        else:
            Y.append([0., 1.])
        for idx in range(len(wtl)):
            code[siteCode[wtl[idx]]][idx] = 1
            code[siteCode[offl[idx]] + 4][idx] = 1
        input.append(code)
        input2dim.append(code.T.flatten())

        if flag == 1:
            for i in range(4):
                input.append(code)
                input2dim.append(code.T.flatten())
                Y.append([1., 0.])
    return np.array(input2dim), np.array(Y)


def load_test_data():
    siteCode = {"A": 0, "G": 1, "C": 2, "T": 3}
    test_data = pickle.load(open("../data/test_data.pkl", "rb"))
    input = []
    input2dim = []

    for wtSeq, offs in test_data.items():
        wt = str(wtSeq)
        wtl = list(wt)
        for off in offs:
            offl = list(off)
            # one sample (8,20)
            code = np.zeros((8, 20))
            for idx in range(20):
                code[siteCode[wtl[idx]]][idx] = 1
                code[siteCode[offl[idx]] + 4][idx] = 1
            input.append(code)
            input2dim.append(code.T.flatten())
    return  np.array(input2dim)


# Create model
def multilayer_perceptron(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])

    # layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Construct model
logits = multilayer_perceptron(X)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)
# Initializing the variables
init = tf.global_variables_initializer()

train_data= pkl.load(open("../data/ptrainData.pkl", "rb"))
y_ = pkl.load(open("../data/label.pkl", "rb"))
print train_data.shape
test_data = pkl.load(open("../data/ptestData.pkl", "rb"))


with tf.Session() as sess:
    sess.run(init)

    # Training cyscle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(len(train_data)/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x = train_data[i:(i+1)*batch_size]
            batch_y = y_[i:(i+1)*batch_size]
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([train_op, loss_op], feed_dict={X: batch_x, Y: batch_y})

            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost={:.9f}".format(avg_cost))
    print("Optimization Finished!")

    # Test model
    pred = tf.nn.softmax(logits)

    res = sess.run(pred, feed_dict={X: test_data})
    prob = res[:, 0]
    print prob

    mlp = {}
    t_data = pickle.load(open("../data/test_data.pkl", "rb"))
    idx = 0
    for wts, os in t_data.items():
        for o in os:
            mlp[o] = prob[idx]
            idx += 1

    pkl.dump(mlp, open("/Users/jieconlin3/Desktop/crispor/crisporPaper-master/CFD_Scoring/mlp_score_v4.pkl", "wb"))















