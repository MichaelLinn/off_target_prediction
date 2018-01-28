# -*- coding: utf-8 -*-
# @Time    : 1/23/18 8:59 PM
# @Author  : Jason Lin
# @File    : plot_guide.py
# @Software: PyCharm Community Edition

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
import random
import matplotlib.pylab as plt
import pickle as pkl
import scipy.cluster.vq as vq
from sklearn.metrics import roc_curve, auc
from scipy import interp
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

from numpy.random import seed
seed(5)
from tensorflow import set_random_seed
set_random_seed(17)


# Parameters
learning_rate = 0.001
training_epochs = 200
batch_size = 100
display_step = 1


# training_epochs = 100, batch_size = 430, 2_layer(60,50)  seed(5,11)  Softmax AUC = 0.919
# training_epochs = 100, batch_size = 430, 2_layer(160,80)  seed(5,11) Sigmod  AUC = 0.908

# Network Parameters
n_hidden_1 = 50 # 1st layer number of neurons
n_hidden_2 = 20 # 2nd layer number of neurons
n_hidden_3 = 10 # 3rd layer number of neurons
n_input = 92    # Data input
n_classes = 2   # Total classes

# tf Graph input
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_classes])

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    'out': tf.Variable(tf.random_normal([n_hidden_3, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


# Create model
def multilayer_perceptron(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])

    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_3, weights['out']) + biases['out']

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

saver = tf.train.Saver()

guideseq_data = pkl.load(open("../data/guideseq_encoded_data_fnn.pkl", "rb"))
test_data = guideseq_data[0]
test_data_label = np.array(guideseq_data[1])
# load test
test = pkl.load(open("../data/guideseq_encode_data_cnn.pkl", "rb"))
cfd_score = test[2]
# print test_data_label[:,0]

train = pkl.load(open("../data/crispor_encoded_data.pkl", "rb"))
train_data, train_data_l = train[0], train[1]
train_data_label = []
for l in train_data_l:
    if l == 0:
        train_data_label.append([0, 1])
    else:
        train_data_label.append([1, 0])

label = []

with tf.Session() as sess:
    sess.run(init)
    # Training cyscle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(len(train_data) / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x = train_data[i:(i + 1) * batch_size]
            batch_y = train_data_label[i:(i + 1) * batch_size]
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([train_op, loss_op], feed_dict={X: batch_x, Y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost={:.9f}".format(avg_cost))
    print("Optimization Finished!")

    pred = tf.nn.softmax(logits)
    res = sess.run(pred, feed_dict={X: test_data})
    prob = res[:, 0]
    print prob

    result = prob
    print result

    fpr, tpr, thresholds = roc_curve(test_data_label[:,0], result)

    fnn_res = [fpr, tpr]

    pkl.dump(fnn_res, open("../result/fnn_guidedata.pkl", "wb"))
    print "fnn result stored!"

    fnn_r = pkl.load(open("../result/fnn_guidedata.pkl", "rb"))
    fpr, tpr = fnn_r[0], fnn_r[1]
    auc_ = auc(fpr, tpr)
    print "auc", auc
    plt.plot(fpr, tpr,lw=2, alpha=.8, ls="--", label=r'FNN_3layer ROC (AUC = %0.3f)' % auc_)

# CNN
cnn_res = pkl.load(open("../result/cnn_guidedata.pkl", "rb"))
fpr, tpr = cnn_res[0], cnn_res[1]
auc_ = auc(fpr, tpr)
print "auc", auc_
plt.plot(fpr, tpr,lw=2, alpha=.8, color='r', label=r'CNN_std ROC (AUC = %0.3f)' % auc_)


#####################       logistic      ######################
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()
all_data = pkl.load(open("../data/crispor_encoded_data.pkl", "rb"))
train_data = all_data[0]
train_data_label = all_data[1]
# print all_data
clf.fit(X=train_data, y=train_data_label)
# test = pkl.load(open("../10_cross_validation/train_data_0.pkl", "rb"))
# test_data, test_data_label, pam_test, test_index = test[0], test[1], test[2], test[3]
res = clf.predict_proba(test_data)
prob = res[:, 1]
##############################################################
result = prob
# print result
# transferRes(result)
# pkl.dump(result, open("../data2/origuide_res.pkl", "wb"))
# mean_fpr = np.linspace(0, 1, 100)

fpr, tpr, thresholds = roc_curve(test_data_label[:, 0], result)
auc_ = auc(fpr, tpr)
print "auc", auc_
plt.plot(fpr, tpr, lw=2, alpha=.8, ls=':', label=r'Logistic Regression ROC (AUC = %0.3f)' % auc_)


#### Random Forest
clf_rf = RandomForestClassifier(max_depth=3, random_state=0, n_estimators=100)
clf_rf.fit(X=train_data, y=train_data_label)
# test = pkl.load(open("../10_cross_validation/train_data_0.pkl", "rb"))
# test_data, test_data_label, pam_test, test_index = test[0], test[1], test[2], test[3]
res = clf_rf.predict_proba(test_data)
prob = res[:, 1]
result = prob
print result
# transferRes(result)
# pkl.dump(result, open("../data2/origuide_res.pkl", "wb"))
# mean_fpr = np.linspace(0, 1, 100)

fpr, tpr, thresholds = roc_curve(test_data_label[:, 0], result)
auc_ = auc(fpr, tpr)
print "auc", auc_
plt.plot(fpr, tpr,lw=2, alpha=.8, ls="-.", label=r'Random Forest ROC (AUC = %0.3f)' % auc_)

# GDBT
clf_gt = GradientBoostingClassifier(n_estimators=200)
clf_gt.fit(X=train_data, y=train_data_label)
res = clf_gt.predict_proba(test_data)
prob = res[:, 1]
result = prob
print result

fpr, tpr, thresholds = roc_curve(test_data_label[:, 0], result)
auc_ = auc(fpr, tpr)
print "auc", auc_
plt.plot(fpr, tpr,lw=2, alpha=.8, label=r'GDT ROC (AUC = %0.3f)' % auc_)


# CFD score
fpr, tpr, thresholds = roc_curve(test_data_label[:, 0], cfd_score)
auc_ = auc(fpr, tpr)
print "auc", auc_
plt.plot(fpr, tpr,lw=2, alpha=.8, label=r'CFD score ROC (AUC = %0.3f)' % auc_)

plt.plot([0,1], [0,1], linestyle=':', lw=2, color='r', alpha=.8)
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic curve')
plt.legend(loc="lower right")
plt.savefig("../../compared_guidedata.eps", format='eps', dpi=5000)
plt.show()


