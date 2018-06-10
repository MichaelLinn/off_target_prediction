# Deep Learning Models used for off-target predicitons in CRISPR-Cas 9 gene editing
This repository includes a deep feedforward neural network and a deep convolutional neural network for predicting the off-targets in CRISPR-Cas9 gene editing. Both of the deep learning models were trained using CRISPOR dataset, which counts with 26,034 putative off-targets.

# Publication
Please cite this paper if using our preditive model:

Jiecong, Lin. & Ka-Chun, Wong. (2018). Off-target predictions in CRISPR-Cas9 gene editing using deep learning (ECCB 2018 Proceeding Special Issue). Bioinformatics

# PREREQUISITE
SynergizingCRISPR was conducted by using Python 2.7.13 version and TensorFLow v1.4.1. 
Following Python packages should be installed:
<ul>
<li><p>scipy</p></li>
<li><p>numpy</p></li>
<li><p>pandas</p></li>
<li><p>scikit-learn</p></li>
<li><p>TensorFlow</p></li>
</ul>


# CONTAINS:
<ul>
<li><p>CFDScoring/cfd-score-calculator.py : Python script to run CFD score </p></li>
<li><p>data/crispor_allscore.csv : The CRIPOR dataset used for model training</p></li>
<li><p>data/GUIDE-seqData.csv : The GUIDE-seq used for model testing</p></li>
<li><p>CNN_std_model/cnn_all_train.ckpt.* : The final convolutional neural network model (CNN_std)</p></li>
<li><p>predictions/plot_guide_roc.py : Python script to test three traditional machine learning models, deep feedforward neural network (FNN_3layer), convolutional neural network (CNN_std) and CFD score on GUIDE-seq dataset and plot the results
<li><p>predictions/cnn_std_prediction.py : Python script to run final convolutional neural network (CNN_std) on GUIDE-seq independently
</p></li>
</ul>

---------------------------------------
Jiecong Lin

jieconlin3-c@my.cityu.edu.hk

January 27 2018
