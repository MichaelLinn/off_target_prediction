# Deep Learning Models used for off-target predicitons in CRISPR-Cas 9 gene editing
This repository includes a deep feedforward neural network and a deep convolutional neural network for predicting the off-targets in CRISPR-Cas9 gene editing. Both of the deep learning models were trained using CRISPOR dataset.

# PUBLICATION
Please cite this paper if using our preditive model:

Jiecong, Lin. & Ka-Chun, Wong. (2018). Off-target predictions in CRISPR-Cas9 gene editing using deep learning (ECCB 2018 Proceeding Special Issue). Bioinformatics, 34(17), i656–i663. http://doi.org/10.1093/bioinformatics/bty554

# PREREQUISITE
The models for off-target predicitons were conducted by using Python 2.7.13 and TensorFLow v1.4.1. 
Following Python packages should be installed:
<ul>
<li><p>scipy</p></li>
<li><p>numpy</p></li>
<li><p>pandas</p></li>
<li><p>scikit-learn</p></li>
<li><p>TensorFlow</p></li>
</ul>

The Keras version of CNN were conducted by Python 3.6, TensorFlow 1.9.0, Keras 2.2.0.
Following Python packages should be installed:
<ul>
<li><p>scipy</p></li>
<li><p>numpy</p></li>
<li><p>pandas</p></li>
<li><p>Keras</p></li>
<li><p>TensorFlow</p></li>
</ul>


# CONTAINS:
<ul>
<li><p>CFDScoring/cfd-score-calculator.py : Python script to run CFD score </p></li>
<li><p>data/crispor_allscore.csv : The CRIPOR dataset used for model validation</p></li>
<li><p>CNN_std_model/cnn_all_train.ckpt.* : The final convolutional neural network model (CNN_std)</p></li>
<li><p>predictions/cnn_std_prediction.py : Python script to run final convolutional neural network (CNN_std)
</p></li>
</ul>

---------------------------------------
Jiecong Lin

jieconlin3-c@my.cityu.edu.hk

January 27 2018
