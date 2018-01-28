# Deep Learning Models used for off-target predicitons in CRISPR-Cas 9 gene editing
This repository includes a deep feedforward neural network and a deep convolutional neural network for predicting the off-targets in CRISPR-Cas9 gene editing. Both of the deep learning models were trained using CRISPOR dataset, which counts with 26,034 putative off-targets.

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
<li><p>chromatin annotations.R : R script to get the ChromHMM and Segway</p></li>
<li><p>conservation.R : R script to get the PhyloP and PhastCons</p></li>
<li><p>Crispr_Data.csv : it stores the raw data including DNA sequences, sgRNA sequences, and features</p></li>
</ul>

---------------------------------------
Jiecong Lin

jieconlin3-c@my.cityu.edu.hk

January 27 2018
