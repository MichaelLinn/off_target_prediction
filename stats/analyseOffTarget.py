# -*- coding: utf-8 -*-
# @Time    : 1/11/2017 5:50 PM
# @Author  : Jason Lin
# @File    : analyseOffTarget.py
# @Software: PyCharm Community Edition

import pandas as pd
import numpy as np
import os
import csv
import matplotlib.pylab as plt
import pickle

os.getcwd()

def deletNegControl():

    filename = "../data/offtarget.csv"
    df = pd.read_csv(filename)

    df = df[df.TranscriptID == "ENST00000262262"]
    df = df[df.Category == "Mismatch"]
    df.to_csv("../data/mismatch_offtarget_noNC.csv", index=False)

def filterOfftarget():
    off = pd.read_csv("../data/mismatch_offtarget_noNC.csv")
    off = off[(off.pa > 6) & (off.pa < 60) & (off.etp > 1)]
    off.to_csv("../data/filtered_mismatch_offtarget_noNC.csv", index=False)

def countActivity():

    off = pd.read_csv("../data/filtered_mismatch_offtarget_noNC.csv")

    type = {}
    for idx, row in off.iterrows():
        name = str(row.Annotation)
        type[name] = []

    for idx, row in off.iterrows():
        name = str(row.Annotation)
        type[name].append(row.etp)

    stats = {}
    threshold = np.log2(4.8)
    for key, item in type.items():
        res = []
        num = np.array(item)
        res.append(str(key))
        res.append(len(num))
        posN = len(num[num > threshold])
        res.append(posN)
        res.append(1.0*posN/len(num))
        stats[key] = res

    fieldname = ['Annotation', "num", "pos_num", "pos_rate"]
    with open("../data/cfd_mismatch.csv","wb") as csvfile:
        wr = csv.writer(csvfile)
        wr.writerow(fieldname)
        for key, value in stats.items():
            wr.writerow(value)


def plotHist():
    mycfd = pd.read_csv("../data/cfd_mismatch.csv")
    cfd = pd.read_csv("../data/cfdScore.csv")
    data1 = mycfd.pos_rate
    data2 = cfd.cfd
    bins = np.arange(0, 1.0, 0.05)
    plt.hist(data1, bins=bins, alpha=0.5, label='mycfd')
    plt.hist(data2, bins=bins, alpha=0.5, label='cfd')
    plt.legend()
    plt.show()

#Reverse complements a given string
def revcom(s):

    basecomp = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A','U':'A'}
    letters = list(s[::-1])
    letters = [basecomp[base] for base in letters]
    return ''.join(letters)

def transferAnnotation():

    off = pd.read_csv("../data/cfd_mismatch.csv")
    for idx, row in off.iterrows():
        temStr = str(row.Annotation).split(",")[0]
        position = str(row.Annotation).split(",")[1]
        s = temStr.replace("T", "U")   # [WT Seq: Offtarget Seq]
        s = s.split(":")
        key = 'r' + s[0] + ':d' + revcom(s[1]) + "," + position
        off.set_value(idx,'Annotation', key)
    off.to_csv("../data/cfd_mismatch.csv", index=False)

def genMyCFD():

    off = pd.read_csv("../data/cfd_mismatch.csv")
    cfd = pickle.load(open("mismatch_score.pkl", "rb"))
    for idx, row in off.iterrows():
        cfd[row.Annotation] = row.pos_rate
    pickle.dump(cfd, open("mymisCFD.pkl", "wb"))



