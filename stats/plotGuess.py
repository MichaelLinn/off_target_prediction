# -*- coding: utf-8 -*-
# @Time    : 28/10/2017 3:56 PM
# @Author  : Jason Lin
# @File    : plotGuess.py
# @Software: PyCharm Community Edition

import os
import pandas as pd
import csv
import numpy as np
import re

def filtering():
    os.getcwd()
    filename = "../data/offtarget.csv"

    df = pd.read_csv(filename)

    index = (df.pa > 6) & (df.pa < 60) & (df.etp > 1) & (df.Category != "PAM")
    # for ix, row in df.iterrows():

    new = df[index]

    new.to_csv("filter_offtarget.csv", index=False)


def countActivity():

    off = pd.read_csv("modifiedMismathData.csv")
    # print off

    type = {}
    for idx, row in off.iterrows():
        name = str(row.mut)
        type[name] = []

    for idx, row in off.iterrows():
        name = str(row.mut)
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
    with open("cfd_mismatch.csv","wb") as csvfile:
        wr = csv.writer(csvfile)
        wr.writerow(fieldname)
        for key, value in stats.items():
            wr.writerow(value)


def stats():
    off = pd.read_csv("../data/offtarget.csv")
    sum = 0
    sgRNA = []
    for idx, row in off.iterrows():
        if row.etp > np.log(4.8):
            if row.Category != "PAM":
                sgRNA.append(row.ConstructBarcode)

    print sgRNA
    print len(sgRNA)
    print len(np.unique(np.array(sgRNA)))

def findMismath():
    off = pd.read_csv("filter_offtarget.csv")
    index = (off.Category == "Mismatch")
    mismatchDF = off[index]
    mismatchDF.to_csv("mismatch_offdata.csv", index=False)

def changeAnnotationOfMismatch():
    mis_df = pd.read_csv("mismatch_offdata.csv")
    for idx, row in mis_df.iterrows():
        an = str(row.Annotation)
        anlist = re.split(",|:", an)
        print  anlist
        s = "r"+anlist[1]+":"+"d"+anlist[0]+","+str(anlist[2])
        mis_df.set_value(idx, "mut", s)

    print mis_df
    mis_df.to_csv("modifiedMismathData.csv", index=False)


def allMismatch():

    filename = "../data/offtarget.csv"

    df = pd.read_csv(filename)

    index = (df.Category == "Mismatch")
    # for ix, row in df.iterrows():

    new = df[index]

    new.to_csv("mismatchALLofftarget.csv", index=False)


countActivity()