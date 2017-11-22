# -*- coding: utf-8 -*-
# @Time    : 2/11/2017 3:58 PM
# @Author  : Jason Lin
# @File    : offtargetPrediction.py
# @Software: PyCharm Community Edition

import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import GradientBoostingRegressor
import os
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm
import tensorflow
from sklearn.neural_network import MLPClassifier
import matplotlib.pylab as plt



os.getcwd()
class OfftargetScorePrediction:

    def __init__(self):
        mismatch_filename = "../data/filtered_mismatch_offtarget_noNC.csv"
        self.data = pd.read_csv(mismatch_filename)

    def oneHotEncode(self):
        siteCode = {"A": 0, "G": 1, "C": 2, "T": 3}
        input = []
        input2dim = []
        Y = []
        for idx, row in self.data.iterrows():
            wt = row.WTSequence
            off = row.MutatedSequence
            wtl = list(wt)
            offl = list(off)
            code = np.zeros((8, 20))
            if row.etp > np.log2(4.8):
                Y.append(1.0)
            else:
                Y.append(0.0)
            for idx in range(len(wtl)):
                code[siteCode[wtl[idx]]][idx] = 1
                code[siteCode[offl[idx]]+4][idx] = 1
            input.append(code)
            input2dim.append(code.flatten())

        return  np.array(input2dim), np.array(input), np.array(Y)

    def halfOneHotEncode(self):
        siteCode = {"A": 0, "G": 1, "C": 2, "T": 3}

        input = []
        Y = []
        for idx, row in self.data.iterrows():
            wt = row.WTSequence
            off = row.MutatedSequence
            wtl = list(wt)
            offl = list(off)
            code = np.zeros((8, 20))
            if row.etp > np.log2(4.8):
                Y.append(1.0)
            else:
                Y.append(0.0)
            for idx in range(len(wtl)):
                code[siteCode[wtl[idx]]][idx] = 1
                code[siteCode[offl[idx]] + 4][idx] = 1
            input.append(code)

        return np.array(input), np.array(Y)

    def normalEncode(self):
        siteCode = {"A":1.0, "G":2.0, "C":3.0, "T":4.0}
        matchCode = {"A:C":1.0, "A:G":2.0, "A:T":3.0, "C:A":4.0, "C:T":5.0, "C:G":6.0, "G:A":7.0, "G:C":8.0, "G:T":9.0, "T:A":10.0, "T:C":11.0, "T:G":12.0}

        data_code = []
        data_ori = []
        Y = []
        for idx, row in self.data.iterrows():
            # print "--------",str(idx),"--------"
            tem = str(row.Annotation)
            type = tem.split(",")[0]
            pos = tem.split(",")[1]
            wt = type.split(":")[0]
            off = type.split(":")[1]
            data_code.append([siteCode[wt], siteCode[off], pos, matchCode[type]])
            # data_code.append([siteCode[wt], siteCode[off], pos])
            data_ori.append([wt, off, pos, type])
            if row.etp > np.log2(4.8):
                Y.append(1)
            else:
                Y.append(0)
        self.siteCode = siteCode
        self.matchCode = matchCode


        label = np.array(Y)
        # Y = np.array(self.data.etp)
        data = np.array(data_code).astype(np.float64)
        Y = np.array(self.data.etp).astype(np.float64)
        return data, Y, label

    def load_test_data(self):

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
                input2dim.append(code.flatten())


        return np.array(input), np.array(input2dim)

    def svmClassifier(self):
        data ,_ , label = self.normalEncode()
        cls = svm.SVC()
        cls.fit(data, label)
        return cls

    def rfClassifier(self):
        data ,_ , label = self.normalEncode()
        cls = RandomForestClassifier(n_estimators=500, random_state=0)
        cls.fit(data, label)
        return cls

    def erfClassifier(self):
        data, _, label = self.normalEncode()
        cls = ExtraTreesClassifier(n_estimators=500, max_depth=None, random_state=0)
        cls.fit(data, label)
        return cls

    def gradientBoostClassifier(self):
        data, _, label = self.normalEncode()
        cls = GradientBoostingClassifier(n_estimators=500, max_depth=None, random_state=0)
        cls.fit(data, label)
        return cls

    def adaboostClassifier(self):
        data, _, label = self.normalEncode()
        cls = AdaBoostClassifier(n_estimators=500, random_state=0)
        cls.fit(data, label)
        return cls

    def mlpClassifier(self):
        data, _, label = self.normalEncode()
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes = (5, 2), random_state = 1)
        print data
        print label
        clf.fit(data, label)
        return clf

    def gbRegression(self):
        data, Y,_ = self.normalEncode()
        reg = GradientBoostingRegressor(n_estimators=500, learning_rate=0.1, random_state=0, loss='ls')
        reg.fit(data, Y)
        return reg

    def bayseRgression(self):
        data, Y,_ = self.normalEncode()
        reg = linear_model.BayesianRidge()
        reg.fit(data, Y)
        return reg

    def randomforestRegression(self):
        data, Y ,_= self.normalEncode()
        reg = RandomForestRegressor(max_depth=1, random_state=0, n_estimators=500)
        reg.fit(data, Y)
        return reg

    def getOldCfdScore(self):
        print os.getcwd()
        misCFD = "../CFDScoring/mismatch_score.pkl"
        mapCFD = pickle.load(open(misCFD,"rb"))
        print mapCFD
        return mapCFD

    def revcom(self, s):
        basecomp = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'U': 'A'}
        letters = list(s[::-1])
        letters = [basecomp[base] for base in letters]
        return ''.join(letters)

    def genNewCFD(self):
        newMapCFD = self.getOldCfdScore()
        # print len(newMapCFD)
        reg = self.rfClassifier()
        for key, code in self.siteCode.items():
            for k, c in self.siteCode.items():
                if key == k:
                    continue
                for i in range(1, 21):
                    input = [[self.siteCode[key], self.siteCode[k], i, self.matchCode[str(key + ":"+ k)]]]
                    # input = [[self.siteCode[key], self.siteCode[k], i]]
                    wt = key.replace("T", "U")
                    off = k.replace("T", "U")

                    res = reg.predict_proba(input)
                    pos = str(i)
                    off = self.revcom(off)
                    newKey = "r" + wt + ":d" + off + "," + pos
                    newMapCFD[newKey] = float(res[0][1])
        print newMapCFD
        return newMapCFD

    def mlpPredictionOneHot(self):
        data2, data, label = self.oneHotEncode()
        cls = MLPClassifier(solver='lbfgs', alpha=1e-5, random_state = 1, hidden_layer_sizes=(100,))
        # print cls
        test, test2 = self.load_test_data()
        cls.fit(data2, label)
        score = cls.predict_proba(test2)
        return score




cc = OfftargetScorePrediction()
a,b = cc.load_test_data()
print b.shape
print a.shape
