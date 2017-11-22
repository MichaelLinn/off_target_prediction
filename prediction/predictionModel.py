# -*- coding: utf-8 -*-
# @Time    : 22/11/2017 1:40 PM
# @Author  : Jason Lin
# @File    : predictionModel.py
# @Software: PyCharm Community Edition

import numpy as np
import pickle as pkl
import os
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.gaussian_process.kernels import WhiteKernel, ExpSineSquared

class PredictionModel:

    def __init__(self):
        os.getcwd()
        self.test_data = pkl.load(open("../data/ptestData.pkl", "rb"))
        self.train_data = pkl.load(open("../data/ptrainData.pkl", "rb"))
        self.label = pkl.load(open("../data/label.pkl", "rb"))
        self.lfc_val = pkl.load(open("../data/logfoldchange.pkl", "rb"))

    def randomForestModel(self):
        cls = RandomForestClassifier(n_estimators=1000)
        reg = RandomForestRegressor(n_estimators=1000, max_features=3, max_depth=2)
        cls.fit(X=self.train_data, y=self.label[:,0])
        cls_res = cls.predict_proba(X=self.test_data)
        reg.fit(X=self.train_data, y=self.lfc_val)
        reg_res = reg.predict(X=self.test_data)
        # label = cls.predict(X=self.test_data)
        print cls_res[:, 1]
        print reg_res
        # print label
        return cls_res[:, 1], reg_res

    def gaussianProcessMpdel(self):
        cls = GaussianProcessClassifier()


    def export_result(self, result):
        t_data = pkl.load(open("../data/test_data.pkl", "rb"))
        idx = 0
        off_prob = {}
        for wts, offs in t_data.items():
            for off in offs:
                off_prob[off] = result[idx]
                idx += 1
        # print off_prob.items()
        pkl.dump(off_prob, open("/Users/jieconlin3/Desktop/crispor/crisporPaper-master/CFD_Scoring/rf_score2.pkl", "wb"))

def main():
    pre = PredictionModel()
    clsRes, regRes = pre.randomForestModel()
    pre.export_result(regRes)

if __name__ == '__main__':
    main()
