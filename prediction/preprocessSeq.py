# -*- coding: utf-8 -*-
# @Time    : 21/11/2017 7:10 PM
# @Author  : Jason Lin
# @File    : preprocessSeq.py
# @Software: PyCharm Community Edition

import numpy as np
import pandas as pd
import pickle as pkl
import os
from sklearn.preprocessing import OneHotEncoder

class SeqPreProcessing:

    def __init__(self):
        os.getcwd()

    # Reverse complements a given string
    def revcom(self, s):
        basecomp = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'U': 'A'}
        letters = list(s[::-1])
        letters = [basecomp[base] for base in letters]
        return ''.join(letters)

    # Unpickle mismatch scores and PAM scores
    def get_mm_pam_scores(self):
        try:
            mm_scores = pkl.load(open('../CFDScoring/mismatch_score.pkl', 'rb'))
            pam_scores = pkl.load(open('../CFDScoring/pam_scores.pkl', 'rb'))
            return (mm_scores, pam_scores)
        except:
            raise Exception("Could not find file with mismatch scores or PAM scores")

    # Calculates CFD score
    def calc_cfd(self, wt, sg, pam):
        mm_scores, pam_scores = self.get_mm_pam_scores()
        score = 1
        sg = sg.replace('T', 'U')
        wt = wt.replace('T', 'U')
        s_list = list(sg)
        wt_list = list(wt)
        for i, sl in enumerate(s_list):
            if i >= 20:
                break
            if wt_list[i] == sl:
                score *= 1
            else:
                key = 'r' + wt_list[i] + ':d' + self.revcom(sl) + ',' + str(i+1)
                score *= mm_scores[key]
        # score *= pam_scores[pam]
        return float(score)

    def processTrainingData(self):
        mismatch_filename = "../data/mismatch_offtarget_noNC.csv"
        data = pd.read_csv(mismatch_filename)
        lfc_val = []
        siteCode = {"A": 1, "G": 2, "C": 3, "T": 4}
        inputX = []
        input2dim = []
        Y = []

        for idx, row in data.iterrows():
            wt = row.WTSequence
            off = row.MutatedSequence
            wtl = list(wt)
            offl = list(off)

            # Regression label
            lfc_val.append(float(row.etp))
            # Classification label
            if row.etp > np.log2(4.8):

                Y.append([1., 0.])
            else:
                Y.append([0., 1.])
            # cfd_score
            cfd_score = self.calc_cfd(wt, off, pam=None)

            # Generate the feature from sgRNA seq and off target seq
            wt_code = []
            off_code = []

            for idx in range(len(wtl)):
                wt_code.append(siteCode[wtl[idx]])
                off_code.append(siteCode[offl[idx]])

            wt_code = np.array(wt_code)
            off_code = np.array(off_code)
            misloc_code = wt_code - off_code
            mis_idx = np.where(misloc_code != 0.)
            # print mis_idx
            misloc_typecode = list(misloc_code)

            # mis_wt = np.array(wtl)[mis_idx][0]
            # mis_off = np.array(offl)[mis_idx][0]
            # mis_type = mis_wt + ":" + mis_off

            misloc_code[mis_idx] = 1.

            mul2seq_code = np.power(wt_code * off_code, 0.5)
            # onehot encode for misType
            # misType_dict = pkl.load(open("misType_dict.pkl", "rb"))
            # enc = OneHotEncoder()
            # enc.fit((np.array((misType_dict.values())) + 10).reshape(-1, 1))
            # misType_code = enc.transform([[misType_dict[mis_type] + 10]]).toarray().tolist()[0]
            # a input vector
            tem_code = []
            tem_code.append(cfd_score)
            # tem_code.append(float(mis_idx[0] + 1))
            # tem_code.append(misType_dict[mis_type])
            input_vec = list(mul2seq_code) + list(misloc_code) + misloc_typecode + tem_code
            print input_vec

            inputX.append(input_vec)

        # pkl.dump(misType_dict, open("misType_dict.pkl", "wb"))
        return np.array(inputX), np.array(Y), np.array(lfc_val)

    def processTestData(self):
        siteCode = {"A": 1, "G": 2, "C": 3, "T": 4}
        test_data = pkl.load(open("../data/test_data.pkl", "rb"))
        inputX = []

        for wtSeq, offs in test_data.items():
            wt = str(wtSeq)
            wtl = list(wt)
            for off in offs:
                offl = list(off)
                cfd_score = self.calc_cfd(wt, off, pam=None)
                # one sample (8,20)
                wt_code = []
                off_code = []
                for idx in range(20):
                    wt_code.append(siteCode[wtl[idx]])
                    off_code.append(siteCode[offl[idx]])

                wt_code = np.array(wt_code)
                off_code = np.array(off_code)
                misloc_code = wt_code - off_code
                mis_idx = np.where(misloc_code != 0.)
                # print mis_idx

                mis_wt = np.array(wtl)[mis_idx]
                mis_off = np.array(offl)[mis_idx]
                misloc_typecode = list(misloc_code)

                # mis_type = mis_wt + ":" + mis_off
                # misType_dict[mis_type] = misloc_code[mis_idx][0]
                misloc_code[mis_idx] = 1.

                mul2seq_code = np.power(wt_code * off_code, 0.5)
                # onehot encode for misType
                # misType_dict = pkl.load(open("misType_dict.pkl", "rb"))
                # enc = OneHotEncoder()
                # enc.fit((np.array((misType_dict.values())) + 10).reshape(-1, 1))
                # misType_code = enc.transform([[misType_dict[mis_type] + 10]]).toarray().tolist()[0]
                # a input vector
                tem_code = []
                tem_code.append(cfd_score)
                # tem_code.append(float(mis_idx[0] + 1))
                input_vec = list(mul2seq_code) + list(misloc_code) + misloc_typecode + tem_code
                inputX.append(input_vec)
                print input_vec

        return np.array(inputX)


def main():

    pre = SeqPreProcessing()
    pre.processTestData()


if __name__ == "__main__":
    main()