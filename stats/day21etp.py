# -*- coding: utf-8 -*-
# @Time    : 28/10/2017 1:18 AM
# @Author  : Jason Lin
# @File    : day21etp.py
# @Software: PyCharm Community Edition

import numpy as np
from openpyxl import load_workbook
import os
import csv
import re

print os.getcwd()

wb = load_workbook(filename = "../data/STable 18 CD33_OffTargetdata.xlsx")
sheets = wb.get_sheet_names()
sheet = sheets[0]

ws = wb.get_sheet_by_name(sheet)

rows = ws.rows

xa_1 = []
h = []

type={}
stats={}
annotation = []
value = []
for row in rows:
    na = str(row[3].value)
    na = na.replace(",", "_")
    tt = na.split(":")
    if len(tt) > 1:
        tt = tt[1]
        h.append(row[1].value)
        if tt == "A_1":
            xa_1.append(row[1].value)

    annotation.append(na)
    value.append(row[6].value)

annotation = np.array(annotation)
annotation = np.unique(annotation)


for ann in annotation:
    type[ann] = []
    stats[ann] = []

rows = ws.rows
for row in rows:
    v = row[6].value
    n = str(row[3].value)
    n = n.replace(",", "_")
    type[n].append(v)

for key, value in type.items():
    val = np.array(value)
    stats[key].append(key)
    stats[key].append(len(val))
    stats[key].append(len(val[val>0]))
    stats[key].append(len(val[val>1]))
    stats[key].append(len(val[val>2.2])*1.0/len(val[val>1]))
    stats[key].append(len(val[val<1])*1.0/len(val))


with open("exp_data.csv","wb") as csvfile:
    """
    s = ""
    for item in type.keys():
        s = s + str(item + ",")
    s = s + "\n"
    file.write(s)
    
    for key, value in type.items():
        s = str(key) + ","
        for v in value:
            s = s + str(v) + ","
        s = s + "\n"
        file.write(s)
    """
    fieldnames = ['Annotation', "Num", "Pos_0", "Pos_1", "Pos_0_rate", "Pos_1_rate"]
    writer = csv.writer(csvfile)
    writer.writerow(fieldnames)
    for key, _ in stats.items():
        writer.writerow(stats[key])

csvfile.close()


ii = np.unique(np.array(h))

print len(ii)
print ii


