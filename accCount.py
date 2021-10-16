#讀excel，算準確率。
import pandas as pd
import numpy as np

df = pd.read_excel("acc.xlsx")
nmp=df.to_numpy()

TN = 0
TP = 0
FN = 0
FP = 0
TN_Probability = []
TP_Probability = []
FN_Probability = []
FP_Probability = []

for i in range(0,len(nmp)):
    if(nmp[i][3] == "TN"):
        TN = TN + 1
        TN_Probability.append(nmp[i][2])
    elif(nmp[i][3] == "TP" or nmp[i][3] == "TP(重複)"):
        TP = TP + 1
        TP_Probability.append(nmp[i][2])
    elif(nmp[i][3] == "FN"):
        FN = FN + 1
        FN_Probability.append(nmp[i][2])
    elif(nmp[i][3] == "FP"):
        FP = FP + 1
        FP_Probability.append(nmp[i][2])

a = 0
b = 0
c = 0
d = 0
e = 0
f = 0

for i in FN_Probability:
    if i < 0.5:
        a = a + 1
    elif i >= 0.5 and i < 0.6:
        b = b + 1
    elif i >= 0.6 and i < 0.7:
        c = c + 1
    elif i >= 0.7 and i < 0.8:
        d = d + 1
    elif i >= 0.8 and i < 0.9:
        e = e + 1
    elif i >= 0.9:
        f = f + 1
print(a+b+c+d+e+f)