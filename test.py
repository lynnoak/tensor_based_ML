# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 11:09:57 2017

@author: victor
"""
import sys
sys.path.append("/Users/lynnoak/Documents/work/RDML/tensor_based_ML/scikit_tensor")

import numpy as np
import scipy
import scipy.sparse as sp
import sklearn
from sklearn import preprocessing
from metric_learn import LMNN,ITML_Supervised,LSML_Supervised,NCA,RCA_Supervised

import sktensor
from sktensor.rescal import als as rescal

from src.mydataset import *
from src.metric_tools import *
from src.print_tools import *



"For  PSD cone (not sure)"
def nearPSD(A,epsilon=0):
   n = A.shape[0]
   eigval, eigvec = np.linalg.eig(A)
   val = np.matrix(np.maximum(eigval,epsilon))
   vec = np.matrix(eigvec)
   T = 1/(np.multiply(vec,vec) * val.T)
   T = np.matrix(np.sqrt(np.diag(np.array(T).reshape((n)) )))
   B = T * vec * np.diag(np.array(np.sqrt(val)).reshape((n)))
   out = B*B.T
   return(out)

def TB_S_ML(X, Y, T, p=0.6):
    margin = 1
    n = len(X)
    d = len(X[0])
    nr = len(T)
    lamd = 0.5
    maxNumIts = 100
    consNum = 200
    relconsNum = int(consNum * p / nr) + 1
    labconsNum = int(consNum * (1 - p) / nr) + 1
    print(relconsNum, labconsNum)

    Xm = np.matrix(X)

    M = np.matrix(np.ones([d, d]))
    for t in range(maxNumIts):
        eta = 1 / (lamd * (t + 1))
        C = sp.csr_matrix((n, n))
        for r in range(nr):
            Cr = sp.csr_matrix((n, n))
            ns = 0
            nb = 0
            for b in range(consNum):

                if nb <= relconsNum:
                    [row, col] = T[r].nonzero()
                    idi = np.random.randint(len(row))
                    i = row[idi]
                    j = col[idi]
                    k = np.random.choice(np.where(T[r].getrow(i).toarray() != 1)[1])
                else:
                    if nb <= relconsNum + labconsNum:
                        i = np.random.randint(len(Y))
                        j = np.random.choice(np.where(Y == Y[i])[0])
                        k = np.random.choice(np.where(Y != Y[i])[0])
                    else:
                        break
                nb = nb + 1

                XiM = M * Xm[i, :].T
                XjM = M * Xm[j, :].T
                XkM = M * Xm[k, :].T
                Kii = Xm[i, :] * XiM
                Kjj = Xm[j, :] * XjM
                Kkk = Xm[k, :] * XkM
                Kij = Xm[j, :] * XiM
                Kji = Kij
                Kik = Xm[k, :] * XiM
                Kki = Kik
                dis_ij = Kii + Kjj - Kij - Kji
                dis_ik = Kii + Kkk - Kik - Kki
                if (dis_ij - dis_ik + margin >= 0):
                    Cr = Cr + sp.csr_matrix(([1, -1, -1, 1, 1, -1], ([j, i, j, i, k, k], [j, j, i, k, i, k])),
                                            shape=(n, n))
                    ns = ns + 1

            if ns != 0:
                Cr = (1 / ns) * Cr
            C = C + Cr
        C = (1 / nr) * C
        grad = Xm.T * C * Xm + lamd * M
        M = M - eta * grad
        M = nearPSD(M)

    print(M)
    K = 5
    S_TB_S = ComputeKNNScoreLF(X, Y, K, M, dim=d)
    return S_TB_S[0]


X,Y,T = data_elite()

S_TB_S = []
for p in [0,0.2,0.4,0.6,0.8,1.0]:
    try:
        S_TB_S.append(TB_S_ML(X,Y,T,p))
    except:
        S_TB_S.append(0)
for i in S_TB_S:
    print("\nTensor base with labels: %0.4f  "%(i))



