# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 11:09:57 2017

@author: victor
"""
import sys
sys.path.append("./src")

import numpy as np
import scipy
import scipy.sparse as sp
import sklearn
from metric_learn import LMNN,ITML_Supervised,LSML_Supervised


import sktensor
from sktensor.rescal import als as rescal

from mydataset import *
from metric_tools import *
from print_tools import *

"""
Learning with the rescal
"""

rank  = 5

X,Y,T = data_elite()

A, R, fval, iter, exectimes = rescal(T, rank)

X = np.column_stack((X,A))
Y = Y
K = 5

Mah_score = ComputeKNNScore(X,Y,K,1,title = "Mah")
"""Mah Accuracy : 0.3969 (+/- 0.0748)""" 	
"Mah Accuracy : 0.4005 (+/- 0.0261)"
"elite"
"Mah Accuracy : 0.7939 (+/- 0.1852)"

Eud_score = ComputeKNNScore(X,Y,K,2,title = "Eud")
"""Eud Accuracy : 0.3895 (+/- 0.0822)"""
"Eud Accuracy : 0.3982 (+/- 0.0232)"
"elite"
"Eud Accuracy : 0.7948 (+/- 0.1853))"

#lmnn = LMNN(learn_rate=1e-6)
#lmnn.fit(X,Y)
#XL = lmnn.transform(X)
#S_LMNN = ComputeKNNScore(XL,Y,K,2,title = "LMNN")
"""AssertionError: 
not enough class labels for specified k (smallest class has 1)"""

#itml = ITML_Supervised(num_constraints=100)
#itml.fit(X,Y)
#XI = itml.transform(X)
#S_ITML = ComputeKNNScore(XI,Y,K,2,title = "ITML")
#"""ITML Accuracy : 0.3515 (+/- 0.0226)"""
#"ITML Accuracy : 0.3850 (+/- 0.0173)"

lsml = LSML_Supervised(num_constraints=100)
lsml.fit(X,Y)
XL = lsml.transform(X)
S_LSML = ComputeKNNScore(XL,Y,K,2,title = "LSML")
"""LSML Accuracy : 0.4160 (+/- 0.0461)"""
"LSML Accuracy : 0.3899 (+/- 0.0144)"
"elite"
"ELSML Accuracy : 0.7986 (+/- 0.1874)"
#
#def CompareScore(Y,T,rank):
#    A, R, fval, iter, exectimes = rescal(T, rank)    
#
#    X = A
#    Y = Y[:,1]
#    K = 5    
#
#    Mah_score = ComputeKNNScore(X,Y,K,1,title = "Mah")
#    S_Mah =  Mah_score[0]	    
#
#    Eud_score = ComputeKNNScore(X,Y,K,2,title = "Eud")
#    S_Eud = Eud_score[0]
#
##    S_ITML = []
##    for i in range(10):
##        itml = ITML_Supervised(num_constraints=100)
##        itml.fit(X,Y)
##        XI = itml.transform(X)
##        t = ComputeKNNScore(XI,Y,K,2,title = "ITML") 
##        S_ITML.append(t[0])
##    S_ITML = np.mean(S_ITML)
##
##    S_LSML = []
##    for i in range(10):
##        lsml = LSML_Supervised(num_constraints=100)
##        lsml.fit(X,Y)
##        XL = lsml.transform(X)
##        t = ComputeKNNScore(XL,Y,K,2,title = "LSML")
##        S_LSML.append(t[0])
##    S_LSML = np.mean(S_LSML)
#
#    itml = ITML_Supervised(num_constraints=100)
#    itml.fit(X,Y)
#    XI = itml.transform(X)
#    S_ITML = ComputeKNNScore(XI,Y,K,2,title = "ITML")[0]
# 
#    lsml = LSML_Supervised(num_constraints=100)
#    lsml.fit(X,Y)
#    XL = lsml.transform(X)
#    S_LSML = ComputeKNNScore(XL,Y,K,2,title = "LSML")[0]
#
#    
#    return S_Mah,S_Eud,S_ITML,S_LSML
##
#Y,T = data_nation()
#S_Mah = []
#S_Eud = []
#S_ITML = []
#S_LSML = []    
#for r in range(2,len(T)-1):
#    m,e,i,l = CompareScore(Y,T,r)
#    S_Mah.append(m)
#    S_Eud.append(e)
#    S_ITML.append(i)
#    S_LSML.append(l)
#    
#printChart(S_Mah,S_Eud,S_ITML,S_LSML)
#
##    

"""
Learning with the network
"""


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


X,Y,T = data_elite()

margin=1

n = len(X)
d = len(X[0])
nr = len(T)
nb = 20
lamd = 0.5
maxNumIts = 100 

X = np.matrix(X)

M = np.ones([d,d])
M = sp.csr_matrix(M)
for t in range(maxNumIts):
    eta = 1/(lamd * (t+1))
    C = sp.csr_matrix((n,n))
    for r in range(nr):
        Cr = sp.csr_matrix((n,n))
        ns = 0        
        for b in range(nb) :
            [row,col] = T[r].nonzero()
            idi = np.random.randint(len(row))
            i = row[idi]
            j = col[idi]
            k = np.random.choice(np.where(T[r].getrow(i).toarray()!=1)[1])
            XiM = M * X[i,:].T
            XjM = M * X[j,:].T
            XkM = M * X[k,:].T
            Kii = X[i,:] * XiM
            Kjj = X[j,:] * XjM
            Kkk = X[k,:] * XkM
            Kij = X[j,:] * XiM
            Kji = Kij
            Kik = X[k,:]* XiM
            Kki = Kik
            dis_ij = Kii + Kjj - Kij - Kji
            dis_ik = Kii + Kkk - Kik - Kki
            if (dis_ij - dis_ik + margin>=0):
                Cr = Cr+sp.csr_matrix(([1,-1,-1,1,1,-1],([j,i,j,i,k,k],[j,j,i,k,i,k])),shape =(n,n))
                ns = ns+1
        Cr = (1/ns)*Cr
        C = C +Cr
    C = (1/nr)*C
    grad = X.T*C*X+ lamd * M
    print(sum(grad))
    M = M - eta * grad
    M = nearPSD(M)
    
K = 5
dim = len(X[0])
Score = ComputeKNNScoreLF(X,Y,K,M,dim = dim)
"Accuracy of myKNN: 0.3955 (+/- 0.0232)"
"elite"
"Network based Accuracy : 0.8034 (+/- 0.1869)"

    
            
 
 
