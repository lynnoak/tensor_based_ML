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
from metric_learn import LMNN,ITML_Supervised,LSML_Supervised

import sktensor
from sktensor.rescal import als as rescal

from src.mydataset import *
from src.metric_tools import *
from src.print_tools import *




"""
Learning with the features
"""

def Fea_ML(X,Y):

    K = 5

    S_Mah = ComputeKNNScore(X,Y,K,1,title = "Mah")

    S_Eud = ComputeKNNScore(X,Y,K,2,title = "Eud")
        
    try:
        itml = ITML_Supervised(num_constraints=100)
        itml.fit(X,Y)
        XI = itml.transform(X)
        S_ITML = ComputeKNNScore(XI,Y,K,2,title = "ITML")
    except:
        S_ITML = [0,0]
    
    try:
        lsml = LSML_Supervised(num_constraints=100)
        lsml.fit(X,Y)
        XL = lsml.transform(X)
        S_LSML = ComputeKNNScore(XL,Y,K,2,title = "LSML")
    except:
        S_LSML = [0,0]

    return S_Mah[0],S_Eud[0],S_ITML[0],S_LSML[0]


"""
Learning with the rescal
"""

def Res_ML(X,Y,T):

    rank  = max(int(len(X[0])*2/3),5)

    A, R, fval, iter, exectimes = rescal(T, rank)

    X = A
    Y = Y
    K = 5
    

    S_Mah = ComputeKNNScore(X,Y,K,1,title = "Mah")

    S_Eud = ComputeKNNScore(X,Y,K,2,title = "Eud")
        
    try:
        itml = ITML_Supervised(num_constraints=100)
        itml.fit(X,Y)
        XI = itml.transform(X)
        S_ITML = ComputeKNNScore(XI,Y,K,2,title = "ITML")
    except:
        S_ITML = [0,0]
    
    try:
        lsml = LSML_Supervised(num_constraints=100)
        lsml.fit(X,Y)
        XL = lsml.transform(X)
        S_LSML = ComputeKNNScore(XL,Y,K,2,title = "LSML")
    except:
        S_LSML = [0,0]

    return S_Mah[0],S_Eud[0],S_ITML[0],S_LSML[0]

"""
Learning with the rescal and features
"""

def RF_ML(X,Y,T):

    rank  = max(int(len(X[0])*2/3),5)

    A, R, fval, iter, exectimes = rescal(T, rank)

    X = np.column_stack((X,A))
    Y = Y
    K = 5

    S_Mah = ComputeKNNScore(X,Y,K,1,title = "Mah")

    S_Eud = ComputeKNNScore(X,Y,K,2,title = "Eud")
        
    try:
        itml = ITML_Supervised(num_constraints=100)
        itml.fit(X,Y)
        XI = itml.transform(X)
        S_ITML = ComputeKNNScore(XI,Y,K,2,title = "ITML")
    except:
        S_ITML = [0,0]
    
    try:
        lsml = LSML_Supervised(num_constraints=100)
        lsml.fit(X,Y)
        XL = lsml.transform(X)
        S_LSML = ComputeKNNScore(XL,Y,K,2,title = "LSML")
    except:
        S_LSML = [0,0]

    return S_Mah[0],S_Eud[0],S_ITML[0],S_LSML[0]
  

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

def TB_ML(X,Y,T):

    
    margin=1
    n = len(X)
    d = len(X[0])
    nr = len(T)
    lamd = 0.5
    maxNumIts = 100 
    consNum = 200
    relconsNum = int(consNum)+1

    X = np.matrix(X)

    M = np.ones([d,d])
    for t in range(maxNumIts):
        eta = 1/(lamd * (t+1))
        C = sp.csr_matrix((n,n))
        for r in range(nr):
            Cr = sp.csr_matrix((n,n))
            ns = 0
            nb = 0
            for b in range(consNum):
                
                if nb <= relconsNum:
                    [row,col] = T[r].nonzero()
                    idi = np.random.randint(len(row))
                    i = row[idi]
                    j = col[idi]
                    k = np.random.choice(np.where(T[r].getrow(i).toarray()!=1)[1])
                else:
                    break
                nb = nb+1
                
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
                    
            if ns!=0:
                Cr = (1/ns)*Cr
            C = C +Cr
        C = (1/nr)*C
        grad = X.T*C*X+ lamd * M
        M = M - eta * grad
        M = nearPSD(M)
        
    K = 5
    dim = len(X[0])
    S_TB = ComputeKNNScoreLF(X,Y,K,M,dim = dim)    
    return S_TB[0]
    
    
def TB_S_ML(X,Y,T,p=0.6):

    
    margin=1
    n = len(X)
    d = len(X[0])
    nr = len(T)
    lamd = 0.5
    maxNumIts = 100 
    consNum = 200
    relconsNum = int(consNum*p)+1
    labconsNum = int(consNum*(1-p))+1
    print(relconsNum,labconsNum)
    
    X = np.matrix(X)

    M = np.ones([d,d])
    M = sp.csr_matrix(M)
    for t in range(maxNumIts):
        eta = 1/(lamd * (t+1))
        C = sp.csr_matrix((n,n))
        for r in range(nr):
            Cr = sp.csr_matrix((n,n))
            ns = 0
            nb = 0
            for b in range(consNum):
                
                if nb<= relconsNum:
                    [row,col] = T[r].nonzero()
                    idi = np.random.randint(len(row))
                    i = row[idi]
                    j = col[idi]
                    k = np.random.choice(np.where(T[r].getrow(i).toarray()!=1)[1])
                else:
                    if nb<= relconsNum+labconsNum:
                        i = np.random.randint(len(Y))
                        j = np.random.choice(np.where(Y==Y[i])[0])
                        k = np.random.choice(np.where(Y!=Y[i])[0])
                    else:
                        break
                nb = nb+1
                    
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
                
            if ns!=0:
                Cr = (1/ns)*Cr
            C = C +Cr
        C = (1/nr)*C
        grad = X.T*C*X+ lamd * M
        M = M - eta * grad
        M = nearPSD(M)
        
    
    
    print(M)
    K = 5
    dim = len(X[0])
    S_TB_S = ComputeKNNScoreLF(X,Y,K,M,dim = dim)    
    return S_TB_S[0]

    

X,Y,T = data_elite()
#[F_Mah,F_Eud,F_ITML,F_LSML] = Fea_ML(X,Y)
#[R_Mah,R_Eud,R_ITML,R_LSML] = Res_ML(X,Y,T)
#[S_Mah,S_Eud,S_ITML,S_LSML] = RF_ML(X,Y,T)
S_TB = TB_ML(X,Y,T)
S_TB_S = []
for p in [0,0.2,0.4,0.6,0.8,1]:
    S_TB_S.append(TB_S_ML(X,Y,T,p))

                                        

#print("With Features: %0.4f & %0.4f & %0.4f & %0.4f "%(F_Mah,F_Eud,F_ITML,F_LSML))
#print("Reasal latent space : %0.4f & %0.4f & %0.4f & %0.4f "%(R_Mah,R_Eud,R_ITML,R_LSML))
#print("Reasal latent space with Features: %0.4f & %0.4f & %0.4f & %0.4f "%(S_Mah,S_Eud,S_ITML,S_LSML))
print("Tensor base: %0.4f  "%(S_TB))
for i in S_TB_S:
    print("Tensor base with labels: %0.4f  "%(i))

         
 
 
