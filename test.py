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




"""
Learning with the features
"""

def Fea_ML(X,Y):

    K = 5

    S_Mah = ComputeKNNScore(X,Y,K,1,title = "Mah")

    S_Eud = ComputeKNNScore(X,Y,K,2,title = "Eud")
        
    try:
        itml = ITML_Supervised(num_constraints=200)
        itml.fit(X,Y)
        XI = itml.transform(X)
        S_ITML = ComputeKNNScore(XI,Y,K,2,title = "ITML")
    except:
        S_ITML = [0,0]
    
    try:
        lsml = LSML_Supervised(num_constraints=200)
        lsml.fit(X,Y)
        XL = lsml.transform(X)
        S_LSML = ComputeKNNScore(XL,Y,K,2,title = "LSML")
    except:
        S_LSML = [0,0]

    try:
        nca = NCA(max_iter=1000, learning_rate=0.01)
        nca.fit(X,Y)
        XN = nca.transform(X)
        S_NCA = ComputeKNNScore(XN,Y,K,2,title = "NCA")
    except:
        S_NCA = [0,0]

    try:
        rca = RCA_Supervised(num_chunks=30, chunk_size=2)
        rca.fit(X,Y)
        XR = rca.transform(X)
        S_RCA = ComputeKNNScore(XR,Y,K,2,title = "RCA")
    except:
        S_RCA = [0,0]


    return S_Mah[0],S_Eud[0],S_ITML[0],S_LSML[0],S_NCA[0],S_RCA[0]


"""       
        Learning with the rescal
"""

def Res_ML(X,Y,T):

    rank  = 10

    A, R, fval, iter, exectimes = rescal(T, rank)

    X = A
    Y = Y
    K = 5
    

    S_Mah = ComputeKNNScore(X,Y,K,1,title = "Mah")

    S_Eud = ComputeKNNScore(X,Y,K,2,title = "Eud")
        
    try:
        itml = ITML_Supervised(num_constraints=200)
        itml.fit(X,Y)
        XI = itml.transform(X)
        S_ITML = ComputeKNNScore(XI,Y,K,2,title = "ITML")
    except:
        S_ITML = [0,0]
    
    try:
        lsml = LSML_Supervised(num_constraints=200)
        lsml.fit(X,Y)
        XL = lsml.transform(X)
        S_LSML = ComputeKNNScore(XL,Y,K,2,title = "LSML")
    except:
        S_LSML = [0,0]

    try:
        nca = NCA(max_iter=1000, learning_rate=0.01)
        nca.fit(X,Y)
        XN = nca.transform(X)
        S_NCA = ComputeKNNScore(XN,Y,K,2,title = "NCA")
    except:
        S_NCA = [0,0]

    try:
        rca = RCA_Supervised(num_chunks=30, chunk_size=2)
        rca.fit(X,Y)
        XR = rca.transform(X)
        S_RCA = ComputeKNNScore(XR,Y,K,2,title = "RCA")
    except:
        S_RCA = [0,0]


    return S_Mah[0],S_Eud[0],S_ITML[0],S_LSML[0],S_NCA[0],S_RCA[0]

"""
Learning with the rescal and features
"""

def RF_ML(X,Y,T):

    rank  = 10

    A, R, fval, iter, exectimes = rescal(T, rank)

    X = np.column_stack((X,A))
    Y = Y
    K = 5

    S_Mah = ComputeKNNScore(X,Y,K,1,title = "Mah")

    S_Eud = ComputeKNNScore(X,Y,K,2,title = "Eud")
        
    try:
        itml = ITML_Supervised(num_constraints=200)
        itml.fit(X,Y)
        XI = itml.transform(X)
        S_ITML = ComputeKNNScore(XI,Y,K,2,title = "ITML")
    except:
        S_ITML = [0,0]
    
    try:
        lsml = LSML_Supervised(num_constraints=200)
        lsml.fit(X,Y)
        XL = lsml.transform(X)
        S_LSML = ComputeKNNScore(XL,Y,K,2,title = "LSML")
    except:
        S_LSML = [0,0]

    try:
        nca = NCA(max_iter=1000, learning_rate=0.01)
        nca.fit(X,Y)
        XN = nca.transform(X)
        S_NCA = ComputeKNNScore(XN,Y,K,2,title = "NCA")
    except:
        S_NCA = [0,0]

    try:
        rca = RCA_Supervised(num_chunks=30, chunk_size=2)
        rca.fit(X,Y)
        XR = rca.transform(X)
        S_RCA = ComputeKNNScore(XR,Y,K,2,title = "RCA")
    except:
        S_RCA = [0,0]


    return S_Mah[0],S_Eud[0],S_ITML[0],S_LSML[0],S_NCA[0],S_RCA[0]

  

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
    relconsNum = int(consNum/nr)+1

    Xm = np.matrix(X)

    M = np.matrix(np.ones([d,d]))
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
                
                XiM = M * Xm[i,:].T
                XjM = M * Xm[j,:].T
                XkM = M * Xm[k,:].T
                Kii = Xm[i,:] * XiM
                Kjj = Xm[j,:] * XjM
                Kkk = Xm[k,:] * XkM
                Kij = Xm[j,:] * XiM
                Kji = Kij
                Kik = Xm[k,:]* XiM
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
        grad = Xm.T*C*Xm+ lamd * M
        M = M - eta * grad
        M = nearPSD(M)
        
    K = 5
    S_TB = ComputeKNNScoreLF(X,Y,K,M,dim = d)    
    return S_TB[0]
    
    
def TB_S_ML(X,Y,T,p=0.6):

    
    margin=1
    n = len(X)
    d = len(X[0])
    nr = len(T)
    lamd = 0.5
    maxNumIts = 100 
    consNum = 200
    relconsNum = int(consNum*p/nr)+1
    labconsNum = int(consNum*(1-p)/nr)+1
    print(relconsNum,labconsNum)
    
    Xm = np.matrix(X)

    M = np.matrix(np.ones([d,d]))
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
                    
                XiM = M * Xm[i,:].T
                XjM = M * Xm[j,:].T
                XkM = M * Xm[k,:].T
                Kii = Xm[i,:] * XiM
                Kjj = Xm[j,:] * XjM
                Kkk = Xm[k,:] * XkM
                Kij = Xm[j,:] * XiM
                Kji = Kij
                Kik = Xm[k,:]* XiM
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
        grad = Xm.T*C*Xm+ lamd * M
        M = M - eta * grad
        M = nearPSD(M)
        
    
    
    print(M)
    K = 5
    S_TB_S = ComputeKNNScoreLF(X,Y,K,M,dim = d)    
    return S_TB_S[0]

"""
test

"""    

#X,Y,T = data_movie()
#S_Fea_ML = Fea_ML(X,Y)
#S_Res_ML = Res_ML(X,Y,T)
#S_RF_ML = RF_ML(X,Y,T)
#S_TB = TB_ML(X,Y,T)
#S_TB_S = []
#for p in [0,0.2,0.4,0.6,0.8]:
#    S_TB_S.append(TB_S_ML(X,Y,T,p))
#print("\nWith Features: "+" %0.4f &"*len(S_Fea_ML) % tuple(S_Fea_ML))
#print("\nReasal latent space : "+" %0.4f &"*len(S_Res_ML) % tuple(S_Res_ML))
#print("\nReasal latent space with Features: "+" %0.4f &"*len(S_RF_ML) % tuple(S_RF_ML))
#print("\nTensor base: %0.4f  "%(S_TB))
#for i in S_TB_S:
#    print("\nTensor base with labels: %0.4f  "%(i))

def My_output(filename = 'output.txt'):
    
    datalist = [data_elite(),data_UW_std(),data_Mutagenesis_std(),data_Mondial_std()]
    datanames = ["elite","UW","Mutagenesis","Mondial"]


    for i in range(len(datalist)):
        X,Y,T = datalist[i]
        S_Fea_ML = Fea_ML(X,Y)
        S_Res_ML = Res_ML(X,Y,T)
        S_RF_ML = RF_ML(X,Y,T)
        #S_TB = TB_ML(X,Y,T)
        #S_TB_S = []
        #for p in [0,0.2,0.4,0.6,0.8]:
        #    S_TB_S.append(TB_S_ML(X,Y,T,p))
        f = open(filename, 'a')
        f.write("\n"+datanames[i])
        f.write("\nWith Features: "+" %0.4f &"*len(S_Fea_ML) % tuple(S_Fea_ML))
        f.write("\nReasal latent space : "+" %0.4f &"*len(S_Res_ML) % tuple(S_Res_ML))
        f.write("\nReasal latent space with Features: "+" %0.4f &"*len(S_RF_ML) % tuple(S_RF_ML))
        #f.write("\nTensor base: %0.4f  "%(S_TB))
        #f.write("\n Tensor base with labels: "+" %0.4f &"*len(S_TB_S) % tuple(S_TB_S))
        f.close()


def mytest(X,Y,T):

    tx = X
    K = 5

    try:
        nca = NCA(max_iter=1000, learning_rate=0.01)
        nca.fit(X, Y)
        XN = nca.transform(X)
        Fea_NCA = ComputeKNNScore(XN, Y, K, 2, title="NCA")
    except:
        Fea_NCA = [0, 0]    

    try:
        rca = RCA_Supervised(num_chunks=30, chunk_size=2)
        rca.fit(X, Y)
        XR = rca.transform(X)
        Fea_RCA = ComputeKNNScore(XR, Y, K, 2, title="RCA")
    except:
        Fea_RCA = [0, 0]

    rank  = 10
    A, R, fval, iter, exectimes = rescal(T, rank)
    X = A
    Y = Y
    K = 5

    try:
        nca = NCA(max_iter=1000, learning_rate=0.01)
        nca.fit(X, Y)
        XN = nca.transform(X)
        Res_NCA = ComputeKNNScore(XN, Y, K, 2, title="NCA")
    except:
        Res_NCA = [0, 0]    

    try:
        rca = RCA_Supervised(num_chunks=30, chunk_size=2)
        rca.fit(X, Y)
        XR = rca.transform(X)
        Res_RCA = ComputeKNNScore(XR, Y, K, 2, title="RCA")
    except:
        Res_RCA = [0, 0]

    X = np.column_stack((tx,A))
 

    try:
        nca = NCA(max_iter=1000, learning_rate=0.01)
        nca.fit(X, Y)
        XN = nca.transform(X)
        RF_NCA = ComputeKNNScore(XN, Y, K, 2, title="NCA")
    except:
        RF_NCA = [0, 0]    

    try:
        rca = RCA_Supervised(num_chunks=30, chunk_size=2)
        rca.fit(X, Y)
        XR = rca.transform(X)
        RF_RCA = ComputeKNNScore(XR, Y, K, 2, title="RCA")
    except:
        RF_RCA = [0, 0]

    print(Fea_NCA[0],Fea_RCA[0],Res_NCA[0],Res_RCA[0],RF_NCA[0],RF_RCA[0])

    return Fea_NCA,Fea_RCA,Res_NCA,Res_RCA,RF_NCA,RF_RCA





