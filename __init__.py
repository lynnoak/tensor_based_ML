# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 11:09:57 2017

@author: victor
"""


from src.mydataset import *
from src.mytools import *

"""
test

"""    
#
#X,Y,T = data_attack_std()
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
#

"""
For result

"""

def MyRes(X,Y,T,scoring = 'accuracy'):

    S = {}

    K = 5

    S_Eud = ComputeKNNScore(X, Y, K, 2,scoring = scoring, title="Eud")
    S['Eud'] =  S_Eud

    X0 = X

    rank = min(300,int(0.25*len(X)))

    A, R, fval, iter, exectimes = rescal(T, rank)

    A = preprocessing.scale(A)
    m = preprocessing.MinMaxScaler()
    A = m.fit_transform(A)

    S_Res = ComputeKNNScore(A, Y, K, 2,scoring = scoring, title="Res")
    S['Res'] =  S_Res

    XF = np.column_stack((X,A))
    XF = preprocessing.scale(XF)
    m = preprocessing.MinMaxScaler()
    X = m.fit_transform(XF)

    S_Res_Fea = ComputeKNNScore(X, Y, K, 2, scoring = scoring,title="Res_Fea")
    S['Res_Fea'] =  S_Res_Fea

    try:
        itml = ITML_Supervised(num_constraints=200)
        itml.fit(X, Y)
        XI = itml.transform(X)
        S_ITML = ComputeKNNScore(XI, Y, K, 2, scoring = scoring,title="ITML")
    except:
        S_ITML = 0
    S['ITML'] =  S_ITML

    try:
        lsml = LSML_Supervised(num_constraints=200)
        try:
            lsml.fit(X, Y)
        except:
            tX = X + 10 ** -4
            lsml.fit(tX, Y)
        XL = lsml.transform(X)
        S_LSML = ComputeKNNScore(XL, Y, K, 2, scoring = scoring,title="LSML")
    except:
        S_LSML = 0

    S['LSML'] =  S_LSML

    try:
        try:
            lfda = LFDA(k=K)
            lfda.fit(X, Y)
        except:
            lfda = LFDA(dim = int(0.9*(len(X[0]))),k=K)
            tX = X + 10 ** -4
            lfda.fit(tX, Y)

        XL = lfda.transform(X)
        S_LFDA = ComputeKNNScore(XL, Y, K, 2, scoring = scoring,title="LFDA")
    except:
        S_LFDA = 0
    S['LFDA'] =  S_LFDA

#    try:
#        nca = NCA(max_iter=100, learning_rate=0.01)
#        nca.fit(X, Y)
#        XN = nca.transform(X)
#        S_NCA = ComputeKNNScore(XN, Y, K, 2, scoring = scoring,title="NCA")
#    except:
#        S_NCA = 0
#    S['NCA'] =  S_NCA
#
#    try:
#        rca = RCA_Supervised(num_chunks=30, chunk_size=2)
#        rca.fit(X, Y)
#        XR = rca.transform(X)
#        S_RCA = ComputeKNNScore(XR, Y, K, 2, scoring = scoring,title="RCA")
#    except:
#        S_RCA = 0
#    S['RCA'] =  S_RCA

    X = X0

    S_TB_nolabel = TB_ML(X,Y,T,scoring = scoring)
    S['TB_nolabel'] =  S_TB_nolabel

    S_TB = []


    titlep = ['all_label','p = 0.8','p = 0.6','p = 0.4','p = 0.2','no_label']
    p = [0,0.2,0.4,0.6,0.8,1]
    for i in range(6):
       S_TB.append({titlep[i]:TB_S_ML(X,Y,T,p[i],scoring = scoring)})

#    S['TB'] =  max(S_TB)

    S['TB_label'] = S_TB

    return S


X,Y,T = data_Mondial_std()
'''
scoring could be 'accuracy','f1','precision','recall'  or any combine as a list.
if scoring = 'test', scoring = ['accuracy','f1','precision','recall']
'''

scoring=['accuracy','f1']

myRes = MyRes(X,Y,T,scoring=scoring)

print(myRes)

