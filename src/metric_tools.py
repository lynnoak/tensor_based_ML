# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 11:45:27 2017

metric computation

"""
import numpy as np
from sklearn import *
from sklearn.neighbors import *


def ComputeKNNScore(X,Y,K,pnorm,scoring = ['accuracy'],title = ""):
    """
    compute the cross validation score of the KNN as Control
    Input: "X,Y" the dataset
        "K" K of KNN alg
         "pnorm" the power of the norm
    Output: the mean and std of KNNscore 
        
    """

    S = {}

    ditscoring = {'precision':metrics.make_scorer(metrics.precision_score,average = 'weighted'),
                  'recall': metrics.make_scorer(metrics.recall_score, average='weighted'),
                  'f1': metrics.make_scorer(metrics.f1_score, average='weighted')}


    if (scoring == 'test'):
        scoring = ['accuracy','f1','precision','recall']

    if not isinstance(scoring,list):
        scoring = [scoring]

    for s in scoring:
        S_mea = []
        for i in range(5):
            KNN = KNeighborsClassifier(n_neighbors=K, p=pnorm)
            KNN.fit(X, Y)
            kf = model_selection.StratifiedKFold(n_splits=3, shuffle=True)
            score_KNN = model_selection.cross_val_score(KNN, X, Y, cv=kf,
                                                        scoring=ditscoring.get(s, 'accuracy'))
            S_mea.append(score_KNN.mean())
        S_mea = np.mean(S_mea)
        print(title + " " + s + " : %0.4f " % (S_mea))
        S[s] =  S_mea
    return S

def metricLF(X1,X2,**kwargs):
    if (len(X1)!=kwargs["dim"]):
        return sum(abs(X1-X2))
    M = kwargs["M"]
    X1 = np.matrix(X1)
    X2 = np.matrix(X2)
    XiM = M * X1.T
    XjM = M * X2.T
    Kii = X1 * XiM
    Kjj = X2 * XjM
    Kij = X2 * XiM
    Kji = Kij
    r = Kii + Kjj - Kij - Kji
    return r[0,0]

    
def ComputeKNNScoreLF(X,Y,K,M,metric=metricLF,scoring = 'accuracy', title=""):
    """
    compute the cross validation score of the metric
    Input: "X,Y" the dataset
        "K" K of KNN alg
         "metric" metric computation function
    Output: the mean and std of score 
        
    """
    S = {}
    S_mea = []
    dim = len(X[0])

    ditscoring = {'precision':metrics.make_scorer(metrics.precision_score,average = 'weighted'),
                  'recall': metrics.make_scorer(metrics.recall_score, average='weighted'),
                  'f1': metrics.make_scorer(metrics.f1_score, average='weighted')}

    if (scoring == 'test'):
        scoring = ['accuracy','f1','precision','recall']

    if not isinstance(scoring,list):
         scoring = [scoring]

    for s in scoring:
         S_mea = []
         for i in range(5):
             myKNN = KNeighborsClassifier(n_neighbors=K,
                                          metric=metric, metric_params={"M": M, "dim": dim})
             myKNN.fit(X, Y)
             kf = model_selection.StratifiedKFold(n_splits=3, shuffle=True)
             score_KNN = model_selection.cross_val_score(myKNN, X, Y, cv=kf,
                                                         scoring=ditscoring.get(s, 'accuracy'))
             S_mea.append(score_KNN.mean())
         S_mea = np.mean(S_mea)
         print(title + " " + s + " : %0.4f " % (S_mea))
         S[s] =  S_mea

    return S
