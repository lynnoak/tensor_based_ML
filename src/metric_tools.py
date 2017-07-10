# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 11:45:27 2017

metric computation

"""
import numpy as np
from sklearn import *
from sklearn.neighbors import *
import timeit
import cvxopt as cvx
from itertools import *

def ComputeKNNScore(X,Y,K,pnorm,title = ""):
    """
    compute the cross validation score of the KNN as Control
    Input: "X,Y" the dataset
        "K" K of KNN alg
         "pnorm" the power of the norm
    Output: the mean and std of KNNscore 
        
    """

    KNN = KNeighborsClassifier(n_neighbors=K,p=pnorm)
    KNN.fit(X,Y)
    score_KNN = cross_validation.cross_val_score(KNN,X,Y,cv=5)
    print(score_KNN)
    print(title+" Accuracy : %0.4f (+/- %0.4f)" % (score_KNN.mean(), score_KNN.std()))  
    return score_KNN.mean(),score_KNN.std()

def metricLF(X1,X2,**kwargs):
    if (len(X1)!=kwargs["dim"]):
        return sum(abs(X1-X2))
    M = kwargs["M"]
    XiM = M * X1.T
    XjM = M * X2.T
    Kii = X1 * XiM
    Kjj = X2 * XjM
    Kij = X2 * XiM
    Kji = Kij
    return Kii + Kjj - Kij - Kji

    
def ComputeKNNScoreLF(X,Y,K,M,dim,metric=metricLF): 
    """
    compute the cross validation score of the metric
    Input: "X,Y" the dataset
        "K" K of KNN alg
         "metric" metric computation function
    Output: the mean and std of score 
        
    """
       
    #KNN with the metric 
    myKNN = KNeighborsClassifier(n_neighbors=K, 
                                 metric=metric,metric_params={"M":M,"dim":dim})
    myKNN.fit(X,Y)
    #cross validation
    score_my = cross_validation.cross_val_score(myKNN,X,Y,cv=5)
    print(score_my)
    print("Relational Tensor based Accuracy : %0.4f (+/- %0.4f)" % (score_my.mean(), score_my.std()))  
    return score_my.mean(),score_my.std()