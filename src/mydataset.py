# -*- coding: utf-8 -*-
"""
Dataset loading

@author: victor
"""
from sklearn import preprocessing
from scipy.sparse import csr_matrix
import numpy as np
localrep ="./data/"



def triples_to_tensor(nx,t):
    nr = set(t[:,0])
    T = []
    for i in nr:
        r = []
        c = []
        for j in range(len(t)):
            if t[j,0] == i:
                r.append(int(t[j,1]))
                c.append(int(t[j,2]))
        d = np.ones(len(r))
        r = np.array(r)
        c = np.array(c)
        T.append(csr_matrix((d, (r, c)), shape=(nx, nx)))    
    return T



"""
Return :
the vector of label of nation Y
the relation tensor T

"""

def data_nation():
    file1=localrep+"nations/category.txt"
    Y = np.genfromtxt(file1,delimiter=",")
    file2=localrep+"nations/triples.txt"
    t = np.genfromtxt(file2,delimiter=",")
    
    nx = len(Y)
    T = triples_to_tensor(nx,t)
    return Y,T
    

"""
Return :
the vector of feature of data X
the vector of label of data Y
the relation tensor T
"""
def data_movie():
    file1=localrep+"movie/movie_data.txt"
    XY = np.genfromtxt(file1,delimiter=",")
    X=XY[:,1:6]
    X = preprocessing.scale(X)
    m = preprocessing.MinMaxScaler()
    X = m.fit_transform(X)  		 
    Y=XY[:,0]
    file2=localrep+"movie/movie_relation.txt"
    t = np.genfromtxt(file2,delimiter=",",dtype = str)
    
    nx = len(Y)
    T = triples_to_tensor(nx,t)
    return X,Y,T
    
def data_sub_movie():
    file1=localrep+"movie/sub_movie_data.txt"
    XY = np.genfromtxt(file1,delimiter=",")
    X=XY[:,1:6]
    X = preprocessing.scale(X)
    m = preprocessing.MinMaxScaler()
    X = m.fit_transform(X)  		 
    Y=XY[:,0]
    file2=localrep+"movie/sub_movie_relation.txt"
    t = np.genfromtxt(file2,delimiter=",",dtype = str)
    
    nx = len(Y)
    T = triples_to_tensor(nx,t)
    return X,Y,T

"""
Return :
the vector of feature of data X
the vector of label of data Y
the relation tensor T
"""
def data_elite():
    file1=localrep+"elite/elite_data.txt"
    XY = np.genfromtxt(file1,delimiter=",")
    X=XY[:,1:8]
    X = preprocessing.scale(X)
    m = preprocessing.MinMaxScaler()
    X = m.fit_transform(X)  		 
    Y=XY[:,0]
    file2=localrep+"elite/elite_relation.txt"
    t = np.genfromtxt(file2,delimiter=",",dtype = str)
    
    nx = len(Y)
    T = triples_to_tensor(nx,t)
    return X,Y,T
    
"""
Return :
the vector of feature of data X
the vector of label of data Y
the relation tensor T
"""
def data_UW_std():
    file1=localrep+"UW_std_data.txt"
    XY = np.genfromtxt(file1,delimiter=",")
    X=XY[:,1:4]
    X = preprocessing.scale(X)
    m = preprocessing.MinMaxScaler()
    X = m.fit_transform(X)  		 
    Y=XY[:,0]
    file2=localrep+"UW_std_relation.txt"
    t = np.genfromtxt(file2,delimiter=",",dtype = str)
    
    nx = len(Y)
    T = triples_to_tensor(nx,t)
    return X,Y,T
    
"""
Return :
the vector of feature of data X
the vector of label of data Y
the relation tensor T
"""
def data_Mutagenesis_std():
    file1=localrep+"Mutagenesis_std_data.txt"
    XY = np.genfromtxt(file1,delimiter=",")
    X=XY[:,1:3]
    X = preprocessing.scale(X)
    m = preprocessing.MinMaxScaler()
    X = m.fit_transform(X)  		 
    Y=XY[:,0]
    file2=localrep+"Mutagenesis_std_relation.txt"
    t = np.genfromtxt(file2,delimiter=",",dtype = str)
    
    nx = len(Y)
    T = triples_to_tensor(nx,t)
    return X,Y,T
    
"""
Return :
the vector of feature of data X
the vector of label of data Y
the relation tensor T
"""
def data_Mondial_std():
    file1=localrep+"Mondial_std_data.txt"
    XY = np.genfromtxt(file1,delimiter=",")
    X=XY[:,1:5]
    X = preprocessing.scale(X)
    m = preprocessing.MinMaxScaler()
    X = m.fit_transform(X)  		 
    Y=XY[:,0]
    file2=localrep+"Mondial_std_relation.txt"
    t = np.genfromtxt(file2,delimiter=",",dtype = str)
    
    nx = len(Y)
    T = triples_to_tensor(nx,t)
    return X,Y,T


"""
Return :
the vector of feature of data X
the vector of label of data Y
the relation tensor T
"""


def data_attack_std():
    file1 = localrep + "attack/attack_std_data.txt"
    XY = np.genfromtxt(file1, delimiter=",")
    X = XY[:, :-1]
    X = preprocessing.scale(X)
    m = preprocessing.MinMaxScaler()
    X = m.fit_transform(X)
    Y = XY[:, -1]
    file2 = localrep + "attack/attack_std_relation.txt"
    t = np.genfromtxt(file2, delimiter=",", dtype=str)

    nx = len(Y)
    T = triples_to_tensor(nx, t)
    return X, Y, T


    