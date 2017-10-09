# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 15:50:17 2017

@author: victor
"""

import numpy as np
localrep ="./data/movie"

t = np.loadtxt(localrep+"movie_relation.csv",delimiter=",",dtype = str)
tx = list(set(np.concatenate((t[:,1],t[:,2]))))
c = 0
X = {}
for i in tx:
    X[i]=c
    c=c+1
for i in range(len(t)):
    t[i,1]=X[t[i,1]]
    t[i,2]=X[t[i,2]]
    
np.savetxt("movie_relation.txt",t,delimiter=",",fmt = '%s')

ll = {'Ct':1, 'Di':2, 'Dr':3, 'Cn':4, 'Co':5, 'Fa':6, 'Hi':7, 'Ho':8, 
'Mu':9,'No':10, 'Ro':11, 'Sc':12, 'Su':13,'Bi':14,'We':15,'Ad':16,'Do':17,
'Ep':18,'My':19,'Po':20}
s19 = '19'
d0 = 'D:'
pl = {'pr':1,'PN':2,'PU':3,'PZ':4}
sl = {'S:':1,'SD':2,'st':3,'SL':4,'SU':5}
cl = {'p':1,'pr':2,'bw':3,'bn':4,'co':5,'cl':6,'\T':7}


fp = open(localrep+"movie_data.csv")    
data = []
for line in fp:
    lp = np.array(line.split(sep = ','))
    if(X.get("b'"+lp[0]+"'",None)!=None ):
        lpt=[X["b'"+lp[0]+"'"]]
        
        lpt.append(ll.get(lp[6][0:2],0))        
        
        n = lp[1].find(s19)
        if n<0:
            lpt.append(0)
        else:
            lpt.append(int(lp[1][n:n+4]))
            
        n = lp[2].find(d0)
        if n<0:
            lpt.append(0)
        else:
            lpt.append(ord(lp[2][n+2]))
        
        lpt.append(pl.get(lp[3][0:2],0))
        
        lpt.append(sl.get(lp[4][0:2],0))
        
        lpt.append(cl.get(lp[5][0:2],0))
        
        if(lpt):
            data.append(lpt)

td = np.zeros([len(X),6],dtype = int)
for l in data:
    td[l[0],:] = l[1:7]

np.savetxt("movie_data.txt",td,delimiter=",",fmt = '%s')


        
        
    