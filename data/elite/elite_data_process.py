# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 15:50:17 2017

@author: victor
"""

import numpy as np
#localrep ="./data/elite"
    
fp = open("DutchElite.txt",encoding='cp1252')
topname = []
topfg = 0
xfg = 0
bfg = 0
rfg = 0
t = []
X = []
R = []
for line in fp:
    if line[0]=='%' or line[0]=='\n' :
        continue
    if line.split()[0].isdigit()==0:  
        topfg = 0
        if xfg ==1 and len(t)!= 0:
            X.append(t)
        if bfg ==1 and len(t)!= 0:
            X.append(t)
        if rfg ==1 and len(t)!= 0:
            R.append(t)
        xfg = 0
        bfg = 0
        rfg = 0
        t = []
        if line=='*Top200\n':
            topfg = 1
            continue
        if line[0:2]=='*x':
            xfg = 1
            continue
        if line[0:2]=='*V':
            bfg = 1
            continue
        if line[0:2]=='*E':
            rfg = 1
            continue        
    if topfg==1:        
        topname.append(line.split(sep = '"')[1])
        continue    
    if xfg ==1:
        t.append(float(topname.count(line.split(sep = '"')[1])))
        continue
    if bfg ==1 :
        t.append(float(line.split()[0]))
        continue
    if rfg ==1 :
        t.append((int(line.split()[0])-1,int(line.split()[1])-1))
        continue

for i in range(len(X)):
    X[i] = [-1 if x == 9999998.000 else x for x in X[i]]

X = np.array(X).T        
r = []
for i in range(len(R)):
    r.extend([[i,x[0],x[1]] for x in R[i]])

np.savetxt("elite_data.txt",X,delimiter=",",fmt = '%s')
np.savetxt("elite_relation.txt",r,delimiter=",",fmt = '%d')
    

    