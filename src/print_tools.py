# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 11:35:24 2017

@author: victor
"""

import numpy as np
import matplotlib.pyplot as plt

def printChart(S_Mah,S_Eud,S_ITML,S_LSML):
    
    plt.figure(1)
    plt.title("Compare")# give plot a title
    plt.xlabel("Rank")
    plt.ylabel("Score")
    plt.xlim(0.0, len(S_Mah)*1.3)
    y_min = np.floor(min(min(S_Mah,S_Eud,S_ITML,S_LSML))*10)/10
    y_max = (np.floor(max(max(S_Mah,S_Eud,S_ITML,S_LSML))*10)+1)/10
    plt.ylim(y_min,y_max)
    plt.plot(range(len(S_Mah)),S_Mah,label = "S_Mah")
    plt.plot(range(len(S_Eud)),S_Eud,label = "S_Eud")
    plt.plot(range(len(S_ITML)),S_ITML,label = "S_ITML")
    plt.plot(range(len(S_LSML)),S_LSML,label = "S_LSML")
    plt.legend()
        
    plt.savefig("Compare.png")
    plt.show()
    