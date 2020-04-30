# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 10:19:31 2020

@author: yanis
"""

import numpy as np
import numpy.linalg as linalg

def BFGS(f, df, x0, x1, N):
    d = x0.shape[0]
    
    s = []
    y = []
    
    s0 = x1 - x0
    y0 = df(x1) - df(x0)
    
    s.append(s0)
    y.append(y0)
    
    gamma = np.dot(y0, y0)/(np.dot(y0, s0))
    B = gamma * np.eye(d)
    
    #Pas fini, à continuer selon les notes de cours ou les articles choisis
        