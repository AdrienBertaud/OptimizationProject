# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 10:19:31 2020

@author: yanis
"""

import numpy as np
import numpy.linalg as linalg

def BFGS(f, df, x0, N):
    d = x0.shape[0]
    
    sigma = []
    y = []
    x = []
    f_ = []
    
    H = np.eye(d)
    df0 = df(x0)
    
    for i in range(N):
        x.append(x0)
        f_.append(f(x0))

        x1 = x0 - np.dot(H, df0)
        df1 = df(x1)
        sigma.append(x1 - x0)
        y.append(df1 - df0)

        st = sigma[-1]
        yt = y[-1]

        Et = 1/(yt.T @ st) * (- H @ yt @ st.T - st @ yt.T @ H + (1+(yt.T @ H @ yt)/(yt.T @ st)) * st @ st.T)
        H = H + Et

        x0 = x1
        df0 = df1

    x.append(x0)
    f_.append(f(x0))

    return x, f_
        