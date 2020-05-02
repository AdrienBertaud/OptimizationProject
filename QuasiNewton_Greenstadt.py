# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 10:41:58 2020

@author: yanis
"""

import numpy as np

def Greenstadt(f, df, x0, N, M):
    d = x0.shape[0]

    sigma = []
    y = []
    x = []
    f = []
    H = np.eye(d)
    df0 = df(x0)

    for i in range(N):
        x.append(x0)
        f.append(f(x0))

        x1 = x0 - np.dot(H, df0)
        df1 = df(x1)
        sigma.append(x1 - x0)
        y.append(df1 - df0)

        st = sigma[-1]
        yt = y[-1]

        Et = 1/(np.dot(np.dot(yt.T, M), yt))*(np.dot(np.dot(st, yt.T), M) + np.dot(np.dot(M, yt), st.T) - np.dot(np.dot(H, yt), np.dot(yt.T, M)) - np.dot(np.dot(M, yt), np.dot(yt.T, H)) - 1/(np.dot(np.dot(yt.T, M), yt)) * (np.dot(yt.T, st) - np.dot(np.dot(yt.T, H), yt)) * np.dot(np.dot(M, yt), np.dot(yt.T, M)))
        H = H + Et

        x0 = x1
        df0 = df1

    return x, f