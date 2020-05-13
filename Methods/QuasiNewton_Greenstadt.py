# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 10:41:58 2020

@author: yanis
"""

import numpy as np
import numpy.linalg as linalg

def Greenstadt(f, df, x0, H0, M, N, verbose=True, debug=False):

    """
    Greenstadt method with parameter M
    TODO: how to define parameter M? Is it different at any step?

    f: function from R^d to R
    df: gradient of f
    x0: starting point
    M: a symmetric and invertible matrix used as a parameter, that can be hessian at x0
    N: number of iteration
    """

    # intialize H
    H = H0 # TODO: how to initialize H?

    # compute x1
    x1 = x0 - np.dot(H, df(x0))

    if verbose:
        print("x0 = ", x0)
        print("x1 = ", x1)

    # create a list to store points
    xList = []
    xList.append(x0)
    xList.append(x1)

    # get the number of dimensions
    d = x0.shape[0]

    # ensure that M is symmetric and invertible
    if linalg.matrix_rank(M) < d:
        print("Error: M is not invertible. Stopping at the actual value.")
        return x1, xList

    tolerance = 1e-05
    if not np.all(np.abs(M-M.T) < tolerance):
        print("Error: M is not symmetric.")
        return x1, xList

    for i in range(N):

        # initialize values
        s = x1 - x0
        df0 = df(x0)
        df1 = df(x1)
        y = df1 - df0

        # E = yTHy
        E = np.dot(np.dot(y.T, H), y)

        # E = yTHy - yTs
        E -= (np.dot(y.T, s))

        # E = (yTHy - yTs) MyyTM
        E *= np.dot(np.dot(M, y), np.dot(y.T, M))

        # E = - 1/(yTMy) (yTs - yTHy) MyyTM
        E /= - np.dot(np.dot(y.T, M), y)

        # E = sytM - 1/(yTMy) (yTs - yTHy) MyyTM
        E += np.dot(np.dot(s, y.T), M)

        # E = sytM + MysT - 1/(yTMy) (yTs - yTHy) MyyTM
        E += np.dot(np.dot(M, y), s.T)

        # E = sytM + MysT - HyyTM - 1/(yTMy) (yTs - yTHy) MyyTM
        E -= np.dot(np.dot(H, y), np.dot(y.T, M))

        # E = sytM + MysT - HyyTM - MyyTH - 1/(yTMy) (yTs - yTHy) MyyTM
        E -= np.dot(np.dot(M, y), np.dot(y.T, H))

        # E = 1/ytMy [sytM + MysT - HyyTM - MyyTH - 1/(yTMy) (yTs - yTHy) MyyTM]
        E /= np.dot(np.dot(y.T, M), y)

        # update estimation of hessian
        H = H + E
        if debug:
            print("E = ", E)
            print("H = ", H)

        # compute next point
        x2 = x1 - np.dot(H, df1)

        # update values for next iteration
        x0 = x1
        x1 = x2

        if verbose:
            print("x = ", x2)

        xList.append(x1)

    return x2, xList