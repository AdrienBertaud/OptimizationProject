# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 12:02:22 2020

@author: yanis
"""

import numpy as np
import numpy.linalg as linalg

def secantD(f, grad, x0, x1, N, verbose=True, debug=False):

    if verbose:
        print("*** Secant method in multiple dimensions***")

    x2 = x0

    for i in range(N):

        grad1 = grad(x1)

        d = grad1.shape[0]




        H = np.empty([d,d])

        H[0:] = grad1 - grad(x0)

        Denom = np.empty([d,d])

        Denom[:0] = x1 - x0

        np.divide(H,Denom.T)

        if linalg.matrix_rank(H) < d:
            print("Error: hessian is not invertible. Stopping at the actual value.")
            break;

        HInv = linalg.inv(H)

        x2 = x1 - HInv.T.dot(grad1)#TODO: is transpose right?

    if verbose:
        print("Root = ", x2)

    return x0

