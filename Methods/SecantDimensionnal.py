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

    if debug:
        print("x0 = ", x0)
        print("x1 = ", x1)

    x2 = x1

    d = grad(x0).shape[0]
    if debug:
        print("d = ", d)

    for i in range(N):

        # gradient of x0
        grad0 = np.zeros([d,1])
        grad0[:,0] = grad(x0)
        if debug:
            print("grad0 = ", grad0)

        # gradient of x1
        grad1 = np.zeros([d,1])
        grad1[:,0] = grad(x1)
        if debug:
            print("grad1 = ", grad1)

        # approximation of hessian
        H = np.zeros([d,d])
        H[0:] = grad1 - grad(x0)
        if debug:
            print("H = ", H)
        Denom = np.empty([d,d])
        if debug:
            print("Denom = ", Denom)
        Denom[:0] = x1 - x0
        if debug:
            print("Denom = ", Denom)
        np.divide(H,Denom.T)
        if debug:
            print("H = ", H)

        # check that H can be inverted
        if linalg.matrix_rank(H) < d:
            print("Error: hessian is not invertible. Stopping at the actual value.")
            break;

        # invert H
        HInv = linalg.inv(H)
        if debug:
            print("HInv = ", HInv)

        # compute next iteration of x
        x2 = x1 - HInv.T.dot(grad1)#TODO: is transpose right?
        if debug:
            print("x2 = ", x2)

    if verbose:
        print("Root = ", x2)

    return x0

