# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 11:23:23 2020

@author: yanis
"""

import numpy.linalg as linalg

def newtond(f, df, d2f, x0, N, verbose=True, debug=False):

    if verbose:
        print("*** Newton-Raphson method for optimization***")

    for i in range(N):

        gradient = df(x0)
        hessian = d2f(x0)
        d = gradient.shape[0]

        if linalg.matrix_rank(hessian) < d:
            print("Error in newton : hessian is not invertible. Stopping at the actual value.")
            break;

        deltax = linalg.solve(hessian, gradient)

        x0 = x0 - deltax

        if verbose:
            print("Root = ", x0)

        if debug:
            print("Error = ", linalg.norm(gradient))

    return x0