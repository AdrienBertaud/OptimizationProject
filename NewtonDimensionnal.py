# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 11:23:23 2020

@author: yanis
"""
import PlotIterations

from importlib import reload
reload(PlotIterations) # Relaod the module, in case it has changed

from PlotIterations import plotIterations


import numpy as np
import numpy.linalg as linalg

def newtond(f, df, d2f, x0, N, verbose=True, debug=False, plot=False):
    
    if verbose:
        print("*** Newton-Raphson method for optimization***")
    
    x = []
    y = []
    
    for i in range(N):
        
        gradient = df(x0)
        hessian = d2f(x0)
        d = gradient.shape[0]
        
        if linalg.matrix_rank(hessian) < d:
            print("Error in newton : hessian is not invertible. Stopping at the actual value.")
            break;
        
        fx0 = f(x0)

        if plot:
            x.append(x0)
            y.append(fx0)
        
        deltax = linalg.solve(hessian, gradient)
        
        x0 = x0 - deltax
        
        if verbose:
            print("Root = ", x0)

        if debug:
            print("Error = ", linalg.norm(gradient))
        
    if plot:

        x.append(x0)
        y.append(f(x0))

        if debug:
            print("x : ", x)
            print("y : ", y)

        plotIterations(f, x, y)

    return x0