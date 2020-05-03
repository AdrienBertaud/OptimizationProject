# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 12:02:22 2020

@author: yanis
"""

import PlotIterations

from importlib import reload
reload(PlotIterations) # Relaod the module, in case it has changed

from PlotIterations import plotIterations


import numpy as np
import numpy.linalg as linalg

def secantd(f, df, x0, x1, N, verbose=True, plot=False, debug=False):
    
    if verbose:
        print("*** Secant method ***")

    x = []
    y = []

    if plot:
        x.append(x0)
        y.append(f(x0))
        x.append(x1)
        y.append(f(x1))

    x2 = x1
    
    for n in range(N):
        
        if debug:
            print("x0 = ", x0)
            print("x1 = ", x1)
        
        