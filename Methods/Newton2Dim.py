# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

import sys
# Adds directories to python modules path.
sys.path.append("./Utils")

import Utils.PlotIterations

from importlib import reload
reload(Utils.PlotIterations) # Relaod the module, in case it has changed

def newtonDim2(f, grad, H, X0, N, verbose=True, plot=False, debug=False):

    """
        Nexton method in 2 dimensions.
    """

    if debug:
        print("X0 = ", X0)

    # Initialize X
    X = np.zeros((N+1,2))
    X[0] = X0

    for i in range(1,N):

        HInv = np.linalg.pinv(H(X[i-1]))

        X[i] = X[i-1] - np.dot(HInv, grad(X[i-1]))

    if debug:
        print("X = ", X)

    if plot:
        plt.plot(X[:,0],X[:,1],'r-o')

        # Save the figure as a PNG
        plt.savefig('Newton_2dim.png')

        plt.show()

    if verbose:
        print("Minimum = ", X[N])

    return X[N]