# -*- coding: utf-8 -*-
import sys
from numpy import array
from importlib import reload

sys.path.append("./Methods")
import Methods.QuasiNewton1
import Methods.Newton2Dim
reload(Methods.QuasiNewton1)
reload(Methods.Newton2Dim)
from Methods.QuasiNewton1 import quasiNewton_1
from Methods.Newton2Dim import newtonDim2

# define objective function
def f(x):
    x1 = x[0]
    x2 = x[1]
    f = x1**2 - 2.0 * x1 * x2 + 4 * x2**2
    return f

# define objective gradient
def grad(x):
    x1 = x[0]
    x2 = x[1]
    grad = []
    grad.append(2.0 * x1 - 2.0 * x2)
    grad.append(-2.0 * x1 + 8.0 * x2)
    return array(grad)

# Exact hessian
def H(x):
    return [[2.0, -2.0],[-2.0, 8.0]]

#initialize hessian
h0 = [[1, 0.0],[0.0, 1]]

# Start location
start = [-3.0, 2.0]

newtonDim2(f, grad, H, start, N=10, verbose=True, plot=True, debug=True)
quasiNewton_1(f, grad, h0, start, N=10, verbose=True, plot=True, debug=True)
