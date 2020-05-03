# -*- coding: utf-8 -*-
from numpy import array

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

# Start location
start = [-3.0, 2.0]

# Non invertible hessian
def HNonInv(x):
    return [[0.0, 0.0],[-2.0, 0.0]]

# Start location
start = [-3.0, 2.0]