# -*- coding: utf-8 -*-

import unittest # https://docs.python.org/3/library/unittest.html

import sys
sys.path.append("..") # Adds higher directory to python modules path.

import NewtonDimensionnal

from importlib import reload
reload(NewtonDimensionnal) # Relaod the module, in case it has changed

from NewtonDimensionnal import newtond

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

class TestNewton_2Dim(unittest.TestCase):

    def testQuadratic(self):
        result = newtond(f, grad, H, start, N=10,
                                        verbose=True,  debug=True)
        self.assertAlmostEqual(result[0], 0, places=1)
        self.assertAlmostEqual(result[1], 0, places=1)

    def testNonInvertible(self):
        result = newtond(f, grad, HNonInv, start, N=10,
                                        verbose=True,  debug=True)

        self.assertAlmostEqual(result[0], start[0], places=1)
        self.assertAlmostEqual(result[1], start[1], places=1)

if __name__ == '__main__':
    unittest.main()