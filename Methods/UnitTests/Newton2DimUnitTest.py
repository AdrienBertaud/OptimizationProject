# -*- coding: utf-8 -*-

import unittest # https://docs.python.org/3/library/unittest.html

import sys
sys.path.append("..") # Adds higher directory to python modules path.

import Newton2Dim

from importlib import reload
reload(Newton2Dim) # Relaod the module, in case it has changed

from Newton2Dim import newtonDim2

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
    return grad

# Exact hessian
def H(x):
    return [[2.0, -2.0],[-2.0, 8.0]]

# Start location
start = [-3.0, 2.0]

class TestNewton_2Dim(unittest.TestCase):

    def test(self):
        result = newtonDim2(f, grad, H, start, N=10,
                                       verbose=True, plot=True, debug=True)

        self.assertAlmostEqual(result[0], 0, places=1)
        self.assertAlmostEqual(result[1], 0, places=1)

if __name__ == '__main__':
    unittest.main()