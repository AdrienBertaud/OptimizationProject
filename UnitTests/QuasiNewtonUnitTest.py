# -*- coding: utf-8 -*-

import unittest # https://docs.python.org/3/library/unittest.html

import sys
sys.path.append("..") # Adds higher directory to python modules path.

import QuasiNewton

from importlib import reload
reload(QuasiNewton) # Relaod the module, in case it has changed

from QuasiNewton import quasiNewton_1

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

#initialize hessian
h0 = [[1, 0.0],[0.0, 1]]

# Start location
start = [-3.0, 2.0]

class TestNewton(unittest.TestCase):

    def test(self):
        result = quasiNewton_1(f, grad, h0, start, N=10,
                                       verbose=True, plot=True, debug=True)

        self.assertAlmostEqual(result[0], 0, places=1)
        self.assertAlmostEqual(result[1], 0, places=1)

if __name__ == '__main__':
    unittest.main()