# -*- coding: utf-8 -*-

import unittest # https://docs.python.org/3/library/unittest.html

import sys
sys.path.append("..") # Adds higher directory to python modules path.

import Newton

from importlib import reload
reload(Newton) # Relaod the module, in case it has changed

from Newton import newton
from Newton import newtonDim2

class TestNewton_1Dim(unittest.TestCase):

    def test_identity(self):
        self.assertEqual(newton(lambda x:x, lambda x:1, 1, 1, False), 0.0)

    def test_const(self):
        self.assertEqual(newton(lambda x:1, lambda x:0, 1.3, 1, False), 1.3)

    def test_x2_1_iter(self):
        self.assertEqual(newton(lambda x:x**2, lambda x:2*x, 1, 1, False), 0.5)

    def test_x2_10_iter(self):
        self.assertAlmostEqual(newton(lambda x:x**2, lambda x:2*x, 1, 10, False), 0.0, places =2)

    def test_verbose(self):
        self.assertEqual(newton(lambda x:x**2, lambda x:2*x, 1, 2), 0.25)

    def test_plot(self):
        self.assertAlmostEqual(newton(lambda x:x**2, lambda x:2*x, 1, N=5, plot=True), 0.03, places =2)

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