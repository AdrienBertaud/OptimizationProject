# -*- coding: utf-8 -*-

import unittest # https://docs.python.org/3/library/unittest.html

import sys
sys.path.append("..") # Adds higher directory to python modules path.

import NewtonDimensionnal
import Functions2D

from importlib import reload

# Relaod, in case of change
reload(NewtonDimensionnal)
reload(Functions2D)

from NewtonDimensionnal import newtond
from Functions2D import f, grad, H, HNonInv, start

class TestNewtonD(unittest.TestCase):

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