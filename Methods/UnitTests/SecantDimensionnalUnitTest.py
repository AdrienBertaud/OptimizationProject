# -*- coding: utf-8 -*-

import unittest # https://docs.python.org/3/library/unittest.html

import sys
sys.path.append("..") # Adds higher directory to python modules path.

import SecantDimensionnal
import Functions2D

from importlib import reload

# Relaod, in case of change
reload(SecantDimensionnal)
reload(Functions2D)

from SecantDimensionnal import secantD
from Functions2D import f, grad, start
from numpy import array

class TestSecantD(unittest.TestCase):

    def testQuadratic(self):
        result = secantD(f, grad, start, array([-2.0, 1.0]), N=10,
                                        verbose=True,  debug=True)
        self.assertAlmostEqual(result[0], 0, places=1)
        self.assertAlmostEqual(result[1], 0, places=1)

if __name__ == '__main__':
    unittest.main()