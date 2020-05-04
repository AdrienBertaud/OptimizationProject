# -*- coding: utf-8 -*-

import unittest # https://docs.python.org/3/library/unittest.html

import sys
sys.path.append("..") # Adds higher directory to python modules path.

import QuasiNewton_Greenstadt
import Functions2D

from importlib import reload

# Relaod, in case of change
reload(QuasiNewton_Greenstadt)
reload(Functions2D)

from QuasiNewton_Greenstadt import Greenstadt
from Functions2D import f, grad, start, H

class TestNewtonG(unittest.TestCase):

    def testQuadratic(self):
        H0 = H(start)
        result = Greenstadt(f, grad, start, H0, N=10)
        self.assertAlmostEqual(result[0][-1][0], 0, places=1)
        self.assertAlmostEqual(result[0][-1][1], 0, places=1)

if __name__ == '__main__':
    unittest.main()