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
from Functions2D import f, grad, start

class TestNewtonG(unittest.TestCase):

    # def testQuadratic(self):
    #     result = Greenstadt(f, grad, H, start, N=10,
    #                                     verbose=True,  debug=True)
    #     self.assertAlmostEqual(result[0], 0, places=1)
    #     self.assertAlmostEqual(result[1], 0, places=1)

if __name__ == '__main__':
    unittest.main()