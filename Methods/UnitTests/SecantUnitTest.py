import unittest # https://docs.python.org/3/library/unittest.html

import sys
sys.path.append("..") # Adds higher directory to python modules path.

import Secant

from importlib import reload
reload(Secant) # Relaod the module, in case it has changed
# -*- coding: utf-8 -*-

from Secant import secant

import numpy as np

class Testsecant(unittest.TestCase):
        
    def test_const(self):
        self.assertEqual(secant(lambda x:1, 1, 1.3, 1, False), 1.3)
        
    def test_identity(self):
        self.assertEqual(secant(lambda x:x, 6, 5, 1, False), 0.0)
        
    def test_inversed_input(self):
        self.assertEqual(secant(lambda x:x, 5, 6, 1, False), 0.0)
        
    def test_x2_1_iter(self):
        self.assertAlmostEqual(secant(lambda x:x**2, 1, 1.1, 1, False), 0.5, places =1)
        
    def test_x2_10_iter(self):
        self.assertAlmostEqual(secant(lambda x:x**2, 1, 1.1, 10, False), 0, places =1)
        
    def test_x3(self):
        self.assertAlmostEqual(secant(lambda x:x**3, 1, 1.1, 10, 
                                      False, debug = False), 0.1, places =1)
    def test_verbose(self):
        self.assertAlmostEqual(secant(lambda x:x**2, 1, 1.1, N=2), 0.35, places =1)
        
    def test_debug(self):
        self.assertAlmostEqual(secant(lambda x:x**2, 1, 1.1, N=2, debug = False), 0.35, places =1)
    
    def test_tanh(self):
        self.assertAlmostEqual(first=secant(lambda x:np.tanh(x),
                                            x0=-10, x1=5, N=10, 
                                            verbose=False, plot = True, debug = False), 
                               second=0, places=0)
                
if __name__ == '__main__':
    unittest.main()