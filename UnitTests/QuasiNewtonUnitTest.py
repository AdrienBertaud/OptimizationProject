import unittest # https://docs.python.org/3/library/unittest.html

import sys
sys.path.append("..") # Adds higher directory to python modules path.

import QuasiNewton

from importlib import reload
reload(QuasiNewton) # Relaod the module, in case it has changed

from QuasiNewton import quasiNewton_1

class TestNewton(unittest.TestCase):
                
    def test_const(self):
        self.assertEqual(quasiNewton_1(lambda x:1, lambda x:0, 1.3, 1, False), 1.3)
    
    def test_identity(self):
        self.assertEqual(quasiNewton_1(lambda x:x, lambda x:1, 1, 1, False), 0.0)
        
    def test_x2_1_iter(self):
        self.assertEqual(quasiNewton_1(lambda x:x**2, lambda x:2*x, 1, 1, False), 0.5)
        
    def test_x2_10_iter(self):
        self.assertAlmostEqual(quasiNewton_1(lambda x:x**2, lambda x:2*x, 1, 10, False), 0.0, places =2)
        
    def test_verbose(self):
        self.assertEqual(quasiNewton_1(lambda x:x**2, lambda x:2*x, 1, 2), 0.25)
        
if __name__ == '__main__':
    unittest.main()