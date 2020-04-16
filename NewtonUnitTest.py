import unittest # https://docs.python.org/3/library/unittest.html
from importlib import reload
import Newton
reload(Newton)
from Newton import newton

class TestNewton(unittest.TestCase):
        
    def test_identity(self):
        self.assertEqual(newton(lambda x:x, lambda x:1, 1, 1, False), 0.0)
        
    def test_const(self):
        self.assertEqual(newton(lambda x:1, lambda x:0, 1.3, 1, False), 1.3)
        
    def test_x2_1_iter(self):
        self.assertEqual(newton(lambda x:x**2, lambda x:2*x, 1, 1, False), 0.5)
        
    def test_x2_10_iter(self):
        self.assertAlmostEqual(newton(lambda x:x**2, lambda x:2*x, 1, 10, False), 0.0, places =2)
        
if __name__ == '__main__':
    unittest.main()