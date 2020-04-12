import os
import unittest
from unittest import TestCase

import numpy as np

print("IMPORTED")

def run_all_tests(test_mod=None,tests=None):
    if tests is None:
        tests = unittest.TestLoader().loadTestsFromModule(test_mod)
    print(tests)
    unittest.TextTestRunner(verbosity=2).run(tests)
    
class testCache(TestCase):

    def setUp(self):
        np.random.seed(seed=0)

    def test(self):
        self.assertEqual(3,1)

    def test2(self):
        self.assertEqual(2,1)

    def test3(self):
        self.assertEqual(1,1)

if __name__ == "__main__":
    
    #run_all_tests(testCache)
    unittest.main()
