import unittest
import random

from .. import common

from ..patterns import *

# ----------------------------------------
# utils

def word_pattern(grpi, context):
    chars = set()
    chars.update(string.ascii_letters, '_', string.digits)
    return CharClass(chars, grpi, context)

# ----------------------------------------
# Test cases

class TestSimple(unittest.TestCase):
    def test_word_pattern(self):
        wp = word_pattern(None, common.Context())
        m = wp.match('a')
        self.assertTrue(m)
        

# def TestBackRef(unittest.TestCase):
#     def test(self):
#         r'\b(\w+)'
        
def main():
    unittest.main()
