import unittest
import string
import random

from mre.parser import *

def token(regstr, flags=common.emptyflags()):
    # Just a helper
    return tokenize(regstr, flags)[0]

def toknoflags(regstr):
    return tokenize(regstr, common.emptyflags())

class Test(unittest.TestCase):
    def test_char_class(self):
        cc = token('[abc]')
        self.assertEqual(cc.data, set('abc'))

        cc = token('[a-d]')
        self.assertEqual(cc.data, set('abcd'))

        cc = token('[a-zA-Z0-9]')
        self.assertEqual(cc.data, set(string.ascii_letters + string.digits))
        
        cc = token(r'[\x00-\xff]')
        self.assertEqual(cc.data, ALL)

        cc = token(r'[\d\d]')
        self.assertEqual(cc.data, set(string.digits))

        cc = token(r'[\w\d_-]')
        self.assertEqual(cc.data, set(string.digits + string.ascii_letters + '_-'))

        # ']' directly after '[' should not end the class.
        cc = token(r'[]abc]')
        self.assertEqual(cc.data, set(']abc'))

        cc = token(r'[^\da-g]')
        self.assertEqual(cc.data, ALL - (DIGIT | set('abcdefg')))

        # some oddness
        cc1, cc2, cc3, cc4, cc5 = toknoflags(r'[^-][-][a-c-][*--a][0-5-9]')
        self.assertEqual(cc1.data, ALL - set('-'))
        self.assertEqual(cc2.data, set('-'))
        self.assertEqual(cc3.data, set('abc-'))
        self.assertEqual(cc4.data, set('*+,-a'))
        self.assertEqual(cc5.data, set('012345-9'))

    def test_simple(self):
        types = ['^', '$', r'\A', r'\b', r'\B', r'\Z', '|', '(', '(?:',
                 '(?=','(?!', ')']
        random.shuffle(types) # order shouldn't matter.
        together = ''.join(types)
        result_types = [token.type for token in toknoflags(together)]
        self.assertEqual(types, result_types)

    def test_greedy_quant(self):
        tokens = toknoflags('+?*{10,20}')
        expected_bounds = [(1, None), (0, 1), (0, None), (10, 20)]
        for token, eb in zip(tokens, expected_bounds):
            self.assertEqual(token.type, 'greedy-quant')
            self.assertEqual(token.data, eb)

def main():
    unittest.main(__name__)
