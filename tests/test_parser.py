import unittest
import string
import random

from mre.parser import *
from mre import common

class TestTokenizer(unittest.TestCase):
    @staticmethod
    def noflags(regstr):
        # Just a helper
        return tokenize(regstr, common.emptyflags())

    @staticmethod
    def single(regstr, flags=common.emptyflags()):
        # Just a helper
        return tokenize(regstr, flags)[0]
    
    def test_char_class(self):
        cc = self.single('[abc]')
        self.assertEqual(cc.data, set('abc'))

        cc = self.single('[a-d]')
        self.assertEqual(cc.data, set('abcd'))

        cc = self.single('[a-zA-Z0-9]')
        self.assertEqual(cc.data, set(string.ascii_letters + string.digits))
        
        cc = self.single(r'[\x00-\xff]')
        self.assertEqual(cc.data, ALL)

        cc = self.single(r'[\d\d]')
        self.assertEqual(cc.data, set(string.digits))

        cc = self.single(r'[\w\d_-]')
        self.assertEqual(cc.data, set(string.digits + string.ascii_letters + '_-'))

        # ']' directly after '[' should not end the class.
        cc = self.single(r'[]abc]')
        self.assertEqual(cc.data, set(']abc'))

        cc = self.single(r'[^\da-g]')
        self.assertEqual(cc.data, ALL - (DIGIT | set('abcdefg')))

        # some oddness
        cc1, cc2, cc3, cc4, cc5 = self.noflags(r'[^-][-][a-c-][*--a][0-5-9]')
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
        result_types = [token.type for token in self.noflags(together)]
        self.assertEqual(types, result_types)

    def test_greedy_quant(self):
        quants = ['+', '?', '*', '{10,20}']
        tokens = self.noflags('+?*{10,20}')
        expected_bounds = [(1, None), (0, 1), (0, None), (10, 20)]
        for token, eb in zip(tokens, expected_bounds):
            self.assertEqual(token.type, 'greedy-quant')
            self.assertEqual(token.data, eb)

        expected_bounds = set(expected_bounds) # order no longer matters
        for k in range(24):
            random.shuffle(quants)
            tokens = self.noflags(''.join(quants))
            bounds = {token.data for token in tokens}
            self.assertEqual(bounds, expected_bounds)

class TestParser(unittest.TestCase):
    @staticmethod
    def nf(regstr):
        """Parse without flags."""
        return parse(regstr, common.emptyflags())[0]
    
    def test(self):
        et = self.nf('ab')
        self.assertEqual(et.token.type, 'product')
        arg1, arg2 = et.args
        self.assertEqual(arg1.token.type, arg2.token.type, 'char')
        self.assertEqual([arg1.token.data, arg2.token.data], ['a', 'b'])
        
def main():
    unittest.main(__name__)
