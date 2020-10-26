import unittest
import random
import string
import functools

from mre import common

from mre.patterns import *

# ----------------------------------------
# utils

def word_class(grpi, context):
    chars = set()
    chars.update(string.ascii_letters, '_', string.digits)
    return CharClass(chars, grpi, context)

def space_class(grpi, context):
    return CharClass(set(string.whitespace), grpi, context)

def plus(child, grpi, context):
    return GreedyQuant(child, 1, None, grpi, context)

def star(child, grpi, context):
    return GreedyQuant(child, 0, None, gpri, context)

def qmark(child, grpi, context):
    return GreedyQuant(child, 0, 1, grpi, context)

def word_pattern(grpi, context):
    # r'\w+'
    return plus(word_class(None, context), grpi, context)

def spaces1(grpi, context):
    # r'\s+'
    return plus(space_class(None, context), grpi, context)

def concat(grpi, context, *args):
    # Makes an assumption about associativity!
    assert len(args) >= 2
    left = args[0]
    for right in args[1:-1]:
        left = Product(left, right, None, context)
    return Product(left, args[-1], grpi, context)

def makestr(chars, grpi, context):
    charpats = (Char(char, None, context) for char in chars)
    return concat(grpi, context, *charpats)

# ----------------------------------------
# Test cases

class Test(unittest.TestCase):
    def test_word_class(self):
        wp = word_class(None, common.Context())
        m = wp.match('a')
        self.assertTrue(m)

        m = wp.match('1')
        self.assertTrue(m)

        m = wp.match('_')
        self.assertTrue(m)

        m = wp.match('-abc-')
        self.assertFalse(m)

        m = wp.search('-abc-')
        self.assertTrue(m)

        m = wp.search('')
        self.assertFalse(m)

        l = wp.findall('1-a-bz|_|ZB-A-0')
        self.assertEqual(l, list('1abz_ZBA0'))

    def test_concat_via_makestr(self):
        p = makestr("first", None, common.Context())
        m = p.match("first second third")
        self.assertTrue(m)

        m = p.match("First second third")
        self.assertFalse(m)

        p = makestr("first", 1, common.Context(1, common.IGNORECASE))
        m = p.search("second First third")

        self.assertTrue(m)
        self.assertEqual(m.group(), m.group(1), "First")

    def test_word_boundary(self):
        p = ZeroWidth.fromstr(r'\b', None, common.Context())
        m = p.match('Here')
        self.assertTrue(m)

        l = p.findall('H.e.l.l.o')
        self.assertEqual(l, ['']*10)

        m = p.match('abc', 3)
        self.assertTrue(m)

    def test_quant_with_char(self):
        # r'(-)+'
        context = common.Context(numgrps=1)
        child = Char('-', 1, context)
        p = plus(child, None, context)
        l = p.findall('-a--b---c----d')
        self.assertEqual(l, ['-', '--', '---', '----'])

        m = p.match('-')
        self.assertEqual(m.group(), '-')

        m = p.search('---')
        self.assertEqual(m.group(), '---')

        # r'-{2,5}'
        context = common.Context()
        child = Char('-', None, context)
        p = GreedyQuant(child, 2, 5, None, context)
        l = p.findall('-a--b---c----d-----e------f')
        self.assertEqual(l, ['-'*2, '-'*3, '-'*4, '-'*5, '-'*5])

        l = p.allstrs('xx------xx', 2)
        self.assertEqual(set(l), {'-' * k for k in range(2, 6)})

        l = p.allstrs('xx----xx', 2)
        self.assertEqual(set(l), {'-' * k for k in range(2, 5)})

    def test_word_pattern(self):
        # r'\w+
        p = word_pattern(None, common.Context())
        
        l = p.findall('.!-first|second..third')
        self.assertEqual(l, ['first', 'second', 'third'])

        l = p.allstrs('abcd.', 0)
        self.assertEqual(set(l), {'a', 'ab', 'abc', 'abcd'})

    def test_spaces1(self):
        context = common.Context()
        p = spaces1(None, context)
        l = list(m.group() for m in p.finditer('aaa bbb \t ccc \t\n'))
        self.assertEqual(l, [' ', ' \t ', ' \t\n'])

    def test_backref(self):
        # r'(\w+)-(\w+)'
        context = common.Context(numgrps=2)
        w1 = word_pattern(1, context)
        w2 = word_pattern(2, context)
        hyphen = Char('-', None, context)
        p = concat(None, context, w1, hyphen, w2)

        m = p.search('xy--yx ab-cd gh-12')
        self.assertEqual(m.group(1,2), m.groups(), ('gh','12'))

        m = p.search('ab-cd gh-12', start=1)
        self.assertEqual(m.group(1,2), m.groups(), ('gh','12'))
        
    def test_double_word(self):
        # r'\b(\w+)\s+\1\b'
        def makeit(ignorecase):            
            context = common.Context(numgrps=1)
            flags = common.I if ignorecase else None
            context = common.Context(numgrps=1, flags=flags)
            b1 = ZeroWidth.fromstr(r'\b', None, context)
            b2 = ZeroWidth.fromstr(r'\b', None, context)
            word = word_pattern(1, context)
            s = space_class(None, context)
            bref = BackRef(1, None, context)
            return concat(None, context, b1, word, s, bref, b2)

        p = makeit(False)
        m = p.match('the the')
        self.assertTrue(m)

        l = p.findall("""
        In November 2009, a a researcher at at the the Rey Juan Carlos
        University in Madrid found that the English Wikipedia had lost lost
        49,000 editors during the first three months of 2009 2009;""")
        self.assertEqual(
            l, ['a a', 'at at', 'the the', 'lost lost', '2009 2009'])

        p = makeit(True)
        m = p.match('abc AbC')
        self.assertEqual(m.group(), 'abc AbC')

        l = p.findall("""
        In November 2009, a a researcher at aT tHe the Rey Juan Carlos
        University in Madrid found that the English Wikipedia had LosT loSt
        49,000 editors during the first three months of 2009 2009;""")
        self.assertEqual(
            l, ['a a', 'at aT', 'tHe the', 'LosT loSt', '2009 2009'])

def main():
    unittest.main(__name__)
