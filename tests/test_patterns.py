import unittest
import random
import string
import functools

from mre import *

########################################
# utils

def word_class(grpis, context):
    """Corresponds to r'\w'."""
    chars = set()
    chars.update(string.ascii_letters, '_', string.digits)
    return CharClass(chars, grpis, context)

def space_class(grpis, context):
    """Corresponds to r'\s'."""
    return CharClass(set(string.whitespace), grpis, context)

def plus(child, grpis, context):
    """Corresponds to r'<child>+'."""
    return GreedyQuant(child, 1, None, grpis, context)

def star(child, grpis, context):
    """Corresponds to r'<child>*'."""
    return GreedyQuant(child, 0, None, gpri, context)

def qmark(child, grpis, context):
    """Corresponds to r'<child>?'."""
    return GreedyQuant(child, 0, 1, grpis, context)

def word_pattern(grpis, context):
    """Corresponds to r'\w+'."""
    return plus(word_class(None, context), grpis, context)

def spaces1(grpis, context):
    """Corresponds to r'\s+'."""
    return plus(space_class(None, context), grpis, context)

def concat(grpis, context, *args):
    """Corresponds to r'<r1><r2>...<rN>', where args[K] is the pattern of <rK>."""
    # Assumes that the concatenation operator is left-associative
    assert len(args) >= 2
    left = args[0]
    for right in args[1:-1]:
        left = Product(left, right, None, context)
    return Product(left, args[-1], grpis, context)

########################################
# Test cases

class Test(unittest.TestCase):
    def test_word_class(self):
        wp = word_class(None, Context())
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
        p = Literal("first", None, Context())
        m = p.match("first second third")
        self.assertTrue(m)

        m = p.match("First second third")
        self.assertFalse(m)

        p = Literal("first", [1], Context(1, IGNORECASE))
        m = p.search("second First third")

        self.assertTrue(m)
        self.assertEqual(m.group(), m.group(1), "First")

    def test_word_boundary(self):
        p = ZeroWidth.fromstr(r'\b', None, Context())
        m = p.match('Here')
        self.assertTrue(m)

        l = p.findall('H.e.l.l.o')
        self.assertEqual(l, ['']*10)

        m = p.match('abc', 3)
        self.assertTrue(m)

    def test_quant_with_char(self):
        # r'(-)+'
        context = Context(numgrps=1)
        child = Literal('-', [1], context)
        p = plus(child, None, context)
        l = p.findall('-a--b---c----d')
        self.assertEqual(l, ['-', '--', '---', '----'])

        m = p.match('-')
        self.assertEqual(m.group(), '-')

        m = p.search('---')
        self.assertEqual(m.group(), '---')

        # r'-{2,5}'
        context = Context()
        child = Literal('-', None, context)
        p = GreedyQuant(child, 2, 5, None, context)
        l = p.findall('-a--b---c----d-----e------f')
        self.assertEqual(l, ['-'*2, '-'*3, '-'*4, '-'*5, '-'*5])

        l = p.allstrs('xx------xx', 2)
        self.assertEqual(set(l), {'-' * k for k in range(2, 6)})

        l = p.allstrs('xx----xx', 2)
        self.assertEqual(set(l), {'-' * k for k in range(2, 5)})

    def test_word_pattern(self):
        # r'\w+
        p = word_pattern(None, Context())
        
        l = p.findall('.!-first|second..third')
        self.assertEqual(l, ['first', 'second', 'third'])

        l = p.allstrs('abcd.', 0)
        self.assertEqual(set(l), {'a', 'ab', 'abc', 'abcd'})

    def test_spaces1(self):
        context = Context()
        p = spaces1(None, context)
        l = list(m.group() for m in p.finditer('aaa bbb \t ccc \t\n'))
        self.assertEqual(l, [' ', ' \t ', ' \t\n'])

    def test_backref(self):
        # r'(\w+)-(\w+)'
        context = Context(numgrps=2)
        w1 = word_pattern([1], context)
        w2 = word_pattern([2], context)
        hyphen = Literal('-', None, context)
        p = concat(None, context, w1, hyphen, w2)

        m = p.search('xy--yx ab-cd gh-12')
        self.assertEqual(m.group(1,2), m.groups(), ('gh','12'))

        m = p.search('ab-cd gh-12', start=1)
        self.assertEqual(m.group(1,2), m.groups(), ('gh','12'))
        
    def test_double_word(self):
        # r'\b(\w+)\s+\1\b'
        def makeit(ignorecase):            
            context = Context(numgrps=1)
            flags = IGNORECASE if ignorecase else None
            context = Context(numgrps=1, flags=flags)
            b1 = ZeroWidth.fromstr(r'\b', [], context)
            b2 = ZeroWidth.fromstr(r'\b', [], context)
            word = word_pattern([1], context)
            s = space_class([], context)
            bref = BackRef(1, [], context)
            return concat([], context, b1, word, s, bref, b2)

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
