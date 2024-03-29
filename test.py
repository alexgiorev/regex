import unittest
import string
import random
import sys
import os.path

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

class TestPatterns(unittest.TestCase):
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

class TestTokenizer(unittest.TestCase):
    @staticmethod
    def noflags(regstr):
        # Just a helper
        return tokenize(regstr, RegexFlags(0))

    @staticmethod
    def single(regstr, flags=RegexFlags(0)):
        # Just a helper
        return tokenize(regstr, flags)[0]
    
    def test_char_class(self):
        cc = self.single('[abc]')
        self.assertEqual(cc.data, set('abc'))

        cc = self.single('[a-d]')
        self.assertEqual(cc.data, set('abcd'))

        cc = self.single('[a-zA-Z0-9]')
        self.assertEqual(cc.data, set(string.ascii_letters + string.digits))
        
        cc = self.single(r'[\x00-\x7f]')
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

class Test(unittest.TestCase):
    """General tests"""
    
    def test1(self):
        p = compile("^The")
        m = p.match("The table is clean.")
        self.assertTrue(m)
        self.assertTrue(m.group(0), "The")

    def test2(self):
        p = compile("abc|xyz", IGNORECASE)
        lst = p.findall('--aBc-_.h--XyZ--ABC-abcxYZ-')
        expected = ['aBc', 'XyZ', 'ABC', 'abc', 'xYZ']
        self.assertEqual(expected, lst)

    def test3(self):
        p = compile(r'^The\s+end$')
        self.assertEqual(p.match('The end').group(0), 'The end')
        self.assertEqual(p.match('The  end').group(0), 'The  end')
        self.assertEqual(p.match('The   end').group(0), 'The   end')
        self.assertFalse(p.match('Theend'))
        self.assertFalse(p.match('The end is coming'))

    def test4(self):
        p = compile(r'abc*')
        self.assertEqual(p.match('ab').group(0), 'ab')
        self.assertEqual(p.match('abc').group(0), 'abc')
        self.assertEqual(p.match('abcccc').group(0), 'abcccc')
        self.assertEqual(p.match('abcabc').group(0), 'abc')

    def test5(self):
        p = compile('((abc))|(xyz)')
        m1, m2 = p.finditer('The first three letters are "abc" and the last three are "xyz"')
        self.assertEqual(m1.group(1), m1.group(2), 'abc')
        self.assertEqual(m1.group(3), None)
        self.assertEqual(m2.group(1), m2.group(2), None)
        self.assertEqual(m2.group(3), 'xyz')

    def test_identifier(self):
        test_string = 'return Literal(token1.data, [], pns.context) + some_word'
        expected = ['return', 'Literal', 'token1', 'data', 'pns', 'context', 'some_word']
        
        p = compile('[a-zA-Z_][a-zA-Z_0-9]*')
        lst = p.findall(test_string)
        self.assertEqual(lst, expected)

        # shortened version
        p = compile('\w[\w\d]*')
        lst = p.findall(test_string)
        self.assertEqual(lst, expected)

        # to test parenthesis and "alternative" operator
        p = compile('\w(?:\w|\d)*')
        lst = p.findall(test_string)
        self.assertEqual(lst, expected)

    def test_string_literal(self):
        p = compile(r'"(\\"|[^"])*"')
        m = p.search(r'He said "I am not sure" and then he left.')
        self.assertEqual(m.group(0), '"I am not sure"')

        m = p.search(r'She said "to quote someone, surround what they say with \" and thats it" to her students')
        self.assertEqual(m.group(0), r'"to quote someone, surround what they say with \" and thats it"')        

    def test_double_word(self):
        p = compile(r'\b(\w+)\s+\1\b')
        l = p.findall("""
        In November 2009, a a researcher at at the the Rey Juan Carlos
        University in Madrid found that the English Wikipedia had lost lost
        49,000 editors during the first three months of 2009 2009;""")
        self.assertEqual(
            l, ['a a', 'at at', 'the the', 'lost lost', '2009 2009'])

        p = compile(r'\b(\w+)\s+\1\b', IGNORECASE)
        m = p.match('abc AbC')
        self.assertEqual(m.group(), 'abc AbC')

        l = p.findall("""
        In November 2009, a a researcher at aT tHe the Rey Juan Carlos
        University in Madrid found that the English Wikipedia had LosT loSt
        49,000 editors during the first three months of 2009 2009;""")
        self.assertEqual(
            l, ['a a', 'at aT', 'tHe the', 'LosT loSt', '2009 2009'])

    def test_anything_but_123(self):
        p = compile(r'(?!123)\d{3}')
        lst = p.findall('first = 123, second = 456, third = 789')
        self.assertEqual(lst, ['456', '789'])

    def test_word_with_t(self):
        p = compile(r'(?=T)\w*', IGNORECASE)
        lst = p.findall('This is the text, my text.')
        self.assertEqual(['This', 'the', 'text', 'text'], lst)

    def test_email(self):
        # Source: https://www.regular-expressions.info/email.html
        p = compile(r'\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b', IGNORECASE)
        lst = p.findall('My new email is alex.giorev@gmail.com, my old one was algi@abv.bg')
        self.assertEqual(lst, ['alex.giorev@gmail.com', 'algi@abv.bg'])

    def test_IP(self):
        # Source: https://www.regular-expressions.info/ip.html
        byte = r'(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)'
        regstr = fr'\b{byte}\.{byte}\.{byte}\.{byte}\b'
        p = compile(regstr)
        text = """IP addresses are written and displayed in human-readable
        notations, such as 172.16.254.1 in IPv4, and 2001:db8:0:1234:0:567:8:1
        in IPv6. The size of the routing prefix of the address is designated in
        CIDR notation by suffixing the address with the number of significant
        bits, e.g., 192.168.1.15/24, which is equivalent to the historically
        used subnet mask 255.255.255.0."""
        m = p.search(text)
        self.assertEqual(m.group(1,2,3,4), ('172', '16', '254', '1'))
        lst = p.findall(text)
        self.assertEqual(lst, ['172.16.254.1', '192.168.1.15', '255.255.255.0'])

    def test_float(self):
        # Source: https://www.regular-expressions.info/floatingpoint.html
        regstr = r'[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?'
        text = '''String.format("%.2f", 1.2399) // returns "1.24e-5"
	String.format("%.3f", 1.2399) // returns "1.240e17"'''
        p = compile(regstr)
        lst = p.findall(text)
        self.assertEqual(lst, ['.2', '1.2399', '1.24e-5', '.3', '1.2399', '1.240e17'])

    def test_hex(self):
        p = compile('0x[\da-f]+', IGNORECASE)
        text = """In programming, a number of notations are used to denote
        hexadecimal numbers, usually involving a prefix or suffix. The prefix 0x
        is used in C and related programming languages, which would denote this
        value as 0x67C2. Here are more examples: 0X12, 0xffff, 0x1234abcdF"""
        self.assertEqual(p.findall(text), ['0x67C2', '0X12', '0xffff', '0x1234abcdF'])

unittest.main(__name__)
