import sys
import enum
import itertools
import string

from collections import namedtuple, deque, OrderedDict
from types import SimpleNamespace as NS
from pyllist import dllist

########################################
# public API

_cache = OrderedDict()
_CACHE_MAXSIZE = 100

def compile(regstr, flags=None):
    if flags is None:
        flags = RegexFlags(0)
    pattern = parse(regstr, flags)
    if len(_cache) == _CACHE_MAXSIZE:
        _cache.popitem(last=False)
    _cache[regstr] = pattern
    return pattern

def match(regstr, string, flags):
    p = compile(regstr, flags)
    return p.match(string)

def search(regstr, string, flags):
    p = compile(regstr, flags)
    return p.search(string)

def findall(regstr, string, flags):
    p = compile(regstr, flags)
    return p.findall(string)

def finditer(regstr, string, flags):
    p = compile(regstr, flags)
    return p.finditer(string)

########################################
# Patterns & Matches

class Match:
    """Base class for match objects.
    The attributes which all Match instances have are:
    - string: the string on which the match was made. match._mstr is a substring
      of this.
    - _mstr: the matched substring.
    - _grpis: a list of the group indexes of the match (often this list is empty
      or a singleton)
    - _groups: the Groups instance
    - _start, _end: these determine the span of the matched substring.
    - _is_exhausted: True when ._mstr has gone through all possible substrings
      of .string which match the pattern
    """

    def __new__(cls, astr, i, pattern):
        '''Assumes that (0 <= i <= len(astr)). Returns a match corresponding to
        the first substring of (astr) starting at (i) that matches (pattern), or
        None if such a string doesn't exist.'''
        raise NotImplementedError

    def _next(self):
        """When trying to match a pattern (P) at a given position (i) in a
        string (S), more than one such substrings are possible. When (self) is
        created, it corresponds to the first such substring. So, for example,
        (self._mstr) is that substring, (self._end) is where it ends,
        (self._start) is where it starts, etc. This is where _next comes in. If
        you want (self) to represent the next substring of (S) starting at (i)
        which matches (P), all you have to do is call (self._next()). If there
        is such a substring, (self._next()) will return True, and the state of
        (self) will change to correspond to that next substring
        (e.g. (self._mstr) and (self._end) will change). But if there is no such
        substring (i.e. when (self) corresponds to the last of the possible
        substrings that match (P) at (i)), then calling (self._next()) will
        return False and at that point (self) has been exhausted. Calling
        (self._next()) when (self) is exhausted raises a ValueError."""
        raise NotImplementedError

    def _check_exhausted(self):
        if self._is_exhausted:
            raise ValueError('Exhausted match object.')        

    ########################################
    # Public functions and their helpers thereof

    def __bool__(self):
        return True

    def __getitem__(self, i):
        return self.group(i)

    def _check_index(self, i):
        """A helper for the functions which make use of groups."""
        if type(i) is not int:
            raise TypeError(f'Index must be an int, not {type(i)}: {i}')
        if not 0 <= i <= len(self._groups):
            raise IndexError(f'No such group: {i}')

    def group(self, *indices):
        """If (not indices), equivalent to group(0). If (len(indices) == 1), let
        (i = indices[0]). If (i > N or i < 0), where (N) is the number of
        groups, an IndexError is raised. Otherwise, the matched string of the
        ith subregex is returned. If the ith subregex didn't match, (None) is
        returned. If it matched multiple times, the string corresponding to the
        last match is returned. If (len(indices) > 1), a tuple T of is returned,
        where (T[k] == group(indices[k])). So if (group(1) == 'first'),
        (group(2) == 'second'), (group(3) == 'third'), then (group(1, 3, 3) ==
        ('first', 'third', 'third')."""
        
        assert not self._is_exhausted
        def extract(i):            
            self._check_index(i)
            if i == 0:
                return self._mstr
            return self._groups.latest(i)
        if len(indices) == 0:
            return extract(0)
        elif len(indices) == 1:
            return extract(indices[0])
        else:
            return tuple(extract(i) for i in indices)

    def groups(self, default=None):
        """Mostly equivalent to (self.group(1, 2, ..., N)) where (N) is the
        number of groups. The difference is that if (group(k) is None), then
        (result[k] is default)."""
        mstrs = (self.group(i) for i in range(1, len(self._groups)+1))
        return tuple(default if mstr is None else mstr for mstr in mstrs)

    def span(self, i=0):
        return (self.start(i), self.end(i))

    def _boundary(self, i, hint):
        """Helper for (self.start) and (self.end). Finds the match (m) which has
        group number (i). If there is no such match, -1 is returned. If (hint ==
        'start'), the start index of (m) is returned, while if (hint == 'end'),
        the end index is returned."""
        
        assert not self._is_exhausted
        self._check_index(i)
        if i == 0:
            m = self
        else:
            m = self._groups.latest(i, hint='match')
            if m is None:
                return -1
        return m._start if hint == 'start' else m._end
    
    def start(self, i=0):
        """If (i) is not a valid group index, None is returned. If
        (self.group(i) is None), -1 is returned. Otherwise, the start index of
        (self.group(i)) within (self.string) is returned."""
        return self._boundary(i, 'start')

    def end(self, i=0):
        """If (i) is not a valid group index, None is returned. If
        (self.group(i) is None), -1 is returned. Otherwise, the end index of
        (self.group(i)) within (self.string) is returned."""
        return self._boundary(i, 'end')


class Pattern:
    """Base class for patterns.

    Patterns can have subpatterns, so they can form a tree. For example,
    r'(ab)*' Consists of 4 patterns total. A star quantifier, which has a single
    child, which is a concatenation operator which has two children, the letters
    'a' and 'b'.

    All patterns in a pattern tree share the same context, stored in the _context
    attribute, so that (p1._context is p2._context) for any two patterns p1,p2
    in the same tree.

    A pattern also has a list of group indexes (the _grpis) attribute, so that
    successful matches know where to put their matched strings.

    All pattern classes also have a _Match inner class. All matches generated
    from a pattern are instances of its _Match. These _Match classes are where
    most of the matching logic resides.
    """
    
    ########################################
    # _match functions
    
    """What is the difference between _match and _newmatch? _newmatch creates
    new groups for the pattern tree, whereas _match uses the current
    groups. _match is used during the matching process, e.g. by parents who need
    to match the child as part of their own matching logic. _newmatch is called
    only once, on the root pattern, to initiate the matching process."""
    
    def _match(self, astr, i):
        """If a substring of (astr) starting at (i) matches (self), the
        corresponding match object is returned. Otherwise, None."""
        assert 0 <= i <= len(astr)
        return self._Match(astr, i, self)
    
    def _newmatch(self, astr, i):
        self._context.initialize()
        return self._match(astr, i)
    
    ########################################
    # Public functions
    
    def match(self, astr, start=0):
        """If (self) matches at the beginning of (astr), the corresponding match
        object is returned. Otherwise, None."""
        return self._newmatch(astr, start)

    def search(self, astr, start=0, end=None):
        """Tries to match (self) at indexes [start, ..., end]. If not
        unsuccessful, None is returned."""
        if end is None:
            end = len(astr)
        if start > end:
            raise ValueError(f'(start <= end) must hold. Given {start},{end}')
        for i in range(start, len(astr) + 1):
            m = self._newmatch(astr, i)
            if m:
                return m
        return None

    def findall(self, astr):
        """Returns a list of all non-overlapping substrings of (astr) which
        match (self), from left to right."""
        return [m.group(0) for m in self.finditer(astr)]

    def finditer(self, astr):
        """Returns an iterator of all non-overlapping matches of (self) in
        (astr), from left to right."""
        i, len_astr = 0, len(astr) # current position within (astr)
        while i <= len_astr:
            m = self._newmatch(astr, i)
            if m:
                start, end = m.span()
                if start == end:
                    # empty match
                    i += 1
                else:
                    i = m._end
                yield m
            else:
                i += 1

    def allstrs(self, astr, i):
        """Returns a list of all of the substrings of (astr) starting at (i)
        which match (self). For example, if (self) corresponds to r'a|ab|abc',
        then (self.allstrs('--abcd--', 2)) will return ['a', 'ab', 'abc']."""
        m = self._newmatch(astr, i)
        if not m:
            return []
        out = [m.group()]
        while m._next():
            out.append(m.group())
        return out

    #################################################
    # Miscellaneous functions

    def _children(self):
        """Returns a list of the children Patterns of (self)."""
        return []

class Literal(Pattern):
    class _Match(Match):
        def __new__(cls, astr, i, pattern):
            m = object.__new__(cls)
            ignorecase = IGNORECASE in pattern._context.flags
            literal = pattern._literal
            substr = astr[i:i+len(literal)]
            matches = (literal.lower() == substr.lower()
                       if ignorecase else literal == substr)
            if matches:
                m.string = astr
                m._start = i
                m._end = i+len(literal)
                m._grpis = pattern._grpis
                m._groups = pattern._context.groups
                m._mstr = substr
                m._is_exhausted = False
                m._groups.add(m)
                return m
            else:
                return None

        def _next(self):
            self._check_exhausted()
            self._is_exhausted = True
            self._start = self._end = self._mstr = None
            self._groups.remove(self)
            return False

    def __init__(self, literal, grpis, context):
        self._literal = literal
        self._grpis = grpis
        self._context = context

class CharClass(Pattern):
    class _Match(Match):
        def __new__(cls, astr, i, pattern):            
            ignorecase = IGNORECASE in pattern._context.flags
            if i == len(astr):
                return None
            else:
                astr_char = astr[i]
                matches = ((astr_char.lower() if ignorecase else astr_char)
                           in pattern._chars)
                if matches:
                    m = object.__new__(cls)
                    m.string = astr
                    m._start = i
                    m._end = i + 1
                    m._groups = pattern._context.groups
                    m._grpis = pattern._grpis
                    m._mstr = astr_char
                    m._is_exhausted = False
                    m._groups.add(m)
                    return m
                else:
                    return None

        def _next(self):
            self._check_exhausted()
            self._is_exhausted = True
            self._mstr = self._start = self._end = None
            self._groups.remove(self)
            return False
                
    def __init__(self, chars, grpis, context):
        self._chars = (chars
                      if not IGNORECASE in context.flags
                      else {char.lower() for char in chars})
        self._grpis = grpis
        self._context = context


class ZeroWidth(Pattern):
    """All zero-width assertions follow a common matching algorithm: just check
    if a predicate (pred) at some position (i) within a string (S) holds, and if
    it does, return a Match object (m) with (m._mstr == '' and m._start ==
    m._end == i and not m._next()). Examples of zero-width assertions are word
    boundaries r'\b', caret '^' and dollar '$', and positive lookaheads
    (?=regex). All of those (and more) are handled by this class. The idea is
    that a predicate function (the _pred attribute of a ZeroWidth instance) is
    used to determine if the condition at (i) holds. Various class methods serve
    to create the predicate function and return the zero-width pattern
    corresponding to it."""
    
    class _Match(Match):
        def __new__(cls, string, i, pattern):
            if pattern._pred(string, i):
                m = object.__new__(cls)
                m.string = string
                m._mstr = ''
                m._grpis = pattern._grpis
                m._groups = pattern._context.groups
                m._start = m._end = i
                m._is_exhausted = False
                m._groups.add(m)
                return m
            else:
                return None

        def _next(self):
            self._check_exhausted()
            self._is_exhausted = True
            self._mstr = self._start = self._end = None
            self._groups.remove(self)
            return False
    
    @classmethod
    def lookahead(cls, pattern, positive, grpis, context):
        """Returns the lookahead zero-width pattern, with (pattern) being the
        pattern that is tested at the position. The boolean (positive)
        determines if the lookahead will be positive or negative."""
        if positive:
            pred = lambda string, i: bool(pattern._match(string, i))
        else:
            pred = lambda string, i: not bool(pattern._match(string, i))
        return cls(pred, grpis, context)

    @classmethod
    def fromstr(cls, letter, grpis, context):
        """Assumes (letter in {'^', '$', r'\b', r'\B', r'\A', r'\Z'}). Returns
        the corresponding zero width assertion pattern. For example, when
        (letter == r'\b'), returns the word boundary zero width pattern."""
        
        dct = {} # maps strings to predicates

        def predicate(letter):
            def decorator(func):
                dct[letter] = func
                return func
            return decorator

        @predicate(r'\b')
        def b(string, i):
            if i == 0 or i == len(string):
                return True
            c1, c2 = string[i-1], string[i]
            return not (c1.isalnum() and c2.isalnum())

        @predicate(r'\B')
        def B(string, i):
            return not b(string, i)

        @predicate(r'\A')
        def A(string, i):
            return i == 0

        @predicate(r'\Z')
        def Z(string, i):
            return i == len(string)

        if MULTILINE in context.flags:
            @predicate('^')
            def caret(string, i):
                return i == 0 or string[i-1] == '\n'
            @predicate('$')
            def dollar(string, i):
                return i == len(string) or string[i] == '\n'
        else:
            dct['^'], dct['$'] = A, Z

        pred = dct.get(letter)
        if pred is None:
            raise ValueError(f'The letter "{letter}" does not correspond to a zero-width assertion.')
        
        return cls(pred, grpis, context)
    
    def __init__(self, pred, grpis, context):
        self._pred = pred
        self._grpis = grpis
        self._context = context


class BackRef(Pattern):
    """Pattern for regexes of the form r'\<int>',
    where <int> is some positive integer."""
    
    class _Match(Match):
        def __new__(cls, astr, i, pattern):
            ref, groups = pattern._ref, pattern._context.groups
            ignorecase = IGNORECASE in pattern._context.flags
            mstr = groups.latest(ref)
            if mstr is None: return None
            end = i + len(mstr)
            substr = astr[i:end]
            matches = (mstr.lower() == substr.lower() if ignorecase
                       else mstr == substr)
            if matches:
                m = object.__new__(cls)
                m.string, m._mstr = astr, substr
                m._grpis, m._groups = pattern._grpis, groups      
                m._start, m._end = i, end
                m._is_exhausted = False
                m._groups.add(m)
                return m
            else:
                return None

        def _next(self):
            self._check_exhausted()
            self._is_exhausted = True
            self._mstr = self._start = self._end = None
            self._groups.remove(self)
            return False

    def __init__(self, ref, grpis, context):
        self._ref = ref
        self._grpis = grpis
        self._context = context


class Alternative(Pattern):
    """Corresponds to a regex of the form '<left-regex>|<right-regex>'.
    Attributes:
    - _left, _right: the left and right patterns, respectively. These patterns
      are also referred to as the child patterns.
    - _child: a match of one of the child patterns. _childleft determines which
      one.
    - _childleft: a boolean indicating if _child is a match of the left
      pattern."""
    
    class _Match(Match):
        def __new__(cls, astr, i, pattern):
            def init(child, on_left):
                m.string = astr
                m._grpis, m._groups = pattern._grpis, pattern._context.groups
                m._is_exhausted = False
                m._left, m._right = pattern._left, pattern._right
                m._on_left_child = on_left
                m._child = child
                m._sync_with_child()
                return m
                
            m = object.__new__(cls)
            child = pattern._left._match(astr, i)
            if child:
                return init(child, True)
            else:
                child = pattern._right._match(astr, i)
                if child:
                    return init(child, False)
                else:
                    return None

        def _sync_with_child(self,groups=False):
            """Assumes (self._child) is set. Always returns True."""
            c = self._child
            self._mstr, self._start, self._end = (c._mstr, c._start, c._end)
            if groups: self._groups.add(self)
                    
        def _next(self):
            self._check_exhausted()            
            if self._child._next():
                self._sync_with_child(groups=False)
                return True
            else:
                if self._on_left_child:
                    child = self._right._match(self.string, self._start)
                    if child:
                        self._on_left_child = False
                        self._child = child
                        return self._sync_with_child() # True
                else:
                    self._mstr = self._start = self._end = None
                    self._is_exhausted = True
                    self._groups.remove(self)
                    return False

    def __init__(self, left, right, grpis, context):
        self._left = left
        self._right = right
        self._grpis = grpis
        self._context = context
        
class GreedyQuant(Pattern):
    """All quantifiers are implemented by this class. This includes '*', '+',
    '?', '{m,n}'. A quantifier has a lower and upper bound. Here is a mapping
    from operators to bounds:
    - '*' -> (0, inf)
    - '+' -> (1, inf)
    - '?' -> (0, 1)
    - '{m,n}' -> (m, n)."""
    
    class _Match(Match):
        def __new__(cls, astr, i, pattern):
            m = object.__new__(cls)
            m.string = astr
            # no need to set _mstr and _end -- they are properties
            m._grpis, m._groups = pattern._grpis, pattern._context.groups
            m._start = i
            m._is_exhausted = False
            m._low, m._high = pattern._low, pattern._high
            m._children = []
            m._base = pattern._child            
            m._take()
            while len(m._children) < m._low:
                if not m._next_high():
                    return None
            m._groups.add(m)
            return m

        @property
        def _mstr(self):            
            if self._is_exhausted:
                return None
            return ''.join(child._mstr for child in self._children)

        @property
        def _end(self):
            if self._is_exhausted:
                return None
            if not self._children:
                return self._start
            return self._children[-1]._end
            
        def _next(self):
            self._check_exhausted()
            while self._next_high():
                if len(self._children) >= self._low:
                    self._groups.add(self)
                    return True
            self._is_exhausted = True
            self._start = None
            self._groups.remove(self)
            return False

        def _next_high(self):
            """Assumes (self) isn't exhausted. Updates (self) to the next match
            that satisfies the upper bound. Does not attempt to satisfy the
            lower bound. Returns True if successful, False otherwise."""
            if not self._children:
                return False
            child = self._children[-1]
            if child._next():
                self._take()
            else:
                self._children.pop()
            return True
        
        def _take(self):
            """This is where greed comes in. Keeps appending matches to
            (self._children) until (self._base) fails to match"""
            base, string, children = self._base, self.string, self._children
            i = self._end
            for k in range(len(children), self._high):
                child = base._match(string, i)
                if child is None:
                    break
                children.append(child)
                end = child._end
                if i == end:
                    # Empty child. Break to avoid an infinite loop.
                    break
                i = end

    def __init__(self, child, low, high, grpis, context):
        self._child = child
        self._low = low
        self._high = sys.maxsize if high is None else high
        self._grpis = grpis
        self._context = context

class Product(Pattern):
    """For regexes of the form '<left-regex><right-regex>'."""
    
    class _Match(Match):
        def __new__(cls, astr, i, pattern):
            left_match = pattern._left._match(astr, i)
            if not left_match:
                return None
            while True:
                right_match = pattern._right._match(astr, left_match._end)
                if right_match:
                    # initialize and return
                    m = object.__new__(cls)
                    m.string = astr
                    # _mstr and _end are properties, no need to assign them here.
                    m._grpis = pattern._grpis
                    m._groups = pattern._context.groups
                    m._start = i
                    m._is_exhausted = False
                    m._leftp = pattern._left
                    m._rightp = pattern._right
                    m._leftm = left_match
                    m._rightm = right_match
                    m._groups.add(m)
                    return m
                if not left_match._next():
                    return None

        @property
        def _mstr(self):
            return (None if self._is_exhausted
                    else self._leftm._mstr + self._rightm._mstr)

        @property
        def _end(self):
            return None if self._is_exhausted else self._rightm._end
            
        def _next(self):
            self._check_exhausted()
            if self._rightm._next():
                self._groups.add(self)
                return True
            else:
                while self._leftm._next():
                    rightmatch = self._rightp._match(
                        self.string, self._leftm._end)
                    if rightmatch:
                        self._rightm = rightmatch
                        self._groups.add(self)
                        return True
                else:
                    self._is_exhausted = True
                    self._start = None
                    self._groups.remove(self)
                    return False

    def __init__(self, left, right, grpis, context):
        self._left = left
        self._right = right
        self._grpis = grpis
        self._context = context

########################################
# Parsing

class _TokenList:
    """Helper class for the internal function (make_oprsargs). Used to
    efficiently iterate over the tokens."""
    
    def __init__(self, base, start=0, end=None):
        self.base = base
        self.start = start
        self.end = len(base) - 1 if end is None else end
        self.current = start

    def __next__(self):
        if self.current > self.end:
            raise StopIteration
        result = self.base[self.current]
        self.current += 1
        return result

    def subtokens(self):
        """A parenthesis token was just encountered. This returns a TokenList
        containing all tokens within the opening parenthesis that was just
        passed and its corresponding closing. After this call, (self)'s index
        will be on the token following the closing parenthesis."""
        
        start = self.current # the index of the token after the opening paren
        parsum = 1
        for token in self:
            if token.type.startswith('('):
                parsum += 1
            elif token.type == ')':
                parsum -= 1
                if parsum == 0:
                    break
        else:
            raise ValueError(f'Missing closing parenthesis.')

        # at this point, (self.current-1) is the index of the closing paren, so
        # we want the tokens from (start) to (self.current-2)
        return _TokenList(self.base, start, self.current-2)
    
    def __iter__(self):
        return self

def parse(regstr, flags):
    """Transforms (regstr) to a Pattern object."""
    return Parsing(regstr,flags).run()

class Parsing:
    # static attributes and methods
    # ════════════════════════════════════════
    
    # A set of the token types which correspond to unary operators. There is no need
    # to keep track of precedence or associativity, because only postfix unary
    # operators are used, and all unary operators have greater precedence than all
    # binary operators.
    unary_ops = {'greedy-quant'}
    
    @staticmethod
    def is_unary_op(token):
        return token.type in Parsing.unary_ops

    # Order operators based on precedence. Each element of the list contains an
    # associativity and a set of operators which have that associativity. All
    # operators in a (binops) element have the same precedence, and they are in
    # descending order of precedence.
    binary_ops = [
        NS(assoc='left', ops={'product'}),
        NS(assoc='left', ops={'|'})
    ]

    new_binary_ops = {}
    for binfo, prec in zip(binary_ops, range(len(binary_ops),0,-1)):
        assoc, ops = binfo.assoc, binfo.ops
        for op in ops:
            new_binary_ops[op] = NS(assoc=assoc, prec=prec)
    binary_ops = new_binary_ops
    
    @staticmethod
    def is_binary_op(token):
        return token.type in Parsing.binary_ops
    
    @staticmethod
    def token_category(token):
        t = token.type
        if t.startswith('(') or t.endswith(')'):
            return 'parenthesis'
        elif t in Parsing.binary_ops or t in Parsing.unary_ops:
            return 'operator'
        else:
            return 'primitive'

    # ════════════════════════════════════════
    
    def __init__(self, regstr, flags):
        self.regstr = regstr
        self.flags = flags
        self.context = Context(numgrps=0, flags=flags)
        self.tokens = None

    def run(self):
        tokens = _TokenList(tokenize(self.regstr,self.flags))
        return self._parse(tokens)
    
    def _parse(self, tokens):
        opsargs, unary_ops, binary_ops = self._make_opsargs(tokens)
        self._process_unary(opsargs, unary_ops)
        self._process_binary(opsargs, binary_ops)
        return opsargs.first.value

    def _make_opsargs(self, tokens):
        """In this function, the token sequence is transformed into a sequence
        of Patterns (the operands) and operators. During this stage, we also
        remember the positions of unary and binary operators for use in later
        parsing stages. In addition, upon return, the Context will be completed
        (e.g. the number of groups will be known).

        Intuition behind this function: I think of an expression, whether
        arithmetic or regex or whatever, as a sequence of operators and
        operands. This simply transforms the token sequence to match this view
        of expressions.

        Note: this function infers the location of product operators and adds
        them. When we have an operand or unary operator that is directly
        followed by another operand, a product operator must be inserted
        in-between. For example, r'ab' contains the tokens 'a' and 'b', but the
        expression is actually 'a<X>b', where <X> stands for the product
        operator. Also consider r'a+b'. The expression that is actually meant is
        r'(a+)<X>b'."""

        # a doubly linked list containing operands and their arguments.
        opsargs = dllist()
        # a list of the positions of the unary operators within (opsargs)
        unary_ops = []
        # a dict which maps precedences to pairs (assoc, posns), (posns) is
        # the positions of the binary operators which have that precedence
        # and (assoc) is their associativity.
        binary_ops = {}
        def add_unary(token):
            pos = opsargs.append(token)
            unary_ops.append(pos)
        def add_binary(token):
            info = Parsing.binary_ops[token.type]
            posn = opsargs.append(token)
            _, posns = binary_ops.setdefault(info.prec, (info.assoc, []))
            posns.append(posn)
        # The flag below is needed to determine if a product operator ought to
        # be inserted.
        last_is_expr_or_unop = False
        for token in tokens:
            category = Parsing.token_category(token)
            if category == 'primitive':
                pattern = self.primitive_to_Pattern(token)
                if last_is_expr_or_unop:
                    add_binary(Token('product', None))
                opsargs.append(pattern)
                last_is_expr_or_unop = True
            elif category == 'operator':
                if Parsing.is_unary_op(token):
                    add_unary(token)
                    last_is_expr_or_unop = True
                else:
                    add_binary(token)
                    last_is_expr_or_unop = False
            else: # (token) is a parenthesis
                pattern = self._parse(tokens.subtokens())
                paren = token.type
                if paren == '(':
                    self.context.numgrps += 1
                    pattern._grpis.append(self.context.numgrps)
                elif paren == '(?:':
                    pass
                elif paren in ('(?=', '(?!'):
                    pattern = self.lookahead_to_Pattern(paren, pattern)
                elif paren == ')':
                    raise ValueError(f'Superfluous closing parenthesis.')
                else:
                    raise AssertionError(f'This should not happen: {token}')
                if last_is_expr_or_unop:
                    add_binary(Token('product', None))
                opsargs.append(pattern)
                last_is_expr_or_unop = True

        binary_ops = [posn for precedence, posn in sorted(binary_ops.items(), reverse=True)]
        return opsargs, unary_ops, binary_ops

    def _process_unary(self, opsargs, unary_ops):
        """Assumes (opsargs) contains only Patterns and operator tokens. Some
        of those operators will be unary. This phase processes them out, so that
        what is left will be an alternating sequence of Patterns and binary
        operators. To achieve this, (unary_ops) is used. It must be a sequence of
        the positions of the unary operators."""

        for unary_pos in unary_ops:
            arg_pos = unary_pos.prev
            if arg_pos is None:
                raise ValueError(f'Missing unary operator argument.')
            operand = arg_pos.value
            if not isinstance(operand, Pattern):
                raise ValueError(f'Bad unary operator argument.')
            token = unary_pos.value
            pattern = self.unary_to_Pattern(token, operand)
            arg_pos.value = pattern
            opsargs.remove(unary_pos)

    def _process_binary(self, opsargs, binary_ops):
        """At this point, (opsargs) should be an alternating sequence of
        Patterns and binary operator tokens. The first and last elements should
        be Patterns. This function reduces (opsargs) to a single Pattern."""

        def get_operand(posn):
            """helper for the loop below."""
            if posn is None:
                raise ValueError(f'Missing operand.')
            value = posn.value
            if not isinstance(value, Pattern):
                raise ValueError(f'Bad operand.')
            return value

        def squeeze_posn(binop_posn, pattern):
            left, right = binop_posn.prev, binop_posn.next
            binop_posn.value = pattern
            opsargs.remove(left)
            opsargs.remove(right)

        def optimize(binop_posns):
            """Assumes all operators at (binop_posns) have the same precedence,
            and that this is currently the maximum precedence within the
            expression sequence."""
            operator_type = binop_posns[0].value.type
            if operator_type == 'product':
                # For example, thanks to this optimization the regex string
                # r'ab' will be parsed to a Literal pattern 'ab' instead of a
                # Product with children 'a' and 'b'. For a more elaborate
                # example, r'abc|xyz' will be parsed to an Alternative with
                # Literal children 'abc' and 'xyz', whereas without the
                # optimization, the children will be Products each with three
                # single character leaves.
                squeezed_posns = [] # to be removed from (binop_posns) after the loop
                for binop_posn in binop_posns:
                    operand1 = get_operand(binop_posn.prev)
                    operand2 = get_operand(binop_posn.next)
                    if (type(operand1) is type(operand2) is Literal
                        and not (operand1._grpis or operand2._grpis)):
                        new_literal = operand1._literal+operand2._literal
                        pattern = Literal(new_literal, [], self.context)
                        squeeze_posn(binop_posn, pattern)
                        squeezed_posns.append(binop_posn)
                for posn in squeezed_posns:
                    binop_posns.remove(posn)
            elif operator_type == '|':
                # Thanks to this optimization, the regex string r'[0-9]|[a-z]'
                # will compile to the CharClass r'[0-9a-z]', which is
                # semantically equivalent.
                squeezed_posns = [] # to be removed from (binop_posns) after the loop
                for binop_posn in binop_posns:
                    operand1 = get_operand(binop_posn.prev)
                    operand2 = get_operand(binop_posn.next)
                    if (type(operand1) is type(operand2) is CharClass
                        and not (operand1._grpis or operand2._grpis)):
                        new_charset = operand1._chars | operand2._chars
                        pattern = CharClass(new_charset, [], self.context)
                        squeeze_posn(binop_posn, pattern)
                        squeezed_posns.append(binop_posn)
                for posn in squeezed_posns:
                    binop_posns.remove(posn)
            else:
                raise AssertionError('This should never happen.')

        for assoc, binop_posns in binary_ops:
            if assoc == 'right':
                binop_posns.reverse()
            optimize(binop_posns)
            for binop_posn in binop_posns:
                operand1 = get_operand(binop_posn.prev)
                operand2 = get_operand(binop_posn.next)
                pattern = self.binary_to_Pattern(binop_posn.value, operand1, operand2)
                squeeze_posn(binop_posn, pattern)
    
    def primitive_to_Pattern(self,token):
        """(token) is a token of a primitive regex. This function returns the
        corresponding Pattern."""
        type = token.type
        if type == 'char':
            return Literal(token.data, [], self.context)
        elif type == 'bref':
            return BackRef(token.data, [], self.context)
        elif type == 'char-class':
            return CharClass(token.data, [], self.context)
        elif type in ('^', '$', r'\A', r'\Z', r'\b', r'\B'): # zero-width assertions
            return ZeroWidth.fromstr(type, [], self.context)
        else:
            raise AssertionError('This should never happen.')

    def lookahead_to_Pattern(self, paren, pattern):
        """(token)'s type is one of "(?=" or "(?!". (pattern) is the lookahead's
        internal regex. This function returns the Pattern corresponding to the
        lookahead."""
        positive = paren == "(?="
        return ZeroWidth.lookahead(pattern, positive, [], self.context)

    def unary_to_Pattern(self, token, operand):
        """(token) corresponds to a unary operator, with (operand) as its operand
        Pattern. This returns the corresponding Pattern."""
        type = token.type
        if type == 'greedy-quant':
            return GreedyQuant(operand, *token.data, [], self.context)
        else:
            raise AssertionError('This should never happen.')

    def binary_to_Pattern(self, token, operand1, operand2):
        """(token) is a binary operator, with (operand1) and (operand2) being
        Patterns that are its operands. This function forms the pattern
        corresponding to the binary operator."""
        if token.type == 'product':
            return Product(operand1, operand2, [], self.context)
        elif token.type == '|':
            return Alternative(operand1, operand2, [], self.context)
        else:
            raise AssertionError('This should never happen.')

########################################
## Tokenization

Token = namedtuple('Token', 'type data')

"""
Tokenization here is not a strictly syntactical operation. It also does some
preliminary processing (like determining the characters in a character set, the
bounds of a greedy quantifier and more.)

The character set used is {chr(i) for i in range(2**7)}. The idea is to support
all ASCII characters. I chose this set for simplicity; the purpose of this
project is to learn, nobody will really use it.

A token is represented via the Token class (see above). An instance of Token has
a (type) and (data) attributes. The type says what kind of token this is
(e.g. "char", "greedy-quant", etc.) while the data gives additional information
that pertain to this specific token. For example, the "data" of a token
representing a character class is the set of characters defined by the class.

Here are the possible tokens:
* operators
** product :: Token('product', None)
e.g. when we have r'<regex1><regex2>', the product operator is implied between them
*** syntax
There is no syntax for this operator. It is implicit when there are two adjacent
operands. Tokenization never produces this token, it is inserted during parsing.
** alternative :: Token('|', None)
*** syntax: Just a '|' character.
** greedy quantifiers :: Token('greedy-quant', (low, high))
*** How the different operators map to 'greedy-quant' tokens. Also shows their syntax.
- '*' -> Token('greedy-quant', (0, inf))
- '+' -> Token('greedy-quant', (1, inf))
- '?' -> Token('greedy-quant', (0, 1))
- '{<int1>,<int2>}' -> Token('greedy-quant', (int(<int1>), int(<int2>)))
* operands
** single character :: Token('char', CHARACTER)
*** syntax :: a non-special character or a special character preceded by backlash
** backreference :: Token('bref', <group-index>)
*** syntax :: "\<int-1-99>"
In here <int-1-99> stands for one of the strings ["1", "2", ..., "99"]
** character class :: Token('char-class', <set-of-characters>)
*** syntax
- "." -> Token('char-class', ALL or DOT_NO_ALL)
- "\d" -> Token('char-class', DIGITS)
- "\D" -> Token('char-class', ALL-DIGITS)
- "\w" -> Token('char-class', ALPHANUMERIC)
- "\W" -> Token('char-class', ALL-ALPHANUMERIC)
- "\s" -> Token('char-class', WHITESPACE)
- "\S" -> Token('char-class', ALL-WHITESPACE)
- '[<spec>]' -> Token('char-class', <set-defined-by-spec>)
  The finer points of <spec> are defined in the (char_class) function.
** zero-width assertions
*** syntax
All of them are literals: "^", "$", "\A", "\b", "\B", "\Z". The tokens are
eponymous and their data member is None. Example tokens are Token("^", None),
Token(r"\Z", None), etc.
* parenthesis
** syntax
All are literals: "(", "(?:", "(?=", "(?!", ")". All tokens are eponymous and
their data member is None. Example tokens are Token("(", None), Token("(?!",
None), etc.
"""

def tokenize(regstr,flags):
    return Tokenization(regstr,flags).run()

# character sets
# ════════════════════
ALL = frozenset(chr(i) for i in range(2**7)) # whole character set.
DOT_NO_ALL = ALL-{'\n'} # dot characters without DOTALL flag.
DIGIT = frozenset(string.digits) & ALL
ALPHANUMERIC = frozenset(string.ascii_letters + string.digits + '_') & ALL
WHITESPACE = frozenset(string.whitespace) & ALL
# shorthands for character classes
CLASS_SHORTS = {'d': DIGIT,
                'D': ALL - DIGIT,
                's': WHITESPACE,
                'S': ALL - WHITESPACE,
                'w': ALPHANUMERIC,
                'W': ALL - ALPHANUMERIC}
SPECIAL = r'\|+*?{}[].^$()'
HEX_DIGITS = f'0123456789abcdefABCDEF'
ESCAPE_CHARS = {'a': '\a', 'f': '\f', 'n': '\n', 'r': '\r',
               't': '\t', 'v': '\v', 'b': '\b'}
# ════════════════════

class Tokenization:
    def __init__(self,regstr,flags):
        self.regstr = regstr
        self.pos = 0
        self.flags = flags
        self.token_funcs = [
            self.simple,
            self.char_class,
            self.greedy_quant,
            self.char_class_shorts,
            self.backref,
            self.char]

    def run(self):
        """Transforms (regstr) into a list of tokens and returns it. (flags) is the
        RegexFlags. Raises ValueError if (regstr) is flawed."""
        tokens = []
        while not self.is_exhausted():
            for tf in self.token_funcs:
                token = tf()
                if token is not None:
                    tokens.append(token)
                    break
            else:
                raise ValueError(f'Cannot extract a token from "{self.rest()}"')
        return tokens
        
    def is_exhausted(self):
        return self.pos == len(self.regstr)

    def takech(self,inc=False):
        if self.is_exhausted():
            return None
        ch = self.regstr[self.pos]
        if inc:
            self.pos += 1
        return ch

    def rest(self):
        return self.regstr[self.pos:]

    def _error(self, msg):
        """A helper"""
        raise ValueError(f'{msg}: "{self.regstr}"')

    def simple(self):
        """For tokens whose type matches the characters in the regex string, and
        whose data is None."""
        # In the tuple below, make sure that if (t1) is a prefix of (t2), (t1) comes
        # after (t2). Consider an example which violates this and the consequence:
        # If '(' comes before '(?=', if '(?=' is a prefix of the regex string at
        # (self.pos), the token that will be extracted is '(', not '(?=' as should be
        # the case.
        types = ('^', '$', r'\A', r'\b', r'\B', r'\Z', '|',
                 '(?:', '(?=','(?!', '(', ')')
        for t in types:
            if self.regstr.startswith(t, self.pos):
                self.pos += len(t)
                return Token(type=t, data=None)
        else:
            return None

    def char_class(self):
        """Tries to extract a character class from the regex string. Returns a token
        of the form Token(type='char-class', data=<chars>), where <chars> is the set
        of characters in the class.

        Character class syntax: A spec is what appears inside the brackets enclosing
        the class (e.g. 'a-z' in [a-z]). I will describe the fine points related to how
        a spec maps to the character set that makes up the class. All of the rest is
        the same as in Python.

        There are a couple of points to consider:

        1. Caret in the beginning

        If the spec begins with '^', the char set will be the negation of that
        determined by the rest of the spec. As a special case, a spec of '^'
        raises an error, because '' is not a valid spec. For the rest of this
        docstring, assume the spec does not begin with '^'.

        2. Hyphens

        Hyphens have two purposes:
        - They stand for themselves. This happens in the following cases:
          - When they appear at the beginning or end of the spec. As a special
            case, '[^-]' is the set of all chars except '-'
          - When they are preceded by an unescaped backlash.
          - When they can't possibly stand for a range. For example, in '0-5-9',
            '0-5' is a range, but the hyphen that follows can't be one, because it
            makes no sense to have a range whose lower bound is another range. So
            the second hyphen stands for itself to result in the charset
            set('012345-9').
        - They delimit ranges.

        3. Character class shorthands, like r'\d', r'\w', etc. If a letter which
           stands for a char-class shorthand is preceded by an unescaped backlash,
           the resulting class will contain the characters of the inner class.

        4. backlashes. Normal escape characters can be included. Also, the
           characters special to char classes (r'\[]^-') can be included by
           backlashing them.
        """

        if self.takech() != '[':
            return None
        # Find index of closing ']' and store it in (cbi). Start at self.pos+2
        # because a beginning ']' does not indicate the end of the class.
        cbi = self.regstr.find(']', self.pos+2)
        while True:
            if cbi == -1:
                raise ValueError(f'No closing bracket: {self.rest()}')
            if self.regstr[cbi-1] == '\\':
                cbi = self.regstr.find(']', cbi+1)
            else:
                break
        spec = self.regstr[self.pos+1:cbi]
        self.pos = cbi + 1
        return Token('char-class', self._spec_to_set(spec))

    def _spec_to_set(self, spec):
        """Processes the insides of a character class (spec) into a set of
        characters."""
        assert spec # (spec) should not be empty

        def error(msg):
            # just a helper
            raise ValueError(f'Bad character class spec "{spec}": {msg}')

        tempiter = iter(spec)
        ########################################    
        # stage 1: take care of initial '^'.
        negate = False
        first = next(tempiter)
        if first == '^':
            negate = True
        else: 
            tempiter = itertools.chain([first], tempiter) # put (first) back

        tokens = deque()
        ########################################
        # stage 2: take care of backlashes. Every element of (tokens) after this
        # will be a character or a set of characters or a '--'. Tokens in this
        # context have nothing to do with regex tokens that are the result of
        # (tokenize).
        for char in tempiter:
            if char == '-':
                tokens.append('--')
            elif char == '\\':
                nxt = next(tempiter, None)
                if nxt is None:
                    error('Missing character after backlash.')
                elif nxt == '-':
                    tokens.append('-')
                elif nxt in CLASS_SHORTS:
                    tokens.append(CLASS_SHORTS[nxt])
                elif nxt in ESCAPE_CHARS:
                    tokens.append(ESCAPE_CHARS[nxt])
                elif nxt == 'x':
                    h1, h2 = next(tempiter, None), next(tempiter, None)
                    if (h1 is None or h1 not in HEX_DIGITS
                        or h2 is None or h2 not in HEX_DIGITS):
                        error('Expected two hex digits after "\\x"')
                    tokens.append(chr(int(h1+h2, 16)))
                else:
                    error(f'Bad char after "\\": "{char}"')
            else:
                tokens.append(char)
        ########################################
        # stage 3: process spans (like 'a-z') and shorthands. 
        result = set()
        while tokens:
            token = tokens.popleft()
            if token == '--':
                result.add('-')
            elif type(token) in (set, frozenset):
                result.update(token)
            else: # (token) is a character
                if len(tokens) < 2: # (token) is not part of a span
                    result.add(token)
                else:
                    nxt = tokens.popleft()
                    if nxt == '--':
                        span_end = tokens.popleft()
                        if type(span_end) is set:
                            error('Invalid input after "-".')
                        elif span_end == '--':
                            span_end = '-'
                        chars = [chr(k) for k in range(ord(token), ord(span_end)+1)]
                        if not chars:
                            error(f'Bad range {token}-{span_end}.')
                        result.update(chars)
                    else:
                        tokens.appendleft(nxt) # put (nxt) back
                        result.add(token)
        ########################################
        return result if not negate else ALL-result

    def greedy_quant(self):
        def error(extra=None):
            # just a helper
            msg = f'Invalid quantifier "{self.rest()}"'
            if extra is not None:
                msg = f'{msg}: {extra}'
            raise ValueError(msg)

        ch = self.takech()
        bounds = {'*': (0, None), '+': (1, None), '?': (0, 1)}.get(ch)
        if bounds is not None:
            self.pos += 1
            return Token(type='greedy-quant', data=bounds)
        elif ch == '{':
            endi = self.regstr.find('}', self.pos)
            if endi == -1:
                error('Missing closing "{".')
            bounds = self.regstr[self.pos+1:endi].split(',')
            if not len(bounds) in (1, 2):
                error()
            if len(bounds) == 1:
                try: low = int(bounds[0])
                except ValueError:
                    error("Bounds must be parsable integers.")
                high = low
            elif not bounds[1].strip():
                try: low = int(bounds[0])
                except ValueError:
                    error("Bounds must be parsable integers.")
                high = None
            else:
                try:
                    low, high = map(int, bounds)
                except ValueError:
                    error("Bounds must be parsable to integers.")
                if low > high:
                    error(f'<low> must not exceed <high>.')
            self.pos = endi+1
            return Token('greedy-quant', (low, high))
        else:
            return None

    def char_class_shorts(self):
        """Handles character class shorthands, like r'\d'. The dot is also
        considered a shorthand."""
        ch = self.takech(True)
        chars = None # the set of characters
        if ch == '.':
            chars = (ALL if DOTALL in self.flags
                     else DOT_NO_ALL)
        elif ch == '\\':
            nxt = self.takech(True)
            if nxt is None:
                self._error(f"Can't end in backlash.")
            chars = CLASS_SHORTS.get(nxt)
            if chars is None:
                self.pos -= 2 # put (ch) and (nxt).
                return None
        else:
            self.pos -= 1 # put back (ch)
            return None
        return Token('char-class', chars)

    def backref(self):
        ch = self.takech()
        if ch != '\\':
            return None
        self.pos += 1
        nxt = self.takech()
        if nxt is None:
            self._error(f"Can't end in backlash.")
        if nxt.isdigit():
            return Token('bref', int(self.extract_digits()))
        else:
            self.pos -= 1 # put back (ch)
            return None

    def extract_digits(self):
        """Helper function. Assumes the current regex string char is a
        digit. Extracts all digits from the current position, and returns
        them. Increments (self.pos) in the process, so that it will point to the
        first non-digit char."""
        digits = []
        ch = self.takech()
        while True:
            digits.append(ch)
            self.pos += 1
            if self.is_exhausted():
                break
            ch = self.takech()
            if not ch.isdigit():
                break
        return ''.join(digits)

    def char(self):
        """Forms 'char' tokens. Call after other tokenization functions so that
        special characters are not interpreted as regular ones. For example, if the
        current char of the regex string is '*', this will interpret it as a
        character token rather than as a quantifier. As another example, this
        function will raise an error with r'\A' because 'A' is not a valid escape
        character, even though r'\A' is a valid regex. So if this is called before
        the function that processes r'\A' regexes, an error will be raised."""

        ch = self.takech(inc=True)
        if ch == '\\':
            nxt = self.takech(inc=True)
            if nxt is None:
                self._error('Cannot end in backlash.')
            elif nxt in SPECIAL:
                return Token('char', nxt)
            elif nxt == 'x':
                x1, x2 = self.takech(True), self.takech(True)
                if (x1 is None or x1 not in HEX_DIGITS
                    or x2 is None or x2 not in HEX_DIGITS):
                    self._error('Expected two hex digits after "\\x"')
                return Token('char', chr(int(x1+x2, 16)))
            elif nxt in ESCAPE_CHARS:
                return Token('char', ESCAPE_CHARS[nxt])
            else:
                self._error(f'"{nxt}" is not a valid escape character, at {self.pos}.')
        else:
            return Token('char', ch)

########################################
# Flags

class RegexFlags(enum.Flag):
    IGNORECASE = I = enum.auto()
    MULTILINE = M = enum.auto()
    DOTALL = S = enum.auto()

# make the flags available at the top level
for name, flag in RegexFlags.__members__.items():
    globals()[name] = flag
    
########################################
# Context

class Context:
    """
    A Context contains the global information needed by the nodes of a pattern
    tree during matching. For example, all nodes share the same flags and
    groups. The flags and groups can be used as part of the matching logic of
    any pattern and subpattern of the pattern tree to which the context
    pertains.

    Contexts enable the implementation of backreferencing. For example r'\3'
    could be implemented by simply referring to (context.groups[3]), where
    (context) belongs to the backreference pattern.

    A context is also used as an indirection layer to allow new groups and flags
    to be quickly updated for all subpatterns globally. Since all subpatterns
    alias the same context instance, creating new groups to be shared by all is
    simply done by changing (context.groups) in one subpattern.
    """
    
    def __init__(self, numgrps=0, flags=None):
        self.groups = None
        self.numgrps = numgrps
        self.flags = RegexFlags(0) if flags is None else flags

    def initialize(self):
        """Create the list (result = [None] + odicts) and bind it to
        (self.groups). (odicts) is a list of OrderedDicts of length
        (self.numgrps). Parenthesis are numbered starting from 1, so this
        allows to reference the proper ordered dict using
        (result[parenthesis_index])."""        
        self.groups = Groups(self.numgrps)
        
########################################
# Groups

class Groups:
    def __init__(self, N):
        self._N = N
        self._lst = [None]
        self._lst.extend(OrderedDict() for k in range(N))

    def __len__(self):
        return self._N

    def latest(self, i, hint='str'):
        """Returns the lates string (when (hint == 'str')) or match (when (hint
        == 'match')) with group index (i). If there is nothing stored at (i),
        None is returned."""
        assert hint in ('str', 'match')
        odict = self._lst[i]
        if not odict: return None
        match = next(reversed(odict.keys()))
        return match._mstr if hint == "str" else match

    def _odicts(self, match):
        for grpi in match._grpis:
            yield self._lst[grpi]
    
    def add(self, match):
        """Assumes (match) is not exhausted"""
        if match._grpis is not None:
            for odict in self._odicts(match):
                odict[match] = True
        return True
    
    def remove(self, match):
        """Assumes (self) is not exhausted"""
        if match._grpis is not None:
            for odict in self._odicts(match):
                odict.pop(match, None)
        return False
