import sys
import enum
import itertools
import string

from collections import namedtuple, deque, OrderedDict
from types import SimpleNamespace as NS

########################################
# Interface

def compile(regstr, flags):
    raise NotImplementedError

def search(regstr, string, flags):
    raise NotImplementedError

def match(regstr, string, flags):
    raise NotImplementedError

def findall(regstr, string, flags):
    raise NotImplementedError

def finditer(regstr, string, flags):
    raise NotImplementedError

########################################
# Patterns & Matches

class Match:
    """Base class for match objects.
    The attributes which all Match instances have are:
    - string: the string on which the match was made. match._mstr is a substring
      of this.
    - _mstr: the matched substring.
    - _groupi: the group index of the parent pattern.
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

    A pattern also has a group index (the _groupi) attribute, so that successful
    matches know where to put their matched strings.

    All pattern classes also have a _Match inner class. All matches generted
    from a pattern are instances of its _Match. These _Match classes are where
    most of the matching logic resides.
    """

    # ----------------------------------------
    # _match functions
    
    """What is the difference between _match and _newmatch? _newmatch creates new
    groups for the pattern tree, whereas _match uses the current groups. _match is
    used during the matching process, e.g. by parents who need to match the child as
    part of their own matching logic. _newmatch is called only once during the
    matching process of the root pattern."""
    
    def _match(self, astr, i):
        """If a substring of (astr) starting at (i) matches (self), the
        corresponding match object is returned. Otherwise, None."""
        assert 0 <= i <= len(astr)
        return self._Match(astr, i, self)
    
    def _newmatch(self, astr, i):
        self._context.initialize()
        return self._match(astr, i)
    
    # ----------------------------------------
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


class Literal(Pattern):
    class _Match(Match):
        """Apart from the attributes which all match objects have, there are no extra ones."""
        
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
                m._groupi = pattern._groupi
                m._groups = pattern._context.groups
                m._mstr = literal
                m._is_exhausted = False
                m._groups.add(m)
                return m
            else:
                return None

        def _next(self):
            self._check_exhausted()
            self._is_exhausted = True
            self._start = self._end = self._mstr = None
            return self._groups.remove(self)

    def __init__(self, literal, groupi, context):
        self._literal = literal
        self._groupi = groupi
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
                    m._groupi = pattern._groupi
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
            return self._groups.remove(self)
                
    def __init__(self, chars, groupi, context):
        self._chars = (chars
                      if not IGNORECASE in context.flags
                      else {char.lower() for char in chars})
        self._groupi = groupi
        self._context = context


class ZeroWidth(Pattern):
    """All zero-width assertions follow a common matching algorithm: just check
    if a predicate (pred) at some position (i) within a string (S) holds, and if
    it does, return a Match object (m) with (m._mstr == '' and m._start ==
    m._end == i and not m._next()). Examples of zero-width assertions are word
    boundaries r'\b', caret '^' and dollar '$', and positive lookaheads
    (?=regex). All of those (and more) are handled by this class. The idea is
    that a predicate function (the _pred attribute of a ZeroWidth instance) is
    used to determine if the condition at (i) holds. Various class methods in
    this class serve to create the predicate function and return the zero-width
    pattern corresponding to it."""
    
    class _Match(Match):
        def __new__(cls, string, i, pattern):
            if pattern._pred(string, i):
                m = object.__new__(cls)
                m.string = string
                m._mstr = ''
                m._groupi = pattern._groupi
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
            return self._groups.remove(self)
    
    @classmethod
    def lookahead(cls, pattern, positive, groupi, context):
        """Returns the lookahead zero-width pattern, with (pattern) being the
        pattern that is tested at the position. The boolean (positive)
        determines if the lookahead will be positive or negative."""
        pred = (lambda string, i: bool(pattern._match(string, i))
                if positive
                else lambda string, i: not bool(pattern._match(string, i)))        
        return cls(pred, groupi, context)

    @classmethod
    def fromstr(cls, letter, groupi, context):
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
        
        return cls(pred, groupi, context)
    
    def __init__(self, pred, groupi, context):
        self._pred = pred
        self._groupi = groupi
        self._context = context


class BackRef(Pattern):
    """Pattern for regexes of the form r'\<int>', where <int> is some positive
    integer."""
    
    class _Match(Match):
        def __new__(cls, astr, i, pattern):
            ref, groups = pattern._ref, pattern._context.groups
            ignorecase = IGNORECASE in pattern._context.flags
            mstr = groups.latest(ref)
            if mstr is None:
                return None
            end = i + len(mstr)
            substr = astr[i:end]
            matches = (mstr.lower() == substr.lower() if ignorecase
                       else mstr == substr)
            if matches:
                m = object.__new__(cls)
                m.string, m._mstr = astr, substr
                m._groupi, m._groups = pattern._groupi, groups      
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
            return self._groups.remove(self)

    def __init__(self, ref, groupi, context):
        self._ref = ref
        self._groupi = groupi
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
        """Exhausted when (self._m is None and not self._atleft)."""
        def __new__(cls, astr, i, pattern):
            def init(left):
                m.string = astr
                m.groupi, m.groups = pattern._groupi, pattern._context.groups
                m._is_exhausted = False
                m._left = m._right = pattern._left, pattern._right
                m._childleft = left
                m._child = child
                m._refresh()
                return m
                
            m = object.__new__(cls)
            child = pattern._left._match(astr, i)
            if child:
                return init(left=True)
            else:
                child = pattern._right._match(astr, i)
                if child:
                    return init(left=False)
                else:
                    return None

        def _refresh(self):
            """Assumes (self._child) is set. Always returns True."""
            c = self._child
            self._mstr, self._start, self._end = (c._mstr, c._start, c._end)
            return self._groups.add(self) # True
                    
        def _next(self):
            self._check_exhausted()            
            if self._child._next():
                return self._refresh() # True
            else:
                if self._childleft:
                    child = self._right._match(self.string, self._start)
                    if child:
                        self._childleft = False
                        self._child = child
                        return self._refresh() # True
                else:
                    self._mstr = self._start = self._end = None
                    self._is_exhausted = True
                    return self._groups.remove(self)

    def __init__(self, left, right, groupi, context):
        self._left = left
        self._right = right
        self._groupi = groupi
        self._context = context

        
class GreedyQuant(Pattern):
    """All quantifiers are implemented by this class. This includes '*', '+',
    '?', '{m,n}'. A quantifier has a lower and upper bound. Here is a mapping
    from operators to bounds:
    - '*' -> (0, inf)
    - '+' -> (1, inf)
    - '?' -> (0, 1)
    - '{m,n}' -> (m, n).
    Keep in mind that (inf) doesn't really mean infinity. It may in the future,
    but for now it stands for (sys.maxsize).

    Extra attributes:
    - _low, _high: the bounds
    - _children: a list of matches or None when exhausted.
    - _base: the pattern used to form the children."""
    
    class _Match(Match):
        def __new__(cls, astr, i, pattern):
            m = object.__new__(cls)
            m.string = astr
            # no need to set _mstr and _end; they are properties
            m._groupi, m._groups = pattern._groupi, pattern._context.groups
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

        # _mstr and _end as properties may be slow under certain
        # circumstances. TODO: Valuate this design.
        
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
                    return self._groups.add(self)
            self._is_exhausted = True
            self._start = None
            return self._groups.remove(self)

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
            """This is where greed comes in. Tries to append as many matches as
            possible to (self._children)."""
            # For performance reasons, take some attributes.
            base, string, children = self._base, self.string, self._children
            i = self._end
            for k in range(len(children), self._high):
                child = base._match(string, i)
                if child is None:
                    break
                children.append(child)
                end = child._end
                if i == end:
                    break
                i = end

    def __init__(self, child, low, high, groupi, context):
        self._child = child
        self._low = low
        self._high = sys.maxsize if high is None else high
        self._groupi = groupi
        self._context = context


class Product(Pattern):
    """For regexes of the form '<left-regex><right-regex>'."""
    
    class _Match(Match):
        """Extra attributes:
        - _leftpattern, _rightpattern: correspond to <left-regex> and <right-regex> above.
        - _leftchild, _rightchild: match objects derived from _leftpattern and
        _rightpattern, respectively.
        """
        
        def __new__(cls, astr, i, pattern):
            leftmatch = pattern._left._match(astr, i)
            if not leftmatch:
                return None
            while True:
                rightmatch = pattern._right._match(astr, leftmatch._end)
                if rightmatch:
                    # initialize and return
                    m = object.__new__(cls)
                    m.string = astr
                    # _mstr and _end are properties, no need to assign them here.
                    m._groupi = pattern._groupi
                    m._groups = pattern._context.groups
                    m._start = i
                    m._is_exhausted = False
                    m._leftpattern = pattern._left
                    m._rightpattern = pattern._right
                    m._leftchild = leftmatch
                    m._rightchild = rightmatch
                    m._groups.add(m)
                    return m
                if not leftmatch._next():
                    return None

        @property
        def _mstr(self):
            return (None if self._is_exhausted
                    else self._leftchild._mstr + self._rightchild._mstr)

        @property
        def _end(self):
            return None if self._is_exhausted else self._rightchild._end
            
        def _next(self):
            self._check_exhausted()
            if self._rightchild._next():
                return self._groups.add(self)
            else:
                while self._leftchild._next():
                    rightmatch = self._rightpattern._match(
                        self.string, self._leftchild._end)
                    if rightmatch:
                        self._rightchild = rightmatch
                        return self._groups.add(self)
                else:
                    self._is_exhausted = True
                    self._start = None
                    return self._groups.remove(self)

    def __init__(self, left, right, groupi, context):
        self._left = left
        self._right = right
        self._groupi = groupi
        self._context = context

########################################
# Parsing

Token = namedtuple('Token', 'type data')
ExprTree = namedtuple('ExprTree', 'token args grpis')

# A set of the token types which correspond to unary operators. There is no need
# to keep track of precedence or associativity, because only postfix unary
# operators are used, and all unary operators have greater precedence than all
# binary operators.
unops = {'greedy-quant'}

def isunop(token):
    return token.type in unops

# Order operators based on precedence. Each element of the list contains an
# associativity and a set of operators which have that associativity. All
# operators in a (binops) element have the same precedence, which is less than
# that of previous elements' operators, and greater than that of next elements'
# operators. In other words, operators are in descending precedence order.
binops = [
    NS(assoc='left', ops={'product'}),
    NS(assoc='left', ops={'|'})
]

# What follows is a script which transforms (binops) into a dict which maps to a
# binary operator it's precedence and associativity. The current version of
# (binops) is easy for humans to manipulate, the new version will be easy for
# programs to query.

new_binops = {}
for binfo, prec in zip(binops, range(len(binops), 0, -1)):
    assoc, ops = binfo.assoc, binfo.ops
    for op in ops:
        new_binops[op] = NS(assoc=assoc, prec=prec)
binops = new_binops

def isbinop(token):
    return token.type in binops

def binfo(token):
    return binops[token.type]

def token_category(token):
    """Tokens can be grouped into 3 categories:
    - primitives: letters, zero-width assertions, etc.
    - operators.
    - parenthesis.
    This functions returns one of ('primitive', 'operator', 'parenthesis')
    """
    
    t = token.type
    if t.startswith('(') or t.endswith(')'):
        return 'parenthesis'
    elif t in unops or t in binops:
        return 'operator'
    else:
        return 'primitive'

class _TokenList:
    """Used to efficiently iterate over the tokens."""
    
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
        """A paren token was just encountered. This returns a TokenList
        containing all tokens within opening paren that was just passed and it's
        corresponding closing. After this call, (self)'s index will be on the
        token following the closing parenthesis."""
        
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

# global parsing namespace. Contains data that is useful for parsing related
# functions. Initialized at each call to (parse).
pns = NS(context=None)

def parse(regstr, flags):
    """Creates the expression tree and the context."""
    pns.context = Context(0, flags)
    tokens = _TokenList(tokenize(regstr, flags))
    return _parse(tokens), pns.context

def _parse(tokens):
    """Parses (tokens) to an ExprTree. Fills out the context. Raises ValueError
    if there is a problem."""
    
    def make_opsargs():
        """In this function, the token sequence is transformed into a sequence
        of ExprTrees (the operands) and operators. During this pass, we also
        remember the positions of unary and binary operators for use in later
        parsing stages. In addition, upon return, the context will be completed
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
        unops = []
        
        # a dict which maps precedences to pairs (assoc, posns), where (assoc)
        # is the associativity of the binary operators at (posns).
        binops = {}

        def add_unop(token):
            pos = opsargs.append(token)
            unops.append(pos)
        
        def add_binop(token):
            bi = binfo(token)
            pos = opsargs.append(token)
            _, posns = binops.setdefault(bi.prec, (bi.assoc, []))
            posns.append(pos)
        
        # The flag below is needed to determine if a product operator ought to
        # be inserted.
        last_is_expr_or_unop = False
            
        for token in tokens:
            category = token_category(token)
            if category == 'primitive':
                subexpr = ExprTree(token, args=None, grpis=[])
                if last_is_expr_or_unop:
                    add_binop(Token('product', None))
                opsargs.append(subexpr)
                last_is_expr_or_unop = True
            elif category == 'operator':
                if isunop(token):
                    add_unop(token)
                    last_is_expr_or_unop = True
                else:
                    add_binop(token)
                    last_is_expr_or_unop = False
            else: # is a parenthesis
                subexpr = _parse(tokens.subtokens())
                paren = token.type
                if paren == '(':
                    pns.context.numgrps += 1
                    subexpr.grpis.append(pns.context.numgrps)
                elif paren == '(?:':
                    pass
                elif paren in ('(?=', '(?!'):
                    subexpr = ExprTree(token, args=[subexpr], grpis=[])
                elif paren == ')':
                    raise ValueError(f'Unneccessary closing parenthesis.')
                else:
                    raise AssertionError(f'This should not happen: {token}')
                if last_is_expr_or_unop:
                    add_binop(Token('product', None))
                opsargs.append(subexpr)
                last_is_expr_or_unop = True

        binops = [item[1] for item in sorted(binops.items(), reverse=True)]        
        return opsargs, unops, binops

    def process_unary_operators(opsargs, unops):
        """Assumes (opsargs) contains only (ExprTree)s and operator tokens. Some
        of those operators will be unary. This phase processes them out, so that
        what is left will be an alternating sequence of (ExprTree)s and binary
        operators. To achieve this, (unops) is used. It must be a sequence of
        the positions of the unary operators."""
        
        for unop_pos in unops:
            arg_pos = unop_pos.prev
            if arg_pos is None:
                raise ValueError(f'Missing unary operator argument.')
            operand = arg_pos.value
            if type(operand) is not ExprTree:
                raise ValueError(f'Bad unary operator argument.')
            subexpr = ExprTree(token=unop_pos.value, # the operator token
                               args=[operand], grpis=[])
            arg_pos.value = subexpr
            opsargs.remove(unop_pos)
                
    def process_binary_operators(opsargs, binops):
        """At this point, (opsargs) should be an alternating sequence of
        ExprTrees and binary operator tokens. The first and last elements should
        be ExprTrees. This function reduces (opsargs) to a single ExprTree."""

        def getarg(pos):
            """helper for the loop below."""
            if pos is None:
                raise ValueError(f'Missing operand.')
            value = pos.value
            if type(value) is not ExprTree:
                raise ValueError(f'Bad operand.')
            return value
        
        for assoc, binop_posns in binops:
            if assoc == 'right':
                binop_posns = reversed(binop_posns)
            for binop_pos in binop_posns:
                left, right = binop_pos.prev, binop_pos.next
                arg1, arg2 = getarg(left), getarg(right)
                subexpr = ExprTree(token=binop_pos.value,
                                   args=[arg1, arg2],
                                   grpis=[])
                binop_pos.value = subexpr
                opsargs.remove(left)
                opsargs.remove(right)

    opsargs, unops, binops = make_opsargs()
    process_unary_operators(opsargs, unops)
    process_binary_operators(opsargs, binops)
    return opsargs.first.value

# ----------------------------------------
# Tokenization

ALL = frozenset(chr(i) for i in range(2**7)) # whole character set.
DOT_NO_ALL = ALL-{'\n'} # dot characters without DOTALL flag.
DIGIT = frozenset(string.digits)
ALPHANUMERIC = frozenset(string.ascii_letters + string.digits + '_')
WHITESPACE = frozenset(string.whitespace)
# shorthands for character classes
CLASS_SHORTS = {'d': DIGIT,
                'D': ALL - DIGIT,
                's': WHITESPACE,
                'S': ALL - WHITESPACE,
                'w': ALPHANUMERIC,
                'W': ALL - ALPHANUMERIC}
SPECIAL = r'\[].^()|+*?{}'
HEX_DIGITS = f'0123456789abcdefABCDEF'
ESCAPE_CHARS = {'a': '\a', 'f': '\f', 'n': '\n', 'r': '\r',
               't': '\t', 'v': '\v', 'b': '\b'}

"""
Tokenization here is not a strictly syntactical operation. It also does some
preliminary processing (like determining the characters in a character set, the
bounds of a greedy quantifier and more.)

The character set used is {chr(i) for i in range(256)}, so all characters whose
encoding can fit in a byte. The idea is to support all ASCII characters. I chose
this set for simplicity; the purpose of this project is to practice, nobody will
really use it.

The possible tokens are:
- characters: (type='char', data=<char>). For example, ('char', 'A'), ('char',
  '\n')
- Parenthesis tokens. The possible types are {'(', '(?:', '(?=', '(?!',
  ')'}. The (data) field is always None.
- Character classes: (type='char-class', data=<set>). Here <set> is a set of
  characters which appear in the class. So if the text is '[a-d1-5]', the token
  will be (type='char-class', data=set('abcd')|set('12345')). This kind of token
  encompasses escapes like (r'\d', r'\D', r'\s', r'\S', r'\w', r'\W'). For
  example, r'\d' will expand to the token (type='char-class',
  data=set('0123456789'))
- Zero width assertions. The possible types are {'^', '$', r'\A', r'\Z', r'\b',
  r'\B'}. For all, the (data) field is None.
- Backreferences: (type='bref', data=<int>) where <int> is an integer >= 1
- Some operators; the (type) is the operator, the (data) is None. The possible
  types are {'|'}
- Greedy quantifiers: (type='greedy-quant', data=(<low>, <high>)). <low> and <high>
  are integers, with <low> <= <high>. This token group encompasses regex
  operators like '*', '+', '?', '{<low>, <high>}'. The mapping from operators to token is:
  '*' -> ('greedy-quant', (0, None))
  '+' -> ('greedy-quant', (1, None))
  '?' -> ('greedy-quant', (0, 1))
  '{<low>,<high>}' -> ('greedy-quant', (int(<low>), int(<high>)))

How regex parts map to tokens:
- character class: (type='char-class', data=<set>)
- Dot: (type='char-class', data=<set>). Here <set> is either the set of all
  ascii characters, or all except '\n'. This depends on the DOTALL flag's
  presense.
- '\<digits>': (type='bref', data=int(<digits>))
- '\d', '\D', '\s', '\S', '\w', '\W': (type='char-class', data=<set>).
- '^', '$', '\A', '\b', '\B', '\Z', '|', '(', '(?:', '(?=', '(?!', ')': (type)
  matches operator strings, (data) is None. For example, '(?=' maps to
  (type='(?=', data=None).
- '+', '*', '?', '{m,n}': (type='greedy-quant', data=(<low>,<high>))
"""

tokenfuncs = []

# tns = tokenization name space
tns = NS(regstr=None, pos=None, flags=None)

def exhausted():
    return len(tns.regstr) == tns.pos

def takech(inc=False):
    if exhausted():
        return None
    ch = tns.regstr[tns.pos]
    if inc:
        tns.pos += 1
    return ch

def rest():
    return tns.regstr[tns.pos:]

def token_error(msg):
    # just a helper
    raise ValueError(f'{msg}: "{tns.regstr}"')
    
def tokenfunc(func):
    tokenfuncs.append(func)
    return func

# main function
def tokenize(regstr, flags):
    """Transforms (regstr) into a list of tokens and returns it. Raises
    ValueError if (regstr) is flawed."""
    tns.regstr, tns.pos, tns.flags = regstr, 0, flags
    tokens = []
    size = len(regstr)
    while not exhausted():
        for tf in tokenfuncs:
            token = tf()
            if token is not None:
                tokens.append(token)
                break
        else:
            raise ValueError(f'Cannot extract a token from "{rest()}"')
    return tokens

@tokenfunc
def simple():
    """For tokens whose type matches the characters in the regex string, and
    whose data is None."""
    # In the tuple below, make sure that if (t1) is a prefix of (t2), (t1) comes
    # after (t2). Consider an example which violates this and the consequence:
    # If '(' comes before '(?=', if '(?=' is a prefix of the regex string at
    # (tns.pos), the token that will be extracted is '(', not '(?=' as should be
    # the case.
    types = ('^', '$', r'\A', r'\b', r'\B', r'\Z', '|',
             '(?:', '(?=','(?!', '(', ')')
    for t in types:
        if tns.regstr.startswith(t, tns.pos):
            tns.pos += len(t)
            return Token(type=t, data=None)
    else:
        return None
        
@tokenfunc
def char_class():
    """Tries to extract a character class from the regex string. Returns a token
    of the form Token(type='char-class', data=<chars>), where <chars> is the set
    of characters in the class.

    Character class syntax: A template is what appears inside the brackets enclosing
    the class (e.g. 'a-z' in [a-z]). I will describe the fine points related to how
    a template maps to the character set that makes up the class. All of the rest is
    the same as in Python.

    There are a couple of points to consider:

    1. Caret in the beginning

    If the template begins with '^', the char set will be the negation of that
    determined by the rest of the template. As a special case, a template of '^'
    raises an error, because '' is not a valid template. For the rest of this
    docstring, assume the template does not begin with '^'.

    2. Hyphens

    Hyphens have two purposes:
    - They stand for themselves. This happens in the following cases:
      - When they appear at the beginning or end of the template. As a special
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
       characters special to char classes - r'\[]^-' - can be included by
       backlashing them.
    """
    
    if takech() != '[':
        return None
    # Find index of closing ']' and store it in (cbi). Start at tns.pos+2
    # because a beginning ']' does not indicate the end of the class.
    cbi = tns.regstr.find(']', tns.pos+2)
    while True:
        if cbi == -1:
            raise ValueError(f'No closing bracket: {rest()}')
        if tns.regstr[cbi-1] == '\\':
            cbi = tns.regstr.find(']', cbi+1)
        else:
            break
    template = tns.regstr[tns.pos+1:cbi]
    tns.pos = cbi + 1
    return Token('char-class', _form_class(template))

def _form_class(template):
    """Processes the insides of a character class (template) into a set of
    characters."""
    assert template # (template) should not be empty

    def error(msg):
        # just a helper
        raise ValueError(f'Bad character class template "{template}": {msg}')
    
    tempiter = iter(template)
        
    # stage 1: take care of initial '^'.
    negate = False
    first = next(tempiter)
    if first == '^':
        negate = True
    else: 
        tempiter = itertools.chain([first], tempiter) # put (first) back

    tokens = deque()
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

    return result if not negate else ALL-result

@tokenfunc
def greedy_quant():
    def error(extra=None):
        # just a helper
        msg = f'Invalid quantifier "{rest()}"'
        if extra is not None:
            msg = f'{msg}: {extra}'
        raise ValueError(msg)
    
    ch = takech()
    bounds = {'*': (0, None), '+': (1, None), '?': (0, 1)}.get(ch)
    if bounds is not None:
        tns.pos += 1
        return Token(type='greedy-quant', data=bounds)
    elif ch == '{':
        endi = tns.regstr.find('}', tns.pos)
        if endi == -1:
            error('Missing closing "{".')
        bounds = tns.regstr[tns.pos+1:endi].split(',')
        if len(bounds) != 2:
            error("One comma required. The format is '{m,n}'")
        try:
            low, high = map(int, bounds)
        except ValueError:
            error("Bounds must be parsable to integers.")
        if low > high:
            error(f'<low> must not exceed <high>.')
        tns.pos = endi+1
        return Token('greedy-quant', (low, high))
    else:
        return None

@tokenfunc
def char_class_shorts():
    """Handles character class shorthands, like r'\d'. The dot is also
    considered such as shorthand."""
    ch = takech(True)
    chars = None # the set of characters
    if ch == '.':
        chars = (ALL if DOTALL in tns.flags
                 else DOTNOALL)
    elif ch == '\\':
        nxt = takech(True)
        if nxt is None:
            token_error(f"Can't end in backlash.")
        chars = CLASS_SHORTS.get(nxt)
        if chars is None:
            tns.pos -= 2 # put (ch) and (nxt).
            return None
    else:
        tns.pos -= 1 # put back (ch)
        return None
    return Token('char-class', chars)
            
@tokenfunc
def backref():
    ch = takech()
    if ch != '\\':
        return None
    tns.pos += 1
    nxt = takech()
    if nxt is None:
        token_error(f"Can't end in backlash.")
    if nxt.isdigit():
        return Token('bref', int(extract_digits()))
    else:
        tns.pos -= 1 # put back (ch)
        return None

def extract_digits():
    """Helper function. Assumes the current regex string char is a
    digit. Extracts all digits from the current position, and returns
    them. Increments (tns.pos) in the process, so that it will point to the
    first non-digit char."""
    digits = []
    while True:
        digits.append(ch)
        tns.pos += 1
        if exhausted():
            break
        ch = takech()
        if not ch.isdigit():
            break
    return ''.join(digits)
    
@tokenfunc
def char():
    """Forms 'char' tokens. Call after other tokenization functions so that
    special characters are not interpreter as regular ones. For example, if the
    current char of the regex string is '*', this will interpret it as a
    character token rather than as a quantifier. As another example, this
    function will raise an error with r'\A' because 'A' is not a valid escape
    character, even though r'\A' is a valid regex. So if this is called before
    the function that processes r'\A' regexes, an error will ensue, which may
    not be desirable."""

    ch = takech(True)
    if ch == '\\':
        nxt = takech(True)
        if nxt is None:
            token_error('Cannot end in backlash.')
        elif nxt in SPECIAL:
            return Token('char', SPECIAL)
        elif nxt == 'x':
            x1, x2 = takech(True), takech(True)
            if (x1 is None or x1 not in HEX_DIGITS
                or x2 is None or x2 not in HEX_DIGITS):
                token_error('Expected two hex digits after "\\x"')
            return Token('char', chr(int(x1+x2, 16)))
        elif nxt in ESCAPE_CHARS:
            return Token('char', ESCAPE_CHARS[nxt])
        else:
            token_error(f'"{nxt}" is not a valid escape character, at {tns.pos}.')
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

    Contexts allow for the implementation of backreferencing. For example r'\3'
    could be implemented by simply referring to (context.groups[3]), where
    (context) belongs to the backreference pattern.

    A context is also used as an indirection layer to allow new groups and flags
    to be quickly updated for all subpatterns globally. Since all subpatterns
    alias the same context instance, creating new groups to be shared by all is
    simply done by changing (context.groups) in one subpattern.

    Attributes:
    - groups: a Groups instance
    - numgrps: the number of groups. If (groups is not None) then (len(groups)
      == numgrps). This is useful for creating (groups) when it is None.
    - flags: For example, IGNORECASE, MULTILINE, etc.
    """
    
    def __init__(self, numgrps=0, flags=None):
        self.groups = None
        self._numgrps = numgrps
        self.flags = RegexFlags(0) if flags is None else flags

    def initialize(self):
        """Create the list (result = [None] + odicts) and bind it to
        (self.groups). (odicts) is a list of OrderedDicts of length
        (self._numgrps). Parenthesis are numbered starting from 1, so this
        allows to reference the proper ordered dict using
        (result[parenthesis_index])."""        
        self.groups = Groups(self._numgrps)
        
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
        == 'match')) at with group index (i). If there is nothing stored at (i),
        None is returned."""
        assert hint in ('str', 'match')
        odict = self._lst[i]
        if not odict:
            return None
        itr = odict.values() if hint == 'str' else odict.keys()
        return next(reversed(itr))

    def add(self, match):
        """Assumes (match) is not exhausted. Always returns True. Returning True
        is useful because in many situations we add to groups and then return
        True. It is more convenient to just write (return m.groups.add(m))
        rather than (m.groups.add(m); return True)."""
        if match._groupi is not None:
            odict = self._lst[match._groupi]
            odict[match] = match._mstr
        return True
    
    def remove(self, match):
        """Assumes (self) is not exhausted. Always returns False. Returing False
        is useful because in many situations we remove from groups and then
        return False. It is more convenient to just write (return m._remove_from_groups())
        rather than (m._remove_from_groups(); return False)."""        
        if match._groupi is not None:
            odict = self._lst[match._groupi]
            odict.pop(match, None)
        return False

