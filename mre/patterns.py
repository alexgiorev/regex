import sys

from collections import OrderedDict

import common

class _Error:
    pass

class Match:
    """Base class for match objects.
    The attributes shared by all Match instances are:
    - string: the string on which the match was made. match._mstr is a substring
      of this.
    - _mstr: the matched substring.
    - _groupi: the group index.
    - _groups
    - _start, _end: these determine the span of the matched substring.
    - _is_exhausted
    """

    @classmethod
    def _first(cls, astr, i, pattern):
        """Assumes that (0 <= i <= len(astr)). Returns a match corresponding to
        the first substring of (astr) starting at (i) that matches (pattern), or
        None if such a string doesn't exist."""
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
        return False and at that point (self) will be exhausted. Calling
        (self._next()) when (self) is exhausted raises a ValueError.

        Consider the following example: let P = r'(abcd)*', S = 'xxxabcyyy', i =
        3, m = P._match(S, i). Then (m._mstr == 'abc'), (m._start == 3), (m._end
        == 6) and (m.group(1) == 'c'). Now assume (m._next()) is executed. It
        will return True, (m._mstr == 'ab'), (m._start == 3), (m._end == 5) and
        (m.group(1) == 'b'). Again we execute (m._next()), again it returns True
        and this time (m._mstr == 'a'), (m._start == 3), (m._end == 4) and
        (m.group(1) == 'a'). One more time! We execute (m._next()), it returns
        True, (m._mstr == ''), (m._start == 3), (m._end == 3) and (m.group(1) ==
        None). Finally, if at this point (m._next()) is executed, it returns
        False, and all of m._mstr, m._start, m._end and m.group(1) are None. If
        now (m._next()) is executed, it will raise a ValueError."""
        raise NotImplementedError

    def _add_to_groups(self):
        """Assumes (self) is not exhausted. Always returns True. Returning True
        is useful because in many situations we add to groups and then return
        True. It is more convenient to just write (return m._add_to_groups())
        rather than (m._add_to_groups(); return True)."""
        
        if self._groupi is not None:
            od = self._groups[self._groupi]
            od[self] = self._mstr
        return True

    def _remove_from_groups(self):
        """Assumes (self) is not exhausted. Always returns False. Returing False
        is useful because in many situations we remove from groups and then
        return False. It is more convenient to just write (return m.remove_from_groups())
        rather than (m.remove_from_groups(); return False)."""
        
        if self._groupi is not None:
            od = self._groups[self._groupi]
            od.pop(self, None)
        return False

    def _check_exhausted(self):
        if self._is_exhausted:
            raise ValueError('Exhausted match object.')        

    # ----------------------------------------
    # Public functions and their helpers
        
    def __bool__(self):
        return True

    def __getitem__(self, i):
        return self.group(i)

    @property
    def _numgrps(self):
        """Just a helper function which returns the number of groups. Group
        numbers start from 1. (self._groups) is a list whose first element is
        None and all other elements correspond to actual groups. This is why the
        length is decremented."""
        return len(self._groups) - 1

    def _check_index(self, i):
        """A helper for the functions which make use of groups."""
        if type(i) is not int:
            raise TypeError(f'Index must be an int, not {type(i)}: {i}')
        if not 0 <= i <= self._numgrps:
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
            odict = self._groups[i]
            if not odict:
                return None
            return next(reversed(odict.values()))
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
        mstrs = (self.group(i) for i in range(1, self._numgrps + 1))
        return tuple(default if mstr is None else mstr for mstr in mstrs)

    def span(self, i=0):
        return (self.start(i), self.end(i))

    def _boundary(self, i, hint):
        """Helper for self.start and self.end. Finds the match (m) which has
        group number (i). If there is no such match, -1 is returned. If (hint ==
        'start'), the start index of (m) is returned, while if (hint == 'end'),
        the end index is returned."""
        
        assert not self._is_exhausted
        self._check_index(i)
        if hint not in ('start', 'end'):
            raise ValueError(f'Bad hint: {repr(hint)}')
        if i == 0:
            m = self
        else:
            odict = self._groups[i]
            if not odict:
                return -1
            m = next(reversed(odict))
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
    """Base class for patterns."""
    
    def _match(self, astr, i):
        """Attempts to make a match at (i). If not possible, None is
        returned, otherwise the match object."""
        assert 0 <= i <= len(astr)
        return self._Match._first(astr, i, self)

    def _newmatch(self, astr, i):
        self._context.newgroups()
        return self._match(astr, i)
    
    # ----------------------------------------
    # Public functions
    
    def match(self, astr):
        """If (self) matches at the beginning of (astr), the corresponding match
        object is returned. Otherwise, None."""
        return self._newmatch(astr, 0)

    def search(self, astr):
        """If (self) doesn't match anywhere in (astr), returns None. Otherwise,
        returns the leftmost match."""
        for i in range(len(astr) + 1):
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
        then (self.allstrings('--abcd--', 2)) will return ['a', 'ab', 'abc']."""
        m = self._newmatch(astr, i)
        if not m:
            return []
        out = [m.group()]
        while m._next():
            out.append(m.group())
        return out


class Char(Pattern):
    class _Match(Match):
        @classmethod
        def _first(cls, astr, i, pattern):
            m = cls()
            ignorecase = common.contains_flag(pattern._context.flags, common.I)
            
            if i == len(astr):
                return None
            else:
                astr_char = astr[i]
                matches = (pattern._char.lower() == astr_char.lower()
                           if ignorecase else pattern._char == astr_char)
                if matches:
                    m.string = astr
                    m._start = i
                    m._end = i + 1
                    m._groupi = pattern._groupi
                    m._groups = pattern._context.groups
                    m._mstr = astr_char
                    m._is_exhausted = False
                    m._add_to_groups()
                    return m
                else:
                    return None

        def _next(self):
            self._check_exhausted()
            self._is_exhausted = True
            self._start = self._end = self._mstr = None
            return self._remove_from_groups()

    def __init__(self, char, groupi, context):
        self._char = char
        self._groupi = groupi
        self._context = context


class CharClass(Pattern):
    class _Match(Match):
        @classmethod
        def _first(cls, astr, i, pattern):            
            ignorecase = common.contains_flag(pattern._context.flags, common.I)            
            if start == len(astr):
                return None
            else:
                astr_char = astr[i]
                matches = ((astr_char.lower() if ignorecase else char)
                           in self._chars)
                if matches:
                    m = cls()
                    m.string = astr
                    m._start = i
                    m._end = i + 1
                    m._groups = pattern._context.groups
                    m._groupi = pattern._groupi
                    m._mstr = astr_char
                    m._is_exhausted = False
                    m._add_to_groups()
                    return m
                else:
                    return None

        def _next(self):
            self._check_exhausted()
            self._is_exhausted = True
            m._mstr = m._start = m._end = None
            return self._remove_from_groups()
                
    def __init__(self, chars, groupi, context):
        self._chars = (chars
                      if not common.contains_flag(context.flags, common.I)
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
    used to determine if the condition at (i) holds. Various class methods serve
    to create the predicate function and return the zero-width pattern
    corresponding to it."""
    
    class _Match(Match):
        @classmethod
        def _first(cls, string, i, pattern):
            if pattern._pred(self.string, i):
                m = cls()
                m.string = string
                m._mstr = ''
                m._groupi = pattern._groupi
                m._groups = pattern._context.groups
                m._start = m._end = i
                m._is_exhausted = False
                m._add_to_groups()
                return m
            else:
                return None

        def _next(self):
            self._check_exhausted()
            self._is_exhausted = True
            self._mstr = self._start = self._end = None
            return self._remove_from_groups()
    
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
    def from_str(cls, letter, groupi, context):
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

        if common.contains_flag(context.flags, re.M):
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
    def _Match(Match):
        @classmethod
        def _first(cls, astr, i, pattern):
            ref, groups = pattern._ref, pattern._context.groups
            ignorecase = common.contains_flag(pattern._context.flags, common.I)
            
            odict = groups[ref]
            if not odict:
                return None
            mstr = next(reversed(odict.values())) # the latest matched string
            end = i + len(mstr)
            substr = astr[i: end]
            matches = (mstr.lower() == substr.lower() if ignorecase
                       else mstr == substr)
            if matches:
                m = cls()
                m.string, m._mstr = astr, substr
                m._groupi, m._groups = pattern._groupi, groups      
                m._start, m._end = i, end
                m._is_exhausted = False
                m._add_to_groups()
                return m
            else:
                return None

        def _next(self):
            self._check_exhausted()
            self._is_exhausted = True
            self._mstr = self._start = self._end = None
            return self._remove_from_groups()

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
        @classmethod
        def _first(cls, astr, i, pattern):
            def init(left):
                # string, _mstr, _groupi, _groups, _start, _end, _is_exhausted
                m.string = astr
                m.groupi, m.groups = pattern._groupi, pattern._context.groups
                m._is_exhausted = False
                m._left = m._right = pattern._left, pattern._right
                m._childleft = left
                m._child = child
                m._refresh()
                return m
                
            m = cls()
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
            return self._add_to_groups() # True
                    
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
                    return self._remove_from_groups()

    def __init__(self, left, right, groupi, context):
        self._left = left
        self._right = right
        self._groupi = groupi
        self._context = context

        
class GreedyQuant(Pattern):
    """All quantifiers are implemented by this class. This includes '*', '+',
    '?', '{m,n}'. A quantifier has a lower and upper bound. Here is a mapping from operators to bounds:
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
        @classmethod
        def _first(cls, astr, i, pattern):
            # string, _mstr, _groupi, _groups, _start, _end, _is_exhausted

            # partially initialize first, so that some methods can be used.
            m = cls()
            m.string = astr
            m._mstr = None
            m._groupi, m._groups = pattern._groupi, pattern._context._groups
            m._start = m._end = i
            m._is_exhausted = False
            m._low, m._high = pattern._low, pattern._high
            m._children = []
            m._base = pattern._child
            
            m._take()
            while len(m._children) < m._high:
                if not m._next_high():
                    return None
            m._add_to_groups()
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
                    return self._add_to_groups()
            self._is_exhausted = True
            self._start = None
            return self._remove_from_groups()

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
    """For regexes of the form '<left-regex><right-regex>'.
    Extra attributes:
    - _leftpattern, _rightpattern: correspond to <left-regex> and <right-regex> above.
    - _leftchild, _rightchild: match objects derived from _leftpattern and
      _rightpattern, respectively."""
    
    class _Match(Match):
        @classmethod
        def _first(cls, astr, i, pattern):
            leftmatch = pattern._left._match(astr, i)
            if not leftmatch:
                return None
            while True:
                rightmatch = pattern._right._match(astr, leftmatch._end)
                if rightmatch:
                    # initialize and return
                    m = cls()
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
                    m._add_to_groups()
                    return m
                if not lm._next():
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
                return self._add_to_groups()
            else:
                while self._leftchild._next():
                    rightmatch = self._rightpattern._match(self.string, leftmatch._end)
                    if rightmatch:
                        self._rightchild = rightmatch
                        return self._add_to_groups()
                else:
                    self._is_exhausted = True
                    self._start = None
                    return self._remove_from_groups()

    def __init__(self, left, right, groupi, context):
        self._left = left
        self._right = right
        self._groupi = groupi
        self._context = context


if __name__ == '__main__':
    raise NotImplementedError