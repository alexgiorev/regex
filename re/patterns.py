import sys

from collections import OrderedDict

import .common


class Match:
    """Base class for match objects. A match object is created from a Pattern
    P. Given a string S, a match corresponds to a substring of S that starts at
    a given position (determined when creating the match) which matches P."""

    @classmethod
    def _first(cls, astr, i, pattern):
        """Assumes that (0 <= i <= len(astr)). Returns a match corresponding to
        the first substring of (astr) starting at (i) that matches (pattern), or
        None if such a string doesn't exist."""
        raise NotImplementedError
    
    def __bool__(self):
        return True

    @property
    def _mstr(self):
        """Returns the matched string if (self) is not exhausted, None if it
        is."""
        raise NotImplementedError

    @property
    def _end(self):
        """Returns the end index in (self.string) where (self._mstr) ends."""
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
        (self._next()) when (self) is exhausted raises a ValueError."""
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
            raise ValueError('Exhausted match.')


class Pattern:
    """Base class for patterns."""
    
    def match(self, astr):
        self.context.newgroups()
        return self._match(astr, 0)

    def _match(self, astr, i):
        """Attempts to make a match at (i). If not possible, None is
        returned."""
        assert 0 <= i <= len(astr)
        return self._Match._first(astr, i, self)


class Char(Pattern):
    class _Match(Match):
        @classmethod
        def _first(cls, astr, i, pattern):
            m = cls()
            ignorecase = common.contains_flag(pattern._context.flags, common.I)
            
            if start == len(astr):
                return None
            else:
                astr_char = astr[i]
                matches = (pattern._char.lower() == astr_char.lower()
                           if ignorecase else char == astr_char)
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
            
        def _next(self):
            self._check_exhausted()
            self._is_exhausted = True
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
            return self._remove_from_groups()
                
    def __init__(self, chars, groupi, context):
        self._chars = (chars
                      if not common.contains_flag(context.flags, common.I)
                      else {char.lower() for char in chars})
        self._groupi = groupi
        self._context = context


class ZeroWidth(Pattern):
    class _Match(Match):
        def __init__(self, string, start, pattern):
            self.string = string
            self._start = start
            self._groups = pattern._context.groups
            self._groupi = pattern._groupi
            self._pred = pattern._pred
            self._mstr = self._end = None
            self._is_exhausted = False

        def _next(self):
            self._check_exhausted()
            if self._mstr is None: # initial match
                if self._pred(self.string, self._start):
                    self._mstr = ''
                    self._end = self._start
                    return self._add_to_groups()
                else:
                    self._is_exhausted = True
                    return False
            else:
                self._is_exhausted = True
                return self._remove_from_groups()
    
    @classmethod
    def poslook(cls, pattern, negate, groupi, context):
        pred = (lambda string, i: bool(pattern._match(string, i))
                if not negate
                else lambda string, i: not bool(pattern._match(string, i)))
        
        return cls(pred, groupi, context)

    @classmethod
    def from_str(cls, letter, groupi, context):
        """Returns the zero-width assertions corresponding to {'^', '$', r'\b',
        r'\B', r'\A', r'\Z'}."""
        
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


class GroupRef(Pattern):
    def _Match(Match):
        def __init__(self, string, start, pattern):
            self.string = string
            self._start = start
            self._i = pattern._i
            self._I = common.contains_flag(pattern._context.flags, common.I)
            self._groups = pattern._groups
            self._groupi = pattern._groupi
            self._mstr = self._end = None
            self._is_exhausted = False


        def _next(self):
            self._check_exhausted()
            if self._mstr is None: # initial match
                od = self._groups[self._i]
                if not od:
                    self._is_exhausted = True
                    return False
                group = next(reversed(od.items()))[1]
                substr = self.string[self._start: self._start + len(group)]
                matches = (group.lower() == substr.lower() if self._I
                           else group == substr)
                if matches:
                    self._mstr = substr
                    self._end = self._start + len(substr)
                    return self._add_to_groups()
                else:
                    self._is_exhausted = True
                    return False
            else:
                self._is_exhausted = True
                return self._remove_from_groups()

    def __init__(self, string, groupi, context):
        self._i = i
        self._groupi = groupi
        self._context = context


class Alternative(Pattern):
    class _Match(Match):
        """Exhausted when (self._m is None and not self._atleft)."""
        def __init__(self, string, start, pattern):
            self.string = string
            self._start = start
            self._lp = pattern._left
            self._rp = pattern._right
            self._groups = pattern._context.groups
            self._groupi = pattern._groupi
            self._m = None
            self._atleft = True

        @property
        def _mstr(self):
            self._check_exhausted()
            return self._m._mstr

        @property
        def _end(self):
            self._check_exhausted()
            return self._m._end
            
        @property
        def _is_exhausted(self):
            return not self._m and not self._atleft
            
        def _next(self):
            self._check_exhausted()
            
            if not self._m and self._atleft: # initial match
                m = self._lp._match(self.string, self._start)
                if m:
                    self._m = m
                else:
                    self._atleft = False
                    m = self._rp._match(self.string, self._start)
                    if m:
                        self._m = m
                    else:
                        return False
                return self._add_to_groups()
            
            if self._m._next():
                return self._add_to_groups()
            else:
                if self._atleft:
                    self._atleft = False
                    m = self._rp._match(self.string, self._start)                
                    if m:
                        self._m = m
                        return self._add_to_groups()
                self._m = None
                return self._remove_from_groups()

    def __init__(self, left, right, groupi, context):
        self._left = left
        self._right = right
        self._groupi = groupi
        self._context = context

class GreedyQuant(Pattern):
    class _Match(Match):
        """
        - When (self.children is None), the match is exhausted. Should I raise an error then?
        - Match objects should be hashable on id so that using OrderedDict works.
        """
        
        def __init__(self, string, start, pattern):
            self.string = string
            self._start = start
            self._pat = pattern._child
            self._low = pattern._low
            self._high = pattern._high
            self._groupi = pattern._groupi
            self._groups = pattern._context.groups
            self._children = None
            self._is_exhausted = False

        @property
        def _mstr(self):
            """Returns the matched string."""
            return ''.join(child._mstr for child in self._children)

        @property
        def _end(self):
            if not self._children:
                return self._start
            else:
                return self._children[-1]._end
            
        def _next(self):
            self._check_exhausted()
            while self._next_upper():
                if len(self._children) >= self._low:
                    return self._add_to_groups()
            return False

        def _next_upper(self):
            """Assumes (self) isn't exhausted."""
            if self.children is None: # initial match
                m._take()
                return True
            if not self._children:
                self._children = None
                self._is_exhausted = True
                return self._remove_from_groups()
            child = self._children[-1]
            if child._next():
                self._take()
            else:
                self._children.pop()
            return True

        def _take(self):
            """Assumes (self._children is not None)."""
            i = self._end
            for k in range(len(self._children), self._high):
                child = self._pat._match(self.string, i)
                if child is None:
                    break
                self._children.append(child)
                end = child._end
                if i == end:
                    break
                i = end
                
    def __init__(self, child, low, high, groupi, context):
        self._child = child
        self._low = low
        self._high = high if high is not None else sys.maxsize
        self._groupi = groupi
        self._context = context


class Concat(Pattern):
    class _Match(Match):
        def __init__(self, string, start, pattern):
            self.string = string
            self._start = start
            self._groups = pattern._context.groups
            self._groupi = pattern._groupi
            self._lp = pattern._left
            self._rp = pattern._right
            self._lm = self._rm = None
            self._is_exhausted = False

        @property
        def _mstr(self):
            return self._lm._mstr + self._rm._mstr

        @property
        def _end(self):
            return self._rm._end
            
        def _next(self):
            self._check_exhausted()
            if self._lm is None: # initial match
                self._lm = lm = self._lp._match(self.string, self._start)
                if lm is None:
                    self._is_exhausted = True
                    return False
                while True:
                    self._rm = rm = self._rp._match(self.string, lm._end)
                    if rm:
                        return self._add_to_groups()
                    if not lm._next():
                        self._is_exhausted = True
                        return False
            else: # not the initial match
                if self._rm._next():
                    return self._add_to_groups()
                else:
                    while self._lm._next():
                        rm = self._rp._match(self.string, lm._end)
                        if rm:
                            self._rm = rm
                            return self._add_to_groups()
                    else:
                        self._is_exhausted = True
                        return self._remove_from_groups()

    def __init__(self, left, right, groupi, context):
        self._left = left
        self._right = right
        self._groupi = groupi
        self._context = context

if __name__ == '__main__':
    raise NotImplementedError
