import sys

from collections import OrderedDict

import . as _re

class Match:
    """Base class for match objects."""
    def __bool__(self):
        return True

    @property
    def _string(self):
        """Returns the matched string."""
        raise NotImplementedError

    @property
    def _end(self):
        """Returns the end index of the matched string within the string."""
        raise NotImplementedError

    def _next(self):
        """Changes the state of (self) to reflect the next match in the
        set. Returns True if there had been a next match. Returns False if
        (self) if there hadn't been a next match and (self) is not exhausted. If
        (self) is exhausted at the time of the call, an error is raised."""
        raise NotImplementedError

    def _add_to_groups(self):
        """Assumes (self) is not exhausted. Always returns True (this is useful
        because in many situations we add to groups and then return True. It is
        more convenient to just write 'return m._add_to_groups())."""
        if self.groupi is not None:
            od = self._groups[self.groupi]
            od[self] = self._string
        return True

    def _remove_from_groups(self):
        if self._groupi is not None:
            od = self._groups[self._groupi]
            del od[self]
        return False

    def _check_exhausted(self):
        if self._is_exhausted:
            raise ValueError('Exhausted match.')


class Pattern:
    """Base class for patterns."""
    def match(self, astr):
        self.context.groups = [OrderedDict()] * self.context.ngroups()
        return self._match(astr, 0)

    def _match(self, astr, i):
        """Assumes (0 <= i <= len(astr)). Attempts to make a match at (i). If
        not possible, None is returned."""
        m = self._Match(astr, i, self)
        return m if m._next() else None


class Char(Pattern):
    class _Match(Match):
        def __init__(self, string, i, pattern):
            self.string = string
            self._starti = i
            self._char = pattern.char
            self._groupi = pattern.groupi
            self._groups = pattern.context.groups
            self._I = _re._contains_flag(pattern.context.flags, _re.I)
            self._string = None
            self._is_exhausted = False
            
        def _next(self):
            self._check_exhausted()
            if self._string is None: # initial match
                if self._starti == len(self.string):
                    self._is_exhausted = True
                    return False
                else:
                    char = self.string[self._starti]
                    matches = (char.lower() == self._char.lower()
                               if self._I else char == self._char)
                    if matches:
                        self._string = char
                        self._end = self._starti + 1
                        return self._add_to_groups()
            else:
                self._is_exhausted = True
                return self._remove_from_groups()

    def __init__(self, char, groupi, context):
        self.char = char
        self.groupi = groupi
        self.context = context


class CharClass(Pattern):
    class _Match(Match):
        def __init__(self, string, i, pattern):
            self.string = string
            self._starti = i
            self._chars = pattern.chars
            self._groups = pattern.context.groups
            self._groupi = pattern.groupi
            self._string = self._end = None
            self._is_exhausted = False
            self._I = _re.contains_flag(pattern.context.flags, _re.I)

        def _next(self):
            self._check_exhausted()
            if self._string is None: # initial match
                if self._starti == len(self.string):
                    self._is_exhausted = True
                    return False
                else:
                    char = self.string[self._starti]
                    matches = ((char.lower() if self._I else char)
                               in self._chars)
                    if matches:
                        self._string = char
                        self._end = self._starti + 1
                        return self._add_to_groups()
            else:
                self._is_exhausted = True
                return self._remove_from_groups()
                
    def __init__(self, chars, groupi, context):
        self.chars = (chars
                      if not _re.contains_flag(context.flags, _re.I)
                      else {char.lower() for char in chars})
        self.groupi = groupi
        self.context = context


class ZeroWidth(Pattern):
    class _Match(Match):
        def __init__(self, string, i, pattern):
            self.string = string
            self._starti = i
            self._groups = pattern.context.groups
            self._groupi = pattern.groupi
            self._pred = pattern.pred
            self._string = self._end = None
            self._is_exhausted = False

        def _next(self):
            self._check_exhausted()
            if self._string is None: # initial match
                if self._pred(self.string, self._starti):
                    self._string = ''
                    self._end = self._starti
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

        if _re._contains_flag(context.flags, re.M):
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
        self.pred = pred
        self.groupi = groupi
        self.context = context


class GroupRef(Pattern):
    def _Match(Match):
        def __init__(self, string, starti, pattern):
            self.string = string
            self._starti = starti
            self._i = pattern.i
            self._I = _re._contains_flag(pattern.context.flags, _re.I)
            self._groups = pattern.groups
            self._groupi = pattern.groupi
            self._string = self._end = None
            self._is_exhausted = False

        def _next(self):
            self._check_exhausted()
            if self._string is None: # initial match
                od = self._groups[self._i]
                if not od:
                    self._is_exhausted = True
                    return False
                group = next(reversed(od.items()))[1]
                substr = self.string[self._starti: self._starti + len(group)]
                matches = (group.lower() == substr.lower() if self._I
                           else group == substr)
                if matches:
                    self._string = substr
                    self._end = self._starti + len(substr)
                    return self._add_to_groups()
                else:
                    self._is_exhausted = True
                    return False
            else:
                self._is_exhausted = True
                return self._remove_from_groups()

    def __init__(self, i, groupi, context):
        self.i = i
        self.groupi = groupi
        self.context = context


class Alternative(Pattern):
    class _Match(Match):
        """Exhausted when (self._m is None and not self._atleft)."""
        def __init__(self, astr, i, pattern):
            self.string = astr
            self._starti = i
            self._lp = pattern.left
            self._rp = pattern.right
            self._groups = pattern.context.groups
            self._groupi = pattern.groupi
            self._m = None
            self._atleft = True

        @property
        def _string(self):
            self._check_exhausted()
            return self._m._string

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
                m = self._lp._match(self.string, self._starti)
                if m:
                    self._m = m
                else:
                    self._atleft = False
                    m = self._rp._match(self.string, self._starti)
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
                    m = self._rp._match(self.string, self._starti)                
                    if m:
                        self._m = m
                        return self._add_to_groups()
                self._m = None
                return self._remove_from_groups()

    def __init__(self, left, right, groupi, context):
        self.left = left
        self.right = right
        self.groupi = groupi
        self.context = context

class GreedyQuant(Pattern):
    class _Match(Match):
        """
        - When (self.children is None), the match is exhausted. Should I raise an error then?
        - Match objects should be hashable on id so that using OrderedDict works.
        """
        
        def __init__(self, string, starti, pattern):
            self.string = string
            self._starti = starti
            self._pat = pattern.child
            self._low = pattern.low
            self._high = pattern.high
            self._groupi = pattern.groupi
            self._groups = pattern.context.groups
            self._children = None
            self._is_exhausted = False

        def _check_exhausted(self):
            if self._children is None:
                raise ValueError(f'Exhausted match.')

        @property
        def _string(self):
            """Returns the matched string."""
            return ''.join(child._string for child in self._children)

        @property
        def _end(self):
            if not self._children:
                return self._starti
            else:
                return self._children[-1]._end
        
        def _next(self):
            self._check_exhausted()
            while self._next_high():
                if len(self._children) >= self._low:
                    return self._add_to_groups()
            return False

        def _next_high(self):
            """Assumes (self) isn't exhausted. Changes (self) to satisfy only
            the upper bound."""
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
        self.child = child
        self.low = low
        self.high = high if high is not None else sys.maxsize
        self.groupi = groupi
        self.context = context


class Concat(Pattern):
    class _Match(Match):
        def __init__(self, string, starti, pattern):
            self.string = string
            self._starti = starti
            self._groups = pattern.context.groups
            self._groupi = pattern.groupi
            self._lp = pattern.left
            self._rp = pattern.right
            self._lm = self._rm = None
            self._is_exhausted = False

        @property
        def _string(self):
            return self._lm._string + self._rm._string

        @property
        def _end(self):
            return self._rm._end
            
        def _next(self):
            self._check_exhausted()
            if self._lm is None: # initial match
                self._lm = lm = self._lp._match(self.string, self._starti)
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
        self.left = left
        self.right = right
        self.groupi = groupi
        self.context = context
