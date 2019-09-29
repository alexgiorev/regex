import sys

from collections import OrderedDict

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
    """Base class for pattern objects."""
    def match(self, astr):
        self.context.groups = [OrderedDict()] * self.context.ngroups()
        return self._match(astr, 0)

    def _match(self, astr, i):
        """Assumes (i) is an index within (astr). Attempts to make a match at
        (i). If not possible, None is returned."""        
        m = self._Match(astr, i, self)
        return m if m._next() else None


class Char(Pattern):
    raise NotImplementedError    


class CharClass(Pattern):
    class _Match(Match):
        raise NotImplementedError

    
class ZeroWidth(Pattern):
    @classmethod
    def poslook(cls, pattern, negate=False):
        raise NotImplementedError
    
    raise NotImplementedError


class GroupRef(Pattern):
    raise NotImplementedError


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

    def _match(self, astr, i):
        m = self._Match(astr, i, self)
        return m if m._next() else None

class GreedyQuant(Pattern):
    class _Match(Match):
        """
        - When (self.children is None), the match is exhausted. Should I raise an error then?
        - Match objects should be hashable on id so that using OrderedDict works.
        - Duplicate code in (make) and (_next_high)
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
            """Assumes (self._children is not None). After _take, (self) will
            satisfy the upper boundary condition."""
            i = self._children[-1]._end if self._children else self._starti
            for k in range(len(self._children), self._high):
                child = self.pat._match(self.string, i)
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
    raise NotImplementedError
