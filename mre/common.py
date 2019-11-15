from collections import OrderedDict

# -------------------- FLAGS --------------------

"""
Flags are represented as singleton sets. The elements of the sets are objects
whose only purpose is to have a unique identity. A collection of flags is a set
of such objects. Observe that this design allows you to combine flags using
bitwise-or operator. (flag1 | flag2 | flag3) results in a collection of flags
which contains flag1, flag2 and flag3.
"""

# This data structure serves as a single point of control for the kinds of
# flags. If a new flag is to be added, simply bind a tuple of the flag's names
# to {object()}. The flag can then be referenced via those names from both this
# module and the global one, because both call define_flags(globals())
_flags = {('I', 'IGNORECASE'): {object()},
          ('M', 'MULTILINE'): {object()},
          ('S', 'DOTALL'): {object()}}

def define_flags(ns):
    for t, v in _flags.items():
        for k in t:
            ns[k] = v

define_flags(globals())
            
def emptyflags():
    return set()

def contains_flag(flags, flag):
    """Checks if (flag) belongs to the collection of flags (flags)."""
    return flag <= flags

# -------------------- CONTEXT --------------------

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
    - groups: None or a list of OrderedDicts which map match objects to the
      strings they match. The reasons for this data structure choice is outlined
      next:
      
      Lets consider an example: we have a pattern (P) with group index (i). Lets
      assume it's parent is the star quantifier '*'. The parent finds 3 matches
      of (P), (m1, m2, m3). At this point parent.group(i) will return the string
      matched by (m3). But assume the parent is then asked to backtrack, and
      it's state changes to (m1, m2). Then we want parent.group(i) to result in
      (m2)'s string. So at first we have context.groups[i] = {m1: m1_string, m2:
      m2_string, m3: m3_string}, and since pattern.group(i) simply returns the
      last string of pattern.context.groups[i], it will return m3_string. As
      part of backtracking, the parent removes m3 from the odict, so that
      context.groups becomes {m1: m1_string, m2: m2_string} so that
      parent.group(i) returns m2_string.
    
      In general, a given group index can be occupied by more than one match
      objects. A match can be alive for a while, but discarded for backtracking
      needs. When discarding, we must remove the proper string, the one
      belonging to that match, which is why a dict is used. But when getting the
      matched string at some group index, we want the latest one, which is why
      order is needed. The combination of order and mapping requirements yields
      the OrderedDict choice.

    - numgrps: the number of groups. If (groups is not None) then (len(groups)
      == numgrps). This is useful for creating (groups) when it is None.
    - flags: For example, IGNORECASE, MULTILINE, etc.
    """
    
    def __init__(self, numgrps=0, flags=None):
        self.groups = None
        self._numgrps = numgrps
        self.flags = emptyflags() if flags is None else flags

    def initialize(self):
        """Create the list (result = [None] + odicts) and bind it to
        (self.groups). (odicts) is a list of OrderedDicts of length
        (self._numgrps). Parenthesis are numbered starting from 1, so this
        allows to reference the proper ordered dict using
        (result[parenthesis_index])."""
        
        self.groups = [None]
        self.groups.extend(OrderedDict() for k in range(self._numgrps))
        
# ----------------------------------------
# utilities

def latest(groups, i, hint='str'):
    """For (groups) at (i), returns the lates string or match, when (hint ==
    'str') or (hint == 'match') respectively. If there is nothing stored at (i),
    None is returned."""
    assert hint in ('str', 'match')
    odict = groups[i]
    if not odict:
        return None
    itr = odict.values() if hint == 'str' else odict.keys()
    return next(reversed(itr))

# ----------------------------------------
# SeekableList

class TokenList:
    def __init__(self, base, start=0, end=None):
        self.base = base
        self.start = start
        self.end = len(base) - 1 if end is None else end
        self.current = start

    def __next__(self):
        if self.current > self.end:
            return
        yield self.base[self.current]
        self.current += 1

    def __getitem__(self, index):
        i = self.start + index
        if i > end:
            raise IndexError
        return self.base[i]

    @property
    def subtokens(self):
        """A paren token was just encountered. This returns a TokenList
        containing all tokens within opening paren that was just passed and it's
        corresponding closing."""
        start = self.current
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
        return TokenList(self.base, start, self.end-1)
    
    def __iter__(self):
        return self
