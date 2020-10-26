from collections import OrderedDict

########################################
# Flags

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
        self.flags = emptyflags() if flags is None else flags

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
        return self.N

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
