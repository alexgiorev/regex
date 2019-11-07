from collections import OrderedDict

# -------------------- FLAGS --------------------

"""
Flags are represented as singleton sets. The elements of the sets are objects
whose only purpose is to have a unique identity. A collection of flags is a set
of such objects. Observe that this design allows you to combine flags using
bitwise-or operator. (flag1 | flag2 | flag3) results in a collection of flags
which contains flag1, flag2 and flag3.
"""

I = IGNORECASE = {object()}
M = MULTILINE = {object()}
S = DOTALL = {object()}

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
    - groups: None or a list of OrderedDicts. When a list, each odict maps Match
      objects to strings. For a match (m), (m.group(k)) corresponds to the last
      string in (con.groups[k]), where con is the context of (m).
    - numgrps: the number of groups. If (groups is not None) then (len(groups)
      == numgrps). This is useful for creating (groups) when it is None.
    - flags
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
        
