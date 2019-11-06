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

    def newgroups(self):
        """Create the list (result = [None] + odicts) and bind it to
        (self.groups). (odicts) is a list of OrderedDicts of length
        (self._numgrps). Parenthesis are numbered starting from 1, so this
        allows to reference the proper ordered dict using
        (result[parenthesis_index])."""
        
        self.groups = [None]
        self.groups.extend(OrderedDict() for k in range(self._numgrps))
        
