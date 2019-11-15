from . import common

# get a copy of the flags so that they can be accessed publically.
common.define_flags(globals())

def compile(regstr, flags):
    """Returns a pattern corresponding to (regstr)."""
    raise NotImplementedError
