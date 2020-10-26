from . import common

# get a copy of the flags so that they can be accessed publically.
common.define_flags(globals())

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
