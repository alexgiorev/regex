import itertools
import string
from collections import namedtuple
from types import SimpleNamespace

from collections import deque

ALL = frozenset(chr(i) for i in range(128)) # whole character set.
DOT_NO_ALL = ALL-{'\n'} # dot characters without DOTALL flag.
DIGIT = frozenset(string.digits)
ALPHANUMERIC = frozenset(string.ascii_letters + string.digits + '_')
WHITESPACE = frozenset(string.whitespace)
# shorthands for character classes
CLASS_SHORTS = {'d': DIGIT,
                'D': ALL - DIGIT,
                's': WHITESPACE,
                'S': ALL - WHITESPACE,
                'w': ALPHANUMERIC,
                'W': ALL - ALPHANUMERIC}
SPECIAL = r'\[].^()|+*?{}'
HEX_DIGITS = f'0123456789abcdefABCDEF'
ESCAPE_CHARS = {'a': '\a', 'f': '\f', 'n': '\n', 'r': '\r',
               't': '\t', 'v': '\v', 'b': '\b'}

# ----------------------------------------
# Tokenization

Token = namedtuple('Token', 'type data')

"""
For some tokens, due to a lack of imagination, the type and lexeme match. For
example, for the token '(?:', the type is also written as '(?:' so that the
whole token is ('(?:', '(?:')

The possible tokens are:
- characters: (type='char', data=<char>). For example, ('char', 'A'), ('char',
  '\n')
- Parenthesis tokens. The possible types are {'(', '(?:', '(?=', '(?!',
  ')'}. The (data) field is always None.
- Character classes: (type='char-class', data=<set>). Here <set> is a set of
  characters which appear in the class. So if the text is '[a-d1-5]', the token
  will be (type='char-class', data=set('abcd')|set('12345')). This kind of token
  encompasses escapes like (r'\d', r'\D', r'\s', r'\S', r'\w', r'\W'). For
  example, r'\d' will expand to the token (type='char-class',
  data=set('0123456789'))
- Zero width assertions. The possible types are {'^', '$', r'\A', r'\Z', r'\b',
  r'\B'}. For all, the (data) field is None.
- Backreferences: (type='bref', data=<int>) where <int> is an integer >= 1
- Some operators; the (type) is the operator, the (data) is None. The possible
  types are {'|'}
- Greedy quantifiers: (type='greedy-quant', data=(<low>, <high>)). <low> and <high>
  are integers, with <low> <= <high>. This token group encompasses regex
  operators like '*', '+', '?', '{<low>, <high>}'. The mapping from operators to token is:
  '*' -> ('greedy-quant', (0, None))
  '+' -> ('greedy-quant', (1, None))
  '?' -> ('greedy-quant', (0, 1))
  '{<low>,<high>}' -> ('greedy-quant', (int(<low>), int(<high>)))

How regex parts map to tokens:
- character class: (type='char-class', data=<set>)
- Dot: (type='char-class', data=<set>). Here <set> is either the set of all
  ascii characters, or all except '\n'. This depends on the DOTALL flag's
  presense.
- '\<digits>': (type='bref', data=int(<digits>))
- '\d', '\D', '\s', '\S', '\w', '\W': (type='char-class', data=<set>).
- '^', '$', '\A', '\b', '\B', '\Z', '|', '(', '(?:', '(?=', '(?!', ')': (type)
  matches operator strings, (data) is None. For example, '(?=' maps to
  (type='(?=', data=None).
- '+', '*', '?', '{m,n}': (type='greedy-quant', data=(<low>,<high>))
"""

tokenfuncs = []

# tns = tokenization name space
tns = SimpleNamespace('regstr'=None, 'pos'=None, flags=None)

def exhausted():
    return len(tns.regstr) == tns.pos

def takech(inc=False):
    if exhausted():
        return None
    ch = tns.regstr[tns.pos]
    if inc:
        tns.pos += 1
    return ch

def rest():
    return tns.regstr[tns.pos:]

# not needed
def seek(i, how='cur'):
    if how == 'cur':
        tns.pos = max(0, min(len(tns.regstr), tns.pos + i))
        return tns.pos
    elif how == 'set':
        tns.pos = max(0, min(len(tns.regstr), i))
    else:
        raise ValueError(f'Bad how: "{how}"')

def token_error(msg):
    # just a helper
    raise ValueError(f'{msg}: "{tns.regstr}".')
    
def tokenfunc(func):
    tokenfuncs.append(func)
    return func

# main function
def tokenize(regstr, flags):
    """Transforms (regstr) into a list of tokens and returns it. Raises
    ValueError if (regstr) is flawed."""
    tns.regstr, tns.pos, tns.flags = regstr, 0, flags
    tokens = []
    size = len(regstr)
    while not exhausted():
        for tf in tokenfuncs:
            token = tf()
            if token is not None:
                tokens.append(token)
                break
        else:
            raise ValueError(f'Cannot extract a token from "{rest()}"')
    return tokens

@tokenfunc
def simple():
    """For tokens whose type matches the characters in the regex string, and
    whose data is None."""    
    types = ('^', '$', r'\A', r'\b', r'\B', r'\Z', '|', '(', '(?:', '(?=',
             '(?!', ')')
    for t in types:
        if tns.regstr.startswith(t):
            tns.pos += len(t)
            return Token(type=t, data=None)
    else:
        return None

def _digits(astr, i):
    """Returns the longest digits-only substring starting at (i)."""
    digits = [astr[i]]
    for i in range(i+1, len(astr)):
        ch = astr[i]
        if ch.isdigit():
            digits.append(ch)
        else:
            break
    return ''.join(digits)
        
@tokenfunc
def paren():
    for prefix in '(?:', '(?=', '(?!', '(', ')':
        if tns.regstr.startswith(prefix, i):
            return (prefix, prefix), len(prefix)
    return (None, None)

@tokenfunc
def char_class():
    """Tries to extract a character class from the regex string. Returns a token
    of the form Token(type='char-class', data=<chars>), where <chars> is the set
    of characters in the class."""
    if takech() != '[':
        return None
    # Find index of closing ']' and store it in (cbi). Start at tns.pos+2
    # because a beginning ']' does not indicate the end of the class.
    cbi = tns.regstr.find(']', tns.pos+2)
    while True:
        if cbi == -1:
            raise ValueError(f'No closing bracket: {rest()}')
        if tns.regstr[cbi-1] == '\\':
            cbi = tns.regstr.find(']', cbi+1)
    template = tns.regstr[tns.pos+1:cbi]
    tns.pos = cbi + 1
    return Token('char-class', _form_class(template))

def _form_class(template):
    """Processes the insides of a character class (template) into a set of
    characters."""
    assert template # (template) should not be empty

    def error(msg):
        # just a helper
        raise ValueError(f'Bad character class template "{template}": {msg}')
    
    tempiter = iter(template)
        
    # stage 1: takech care of initial '^'.
    negate = False
    first = next(tempiter)
    if first == '^':
        negate = True
    else: 
        tempiter = itertools.chain([first], tempiter) # put (first) back

    tokens = deque()
    # stage 2: takech care of backlashes. Every element of (tokens) after this
    # will be a character or a set of characters or a '--'. Tokens in this
    # context have nothing to do with regex tokens that are the result of
    # (tokenize).
    for char in tempiter:
        if char == '-':
            tokens.append('--')
        elif char == '\\':
            nxt = next(tempiter, None)
            if nxt is None:
                error('Missing character after backlash.')
            if nxt in CLASS_SHORTS:
                tokens.append(CLASS_SHORTS[nxt])
            elif nxt in ESCAPE_CHARS:
                tokens.append(ESCAPE_CHARS[nxt])
            elif nxt == 'x':
                h1, h2 = next(tempiter, None), next(tempiter, None)
                if (h1 is None or h1 not in HEX_DIGITS
                    or h2 is None or h2 not in HEX_DIGITS):
                    error('Expected two hex digits after "\\x"')
                tokens.append(chr(int(h1+h2, 16)))
            else:
                error(f'Bad char after "\\": "{char}"')
        else:
            tokens.append(char)

    # stage 3: process spans (like 'a-z') and shorthands. 
    result = set()
    while tokens:
        token = tokens.popleft()
        if token == '--':
            result.add('-')
        elif type(token) is set:
            result.update(set)
        else: # (token) is a character            
            if len(tokens) < 2: # (token) is not part of a span
                result.add(token)
            else:
                nxt = tokens.popleft()
                if nxt == '--':
                    span_end = tokens.popleft()
                    if type(span_end) is set:
                        error('Invalid input after "-".')
                    elif span_end == '--':
                        span_end = '-'
                    chars = [chr(k) for k in range(ord(token), ord(span_end)+1)]
                    if not chars:
                        error(f'Bad range {token}-{span_end}.')
                    result.update(chars)
                else:
                    tokens.appendleft(nxt) # put (nxt) back
                    result.add(token)

    return result if not negate else ALL-result

@tokenfunc
def greedy_quant():
    def error(extra=None):
        # just a helper
        msg = f'Invalid quantifier "{rest()}"'
        if extra is not None:
            msg = f'{msg}: {extra}'
        raise ValueError(msg)
    
    ch = takech()
    bounds = {'*': (0, None), '+': (1, None), '?': (0, 1)}.get(ch)
    if bounds is not None:
        tns.pos += 1
        return Token(type='greedy-quant', data=bounds)
    elif ch == '{':
        endi = tns.regstr.find('}', tns.pos)
        if endi == -1:
            error('Missing closing "{".')
        bounds = tns.regstr[tns.pos+1:endi].split(',')
        if len(bounds) != 2:
            error("One comma required. The format is '{m,n}'")
        try:
            low, high = map(int, bounds)
        except ValueError:
            error("Bounds must be parsable to integers.")
        if low > high:
            error(f'<low> must not exceed <high>.')
        tns.pos = endi+1
        return ('greedy-quant', (low, high))
    else:
        return None

@tokenfunc
def char_class_shorts():
    """Handles character class shorthands, like r'\d'."""
    ch = take(True)
    chars = None # the set of characters
    if ch == '.':
        chars = (ALL if common.contains_flag(tns.flags, common.DOTALL)
                 else DOTNOALL)
    elif ch == '\\':
        nxt = take(True)
        if nxt is None:
            raise ValueError(f'Can\'t end in backlash: "{tns.regstr}".')
        chars = CLASS_SHORTS.get(nxt)
        if nxt in CLASS_SHORTS:
            
            
            

@tokenfunc
def backref():
    raise NotImplementedError

@tokenfunc
def char():
    """Forms 'char' tokens. Call after other tokenization functions so that
    special characters are not interpreter as regular ones. For example, if the
    current char of the regex string is '*', this will interpret it as a
    character token rather than as a quantifier. As another example, this
    function will raise an error with r'\A' because 'A' is not a valid escape
    character, even though r'\A' is a valid regex. So if this is called before
    the function that processes r'\A' regexes, an error will ensue, which may
    not be desirable."""

    ch = takech(True)
    if ch == '\\':
        nxt = takech(True)
        if nxt is None:
            token_error('Cannot end in backlash.')
        elif nxt in SPECIAL:
            return Token('char', SPECIAL)
        elif nxt == 'x':
            x1, x2 = takech(True), takech(True)
            if (x1 is None or x1 not in HEX_DIGITS
                or x2 is None or x2 not in HEX_DIGITS):
                token_error('Expected two hex digits after "\x"')
            return Token('char', chr(int(x1+x2, 16)))
        elif nxt in ESCAPE_CHARS:
            return Token('char', ESCAPE_CHARS[nxt])
        else:
            token_error(f'"{nxt}" is not a valid escape character, at {tns.pos}.')
    else:
        tns.pos += 1
        return Token('char', ch)
