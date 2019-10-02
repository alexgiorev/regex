import itertools
import string

from collections import deque

ALL = frozenset(chr(i) for i in range(128)) # whole ALL character set.
DIGIT = frozenset(string.digits)
ALPHANUMERIC = frozenset(string.ascii_letters + string.digits + '_')
WHITESPACE = frozenset(string.whitespace)
CHARSHORTS = {'d': DIGIT,
              'D': ALL - DIGIT,
              's': WHITESPACE,
              'S': ALL - WHITESPACE,
              'w': ALPHANUMERIC,
              'W': ALL - ALPHANUMERIC}
SPECIAL = r'\[].^()|+*?{}'
HEXDIGITS = f'0123456789abcdefABCDEF'
ESCAPECHARS = {'a': '\a', 'f': '\f', 'n': '\n', 'r': '\r',
               't': '\t', 'v': '\v', 'b': '\b'}

tokenfuncs = []

def tokenfunc(func):
    tokenfuncs.append(func)
    return func

def tokenize(regstr, flags):
    """
    A token is a pair (type, lexeme). For some tokens, due to a lack of
    imagination, the type and lexeme match. For example, for the token '(?:',
    the type is also written as '(?:' so that the whole token is ('(?:', '(?:')

    The possible tokens are:
    - characters: ('char', <char>). For example, ('char', 'A'), ('char', '\n')
    - Parenthesis tokens. The types and the lexemes match. All possible types
      are {'(', '(?:', '(?=', '(?!', ')'}
    - Character classes: ('char-class', set). Here (set) is a set of characters
      which appear in the class. So if the text is '[a-d1-5]', the token will be
      ('char-class', set('abcd')|set('12345'))
    - {'^', '$', '\A', '\Z', '\b', '\B'}. Zero width assertions. As with
      parenthesis, the types and the lexemes match.
    - Group references: ('group-index', int) where (int) is an integer >= 1
    - ('|', '|')
    - ('greedy-quant', (int, int)). See below for details.

    r'\d', r'\D', r'\s', r'\S', r'\w', r'\W' simply  expand to character
    classes. For example, r'\d' will expand to ('char-class', set('0123456789'))

    The greedy quantifiers all expand to the same type of token 'greedy-quant':
    '*' -> ('greedy-quant', (0, None))
    '+' -> ('greedy-quant', (1, None))
    '?' -> ('greedy-quant', (0, 1))
    {m,n} -> ('greedy-quant', (int(m), int(n)))
    """
    
    tokens = []
    i, size = 0, len(regstr)
    while i < size:
        for tf in tokenfuncs:
            token, taken = tf(regstr, i)
            if token is not None:
                tokens.append(token)
                i += taken
                break
        else:
            raise ValueError(f'Cannot extract a token from "{regstr[i:]}"')
    return tokens

@tokenfunc
def backlash(regstr, i):    
    def error():
        raise ValueError(f'Bad escape: "{regstr[i]}"')

    if regstr[i] != '\\':
        return (None, None)    
    if i == len(regstr) - 1:
        error()
    nxt = regstr[i+1]
    dct = {'A': (r'\A', r'\A'), 'b': (r'\b', r'\b'),
           'B': (r'\B', r'\B'), 'Z': (r'\Z', r'\Z')}    
    if nxt in dct:
        return dct[nxt], 2        
    if nxt in CHARSHORTS:
        return ('char-class', CHARSHORTS[nxt]), 2
    if nxt in SPECIAL:
        return ('char', nxt), 2
    if nxt.isdigit():
        digits = _digits(regstr, i+1)
        return ('group-index', int(digits)), len(digits)
    if nxt == 'x':
        nxt2 = regstr[i+2: i+4]
        if len(nxt2) < 2 or all(c in HEXDIGITS for c in nxt2):
            error()
        return ('char', chr(int(nxt2, 16))), 4
    if nxt in ESCAPECHARS:
        return ('char', ESCAPECHARS[nxt]), 2
    error()

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
def paren(regstr, i):
    for prefix in '(?:', '(?=', '(?!', '(', ')':
        if regstr.startswith(prefix, i):
            return (prefix, prefix), len(prefix)
    return (None, None)

@tokenfunc
def char_class(regstr, i):
    if regstr[i] != '[':
        return (None, None)
    # Find index of closing ']'. Start at i+2 because a beginning ']' does not
    # indicate the end of the class.
    for j in range(i+2, len(regstr)):
        ch = regstr[j]
        if ch == ']' and regstr[j-1] != '\\':
            break
    else:
        raise ValueError(f'Missing closing "]" for class: "{regstr[i:]}"')
    return ('char-class', _form_class(regstr[i+1:j])), j+1-i

def _form_class(chars):
    """Assumes (chars) is a non-empty iterator of characters."""
    chars = iter(chars)
    
    # stage 1: take care of initial '^'.
    negate = False
    first = next(chars)
    if first == '^':
        negate = True
    else: 
        chars = itertools.chain([first], chars) # put (first) back

    # stage 2: take care of backlashes. Every token will be a character or a set
    # of characters or a '--'
    tokens = deque()
    for char in chars:
        if char == '-':
            tokens.append('--')
        elif char == '\\':
            nxt = next(chars, None)
            if nxt is None:
                raise ValueError(f'Missing char after backlash: [{chars}]')
            if nxt in CHARSHORTS:
                tokens.append(CHARSHORTS[nxt])
            elif nxt in ESCAPECHARS:
                tokens.append(ESCAPECHARS[nxt])
            elif nxt == 'x':
                h1, h2 = next(chars, None), next(chars, None)
                if (h1 is None or h1 not in HEXDIGITS
                    or h2 is None or h2 not in HEXDIGITS):
                    raise ValueError(f'Expected two hex digits after "\\x": [{chars}]')
                tokens.append(chr(int(h+h, 16)))
            else:
                raise ValueError(f'Bad escape: [{chars}]')
        else:
            tokens.append(char)

    # stage 3;
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
                        raise ValueError('Bad character set; set after dash.')
                    elif span_end == '--':
                        span_end = '-'
                    chars = [chr(k) for k in range(ord(token), ord(span_end)+1)]
                    if not chars:
                        raise ValueError(f'Bad character set: invalid span '
                                         f'between "{token}" and "{span_end}"')
                    result.update(chars)
                else:
                    tokens.appendleft(nxt)
                    result.add(token)

    return result if not negate else ALL-result

@tokenfunc
def selfeval(regstr, i):
    char, chars = regstr[i], '^$|'    
    return ((char, char), 1) if char in chars else (None, None)

@tokenfunc
def greedy_quant(regstr, i):
    def error():
        raise ValueError(f'Invalid quantifier at {i}: "{regstr}"')
    
    ch = regstr[i]
    if ch == '*':
        return ('greedy-quant', (0, None)), 1
    elif ch == '+':
        return ('greedy-quant', (1, None)), 1
    elif ch == '?':
        return ('greedy-quant', (0, 1)), 1
    elif ch == '{':
        j = regstr.find('}', i)
        if j == -1:
            error()
        bounds = regstr[i+1:j].split(',')
        if len(bounds) != 2:
            error()
        try:
            m, n = map(int, bounds)
        except ValueError:
            error()
        if m > n:
            error()
        return ('greedy-quant', (m, n)), j+1-i
    else:
        return (None, None)    
    
@tokenfunc
def char(regstr, i):
    return ('char', regstr[i]), 1

@tokenfunc
def dot(regstr, i):
    raise NotImplementedError

if __name__ == '__main__':
    assert _form_class('abc') == set('abc')
    assert _form_class('a-d') == set('abcd')
    assert _form_class('a-z') == set(string.ascii_lowercase)
    assert _form_class('0-9') == set(string.digits)
