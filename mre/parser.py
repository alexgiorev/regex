import itertools
import string
from collections import namedtuple, deque
from types import SimpleNamespace as NS

from . import common
from .llist import dllist


# ----------------------------------------
# Parsing

Token = namedtuple('Token', 'type data')
ExprTree = namedtuple('ExprTree', 'token args grpis')

# A set of the token types which correspond to unary operators. There is no need
# to keep track of precedence or associativity, because only postfix unary
# operators are used, and all unary operators have greater precedence than all
# binary operators.
unops = {'greedy-quant'}

def isunop(token):
    return token.type in unops

# Order operators based on precedence. Each element of the list contains an
# associativity and a set of operators which have that associativity. All
# operators in a (binops) element have the same precedence, which is less than
# that of previous elements' operators, and greater than that of next elements'
# operators. In other words, operators are in descending precedence order.
binops = [
    NS(assoc='left', ops={'product'}),
    NS(assoc='left', ops={'|'})
]

# What follows is a script which transforms (binops) into a dict which maps to a
# binary operator it's precedence and associativity. The current version of
# (binops) is easy for humans to manipulate, the new version will be easy for
# programs to query.

new_binops = {}
for binfo, prec in zip(binops, range(len(binops), 0, -1)):
    assoc, ops = binfo.assoc, binfo.ops
    for op in ops:
        new_binops[op] = NS(assoc=assoc, prec=prec)
binops = new_binops

def isbinop(token):
    return token.type in binops

def binfo(token):
    return binops[token.type]

def token_category(token):
    """Tokens can be grouped into 3 categories:
    - primitives: letters, zero-width assertions, etc.
    - operators.
    - parenthesis.
    This functions returns one of ('primitive', 'operator', 'parenthesis')
    """
    
    t = token.type
    if t.startswith('(') or t.endswith(')'):
        return 'parenthesis'
    elif t in unops or t in binops:
        return 'operator'
    else:
        return 'primitive'

class _TokenList:
    """Used to efficiently iterate over the tokens."""
    
    def __init__(self, base, start=0, end=None):
        self.base = base
        self.start = start
        self.end = len(base) - 1 if end is None else end
        self.current = start

    def __next__(self):
        if self.current > self.end:
            raise StopIteration
        result = self.base[self.current]
        self.current += 1
        return result

    def subtokens(self):
        """A paren token was just encountered. This returns a TokenList
        containing all tokens within opening paren that was just passed and it's
        corresponding closing. After this call, (self)'s index will be on the
        token following the closing parenthesis."""
        
        start = self.current # the index of the token after the opening paren
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

        # at this point, (self.current-1) is the index of the closing paren, so
        # we want the tokens from (start) to (self.current-2)
        return _TokenList(self.base, start, self.current-2)
    
    def __iter__(self):
        return self

# global parsing namespace. Contains data that is useful for parsing related
# functions. Initialized at each call to (parse).
pns = NS(context=None)

def parse(regstr, flags):
    """Creates the expression tree and the context."""
    pns.context = common.Context(0, flags)
    tokens = _TokenList(tokenize(regstr, flags))
    return _parse(tokens), pns.context

def _parse(tokens):
    """Parses (tokens) to an ExprTree. Fills out the context. Raises ValueError
    if there is a problem."""
    
    def make_opsargs():
        """In this function, the token sequence is transformed into a sequence
        of ExprTrees (the operands) and operators. During this pass, we also
        remember the positions of unary and binary operators for use in later
        parsing stages. In addition, upon return, the context will be completed
        (e.g. the number of groups will be known).

        Intuition behind this function: I think of an expression, whether
        arithmetic or regex or whatever, as a sequence of operators and
        operands. This simply transforms the token sequence to match this view
        of expressions.

        Note: this function infers the location of product operators and adds
        them. When we have an operand or unary operator that is directly
        followed by another operand, a product operator must be inserted
        in-between. For example, r'ab' contains the tokens 'a' and 'b', but the
        expression is actually 'a<X>b', where <X> stands for the product
        operator. Also consider r'a+b'. The expression that is actually meant is
        r'(a+)<X>b'."""

        # a doubly linked list containing operands and their arguments.
        opsargs = dllist()

        # a list of the positions of the unary operators within (opsargs)
        unops = []
        
        # a dict which maps precedences to pairs (assoc, posns), where (assoc)
        # is the associativity of the binary operators at (posns).
        binops = {}

        def add_unop(token):
            pos = opsargs.append(token)
            unops.append(pos)
        
        def add_binop(token):
            bi = binfo(token)
            pos = opsargs.append(token)
            _, posns = binops.setdefault(bi.prec, (bi.assoc, []))
            posns.append(pos)
        
        # The flag below is needed to determine if a product operator ought to
        # be inserted.
        last_is_expr_or_unop = False
            
        for token in tokens:
            category = token_category(token)
            if category == 'primitive':
                subexpr = ExprTree(token, args=None, grpis=[])
                if last_is_expr_or_unop:
                    add_binop(Token('product', None))
                opsargs.append(subexpr)
                last_is_expr_or_unop = True
            elif category == 'operator':
                if isunop(token):
                    add_unop(token)
                    last_is_expr_or_unop = True
                else:
                    add_binop(token)
                    last_is_expr_or_unop = False
            else: # is a parenthesis
                subexpr = _parse(tokens.subtokens())
                paren = token.type
                if paren == '(':
                    pns.context.numgrps += 1
                    subexpr.grpis.append(pns.context.numgrps)
                elif paren == '(?:':
                    pass
                elif paren in ('(?=', '(?!'):
                    subexpr = ExprTree(token, args=[subexpr], grpis=[])
                elif paren == ')':
                    raise ValueError(f'Unneccessary closing parenthesis.')
                else:
                    raise AssertionError(f'This should not happen: {token}')
                if last_is_expr_or_unop:
                    add_binop(Token('product', None))
                opsargs.append(subexpr)
                last_is_expr_or_unop = True

        binops = [item[1] for item in sorted(binops.items(), reverse=True)]        
        return opsargs, unops, binops

    def process_unary_operators(opsargs, unops):
        """Assumes (opsargs) contains only (ExprTree)s and operator tokens. Some
        of those operators will be unary. This phase processes them out, so that
        what is left will be an alternating sequence of (ExprTree)s and binary
        operators. To achieve this, (unops) is used. It must be a sequence of
        the positions of the unary operators."""
        
        for unop_pos in unops:
            arg_pos = unop_pos.prev
            if arg_pos is None:
                raise ValueError(f'Missing unary operator argument.')
            operand = arg_pos.value
            if type(operand) is not ExprTree:
                raise ValueError(f'Bad unary operator argument.')
            subexpr = ExprTree(token=unop_pos.value, # the operator token
                               args=[operand], grpis=[])
            arg_pos.value = subexpr
            opsargs.remove(unop_pos)
                
    def process_binary_operators(opsargs, binops):
        """At this point, (opsargs) should be an alternating sequence of
        ExprTrees and binary operator tokens. The first and last elements should
        be ExprTrees. This function reduces (opsargs) to a single ExprTree."""

        def getarg(pos):
            """helper for the loop below."""
            if pos is None:
                raise ValueError(f'Missing operand.')
            value = pos.value
            if type(value) is not ExprTree:
                raise ValueError(f'Bad operand.')
            return value
        
        for assoc, binop_posns in binops:
            if assoc == 'right':
                binop_posns = reversed(binop_posns)
            for binop_pos in binop_posns:
                left, right = binop_pos.prev, binop_pos.next
                arg1, arg2 = getarg(left), getarg(right)
                subexpr = ExprTree(token=binop_pos.value,
                                   args=[arg1, arg2],
                                   grpis=[])
                binop_pos.value = subexpr
                opsargs.remove(left)
                opsargs.remove(right)

    opsargs, unops, binops = make_opsargs()
    process_unary_operators(opsargs, unops)
    process_binary_operators(opsargs, binops)
    return opsargs.first.value

# ----------------------------------------
# Tokenization

ALL = frozenset(chr(i) for i in range(2**7)) # whole character set.
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

"""
Tokenization here is not a strictly syntactical operation. It also does some
preliminary processing (like determining the characters in a character set, the
bounds of a greedy quantifier and more.)

The character set used is {chr(i) for i in range(256)}, so all characters whose
encoding can fit in a byte. The idea is to support all ASCII characters. I chose
this set for simplicity; the purpose of this project is to practice, nobody will
really use it.

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
tns = NS(regstr=None, pos=None, flags=None)

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

def token_error(msg):
    # just a helper
    raise ValueError(f'{msg}: "{tns.regstr}"')
    
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
    # In the tuple below, make sure that if (t1) is a prefix of (t2), (t1) comes
    # after (t2). Consider an example which violates this and the consequence:
    # If '(' comes before '(?=', if '(?=' is a prefix of the regex string at
    # (tns.pos), the token that will be extracted is '(', not '(?=' as should be
    # the case.
    types = ('^', '$', r'\A', r'\b', r'\B', r'\Z', '|',
             '(?:', '(?=','(?!', '(', ')')
    for t in types:
        if tns.regstr.startswith(t, tns.pos):
            tns.pos += len(t)
            return Token(type=t, data=None)
    else:
        return None
        
@tokenfunc
def char_class():
    """Tries to extract a character class from the regex string. Returns a token
    of the form Token(type='char-class', data=<chars>), where <chars> is the set
    of characters in the class.

    Character class syntax: A template is what appears inside the brackets enclosing
    the class (e.g. 'a-z' in [a-z]). I will describe the fine points related to how
    a template maps to the character set that makes up the class. All of the rest is
    the same as in Python.

    There are a couple of points to consider:

    1. Caret in the beginning

    If the template begins with '^', the char set will be the negation of that
    determined by the rest of the template. As a special case, a template of '^'
    raises an error, because '' is not a valid template. For the rest of this
    docstring, assume the template does not begin with '^'.

    2. Hyphens

    Hyphens have two purposes:
    - They stand for themselves. This happens in the following cases:
      - When they appear at the beginning or end of the template. As a special
        case, '[^-]' is the set of all chars except '-'
      - When they are preceded by an unescaped backlash.
      - When they can't possibly stand for a range. For example, in '0-5-9',
        '0-5' is a range, but the hyphen that follows can't be one, because it
        makes no sense to have a range whose lower bound is another range. So
        the second hyphen stands for itself to result in the charset
        set('012345-9').
    - They delimit ranges.

    3. Character class shorthands, like r'\d', r'\w', etc. If a letter which
       stands for a char-class shorthand is preceded by an unescaped backlash,
       the resulting class will contain the characters of the inner class.

    4. backlashes. Normal escape characters can be included. Also, the
       characters special to char classes - r'\[]^-' - can be included by
       backlashing them.
    """
    
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
        else:
            break
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
        
    # stage 1: take care of initial '^'.
    negate = False
    first = next(tempiter)
    if first == '^':
        negate = True
    else: 
        tempiter = itertools.chain([first], tempiter) # put (first) back

    tokens = deque()
    # stage 2: take care of backlashes. Every element of (tokens) after this
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
            elif nxt == '-':
                tokens.append('-')
            elif nxt in CLASS_SHORTS:
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
        elif type(token) in (set, frozenset):
            result.update(token)
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
        return Token('greedy-quant', (low, high))
    else:
        return None

@tokenfunc
def char_class_shorts():
    """Handles character class shorthands, like r'\d'. The dot is also
    considered such as shorthand."""
    ch = takech(True)
    chars = None # the set of characters
    if ch == '.':
        chars = (ALL if common.contains_flag(tns.flags, common.DOTALL)
                 else DOTNOALL)
    elif ch == '\\':
        nxt = takech(True)
        if nxt is None:
            token_error(f"Can't end in backlash.")
        chars = CLASS_SHORTS.get(nxt)
        if chars is None:
            tns.pos -= 2 # put (ch) and (nxt).
            return None
    else:
        tns.pos -= 1 # put back (ch)
        return None
    return Token('char-class', chars)
            
@tokenfunc
def backref():
    ch = takech()
    if ch != '\\':
        return None
    tns.pos += 1
    nxt = takech()
    if nxt is None:
        token_error(f"Can't end in backlash.")
    if nxt.isdigit():
        return Token('bref', int(extract_digits()))
    else:
        tns.pos -= 1 # put back (ch)
        return None

def extract_digits():
    """Helper function. Assumes the current regex string char is a
    digit. Extracts all digits from the current position, and returns
    them. Increments (tns.pos) in the process, so that it will point to the
    first non-digit char."""
    digits = []
    while True:
        digits.append(ch)
        tns.pos += 1
        if exhausted():
            break
        ch = takech()
        if not ch.isdigit():
            break
    return ''.join(digits)
    
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
                token_error('Expected two hex digits after "\\x"')
            return Token('char', chr(int(x1+x2, 16)))
        elif nxt in ESCAPE_CHARS:
            return Token('char', ESCAPE_CHARS[nxt])
        else:
            token_error(f'"{nxt}" is not a valid escape character, at {tns.pos}.')
    else:
        return Token('char', ch)
