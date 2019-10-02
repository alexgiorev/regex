# -------------------- FLAGS --------------------

I = IGNORECASE = {1}
M = MULTILINE = {2}
S = DOTALL = {3}

def _contains_flag(flags, flag):    
    return flag <= flags
