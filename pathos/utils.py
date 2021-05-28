from itertools import tee, zip_longest

def gv_escape(s):
    res = []
    for c in s:
        if c == '\\':
            res.append('\\\\')
        elif c == '"':
            res.append('\\"')
        else:
            res.append(c)
    return f'"{"".join(res)}"'

def pairwise_longest(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..., (s_n, None)"
    a, b = tee(iterable)
    next(b, None)
    return zip_longest(a, b)