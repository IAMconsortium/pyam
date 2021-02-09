from pyam.utils import islistable

# lists for super/subscript conversion tools
NORMALSCRIPT = '0123456789x'
SUBSCRIPT = '₀₁₂₃₄₅₆₇₈₉ₓ'
SUPERSCRIPT = '⁰¹²³⁴⁵⁶⁷⁸⁹ˣ'

# string utils
MAKE_SUP = str.maketrans(NORMALSCRIPT, SUPERSCRIPT)
MAKE_SUB = str.maketrans(NORMALSCRIPT, SUBSCRIPT)
UNDO_SUP = str.maketrans(SUPERSCRIPT, NORMALSCRIPT)
UNDO_SUB = str.maketrans(SUBSCRIPT, NORMALSCRIPT)


def format_superscript(x):
    """Format numbers and x in a string (or list of strings) to superscripts"""
    return _maketrans(x, MAKE_SUP)


def format_subscript(x):
    """Format numbers and x in a string (or list of strings) to subscripts"""
    return _maketrans(x, MAKE_SUB)


def unformat_superscript(x):
    """Remove formatting of superscripts from a string (or list of strings)"""
    return _maketrans(x, UNDO_SUP)


def unformat_subscript(x):
    """Remove formatting of subscripts from a string (or list of strings)"""
    return _maketrans(x, UNDO_SUB)


def _maketrans(x, method):
    """Utility function to apply a formatting method to a str or list of str"""
    if islistable(x):
        return [_maketrans(i, method) for i in x]
    return x.translate(method)
