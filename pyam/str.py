import re

import numpy as np
import pandas as pd
from pandas.api.types import is_list_like

REGEXP_CHARACTERS = r".^$+?()[]{}|"


def concat_with_pipe(x, *args, cols=None):
    """Concatenate a list or pandas.Series using ``|``, drop None or numpy.nan"""
    if args:
        # Guard against legacy-errors when adding `*args` (#778)
        # TODO: deprecated, remove for release >= 3.1
        for i in args:
            if is_list_like(i):
                raise DeprecationWarning(f"Please use `cols={i}`.")
        x = [x] + list(args)
    cols = cols or (x.index if isinstance(x, pd.Series) else range(len(x)))
    return "|".join([x[i] for i in cols if x[i] not in [None, np.nan, ""]])


def find_depth(data, s="", level=None):
    """Return or assert the depth (number of ``|``) of variables

    Parameters
    ----------
    data : str or list of strings
        IAMC-style variables
    s : str, default ''
        remove leading `s` from any variable in `data`
    level : int or str, optional
        If None, return depth (number of ``|``); else, return list of booleans
        whether depth satisfies the condition (equality if level is int,
        >= if ``.+``,  <= if ``.-``)
    """
    if is_list_like(level):
        raise ValueError(
            "Level is only run with ints or strings, not lists. Use strings with "
            "integers and + or - to filter by ranges."
        )
    if is_str(data):
        return _find_depth([data], s, level)[0]

    return _find_depth(data, s, level)


def _find_depth(data, s="", level=None):
    """Internal implementation of `find_depth()´"""
    # remove wildcard as last character from string, escape regex characters
    _s = re.compile("^" + escape_regexp(s.rstrip("*")))
    _p = re.compile("\\|")

    # find depth
    def _count_pipes(val):
        return len(_p.findall(re.sub(_s, "", val))) if _s.match(val) else None

    n_pipes = map(_count_pipes, data if is_list_like(data) else list(data))

    # if no level test is specified, return the depth as (list of) int
    if level is None:
        return list(n_pipes)

    # if `level` is given, set function for finding depth level =, >=, <= |s
    if not is_str(level):
        # test = lambda x: level == x if x is not None else False
        def test(x):
            return level == x if x is not None else False

    elif level[-1] == "-":
        level = int(level[:-1])

        # test = lambda x: level >= x if x is not None else False
        def test(x):
            return level >= x if x is not None else False

    elif level[-1] == "+":
        level = int(level[:-1])

        # test = lambda x: level <= x if x is not None else False
        def test(x):
            return level <= x if x is not None else False

    else:
        raise ValueError(f"Unknown level type: `{level}`")

    return list(map(test, n_pipes))


def get_variable_components(x, level, join=False):
    """Return components for requested level in a list or join these in a str.

    Parameters
    ----------
    x : str
        Uses ``|`` to separate the components of the variable.
    level : int or list of int
        Position of the component.
    join : bool or str, optional
        If True, IAMC-style (``|``) is used as separator for joined components.

    Returns
    -------
    str
    """
    _x = x.split("|")
    if join is False:
        return [_x[i] for i in level] if is_list_like(level) else _x[level]
    else:
        level = [level] if isinstance(level, int) else level
        join = "|" if join is True else join
        return join.join([_x[i] for i in level])


def reduce_hierarchy(x, depth):
    """Reduce the hierarchy (indicated by ``|``) of x to the specified depth

    Parameters
    ----------
    x : str
        Uses ``|`` to separate the components of the variable.
    depth : int or list of int
        Position of the components.

    """
    _x = x.split("|")
    depth = len(_x) + depth - 1 if depth < 0 else depth
    return "|".join(_x[0 : (depth + 1)])


def escape_regexp(s):
    """Escape characters with specific regexp use"""
    s = str(s)
    for c in REGEXP_CHARACTERS:
        s = s.replace(c, "\\" + c)
    # pyam uses `*` as wildcard, replace with `.*` for regex
    s = s.replace("*", ".*")
    return s


def is_str(x):
    """Returns True if x is a string"""
    return isinstance(x, str)
