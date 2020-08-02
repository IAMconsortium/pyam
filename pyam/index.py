import pandas as pd


def get_index_levels(df, level):
    """Return the levels for a specific level"""
    for i, n in enumerate(df.index.names):
        if n == level:
            return list(df.index.levels[i])

    raise ValueError(f'This object does not have an index level `{level}`!')


def replace_index_value(df, level, current, to, verify_integrity=True):
    """Replace a value in a particular index level"""
    levels = []
    has_level = False
    for n, l in zip(df.index.names, df.index.levels):
        if n == level:
            levels.append([to if i == current else i for i in l])
            has_level = True
        else:
            levels.append(l)
    if not has_level:
        msg = f'This object does not have an index level `{level}`!'
        raise ValueError(msg)

    return pd.MultiIndex(codes=df.index.codes, names=df.index.names,
                         levels=levels, verify_integrity=verify_integrity)
