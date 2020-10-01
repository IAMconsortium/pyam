import pandas as pd


def get_index_levels(df, level):
    """Return the category-values for a specific level"""
    return list(df.index.levels[df.index._get_level_number(level)])


def replace_index_values(df, level, mapping):
    """Replace one or several category-values at a specific level"""
    index = df.index.copy()
    n = index._get_level_number(level)
    unused_levels = False

    for key, value in mapping.items():
        _levels = list(index.levels[n])
        try:  # replace in index codes if value (target) is in the index levels
            _k = _levels.index(key)
            _v = _levels.index(value)
            index = index.set_codes([_v if c == _k else c
                                     for c in index.codes[n]], n)
            unused_levels = True
        except ValueError:  # else replace key for value in the levels
            index = index.set_levels([value if i == key else i
                                     for i in _levels], n)
    if unused_levels:
        index = index.remove_unused_levels()
    return index

def append_index_level(index, codes, level, name, order=False):
    """Append a level to a pd.MultiIndex"""
    new_index = pd.MultiIndex(
        codes=index.codes + [codes],
        levels=index.levels + [level],
        names=index.names + [name])
    if order:
        new_index = new_index.reorder_levels(order)
    return new_index
