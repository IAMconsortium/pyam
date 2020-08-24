import pandas as pd


def get_index_levels(df, level):
    """Return the category-values for a specific level"""
    return list(df.index.levels[df.index._get_level_number(level)])


def replace_index_values(df, level, mapping):
    """Replace one or several category-values at a specific level"""
    return df.index.set_levels([mapping[i] if i in mapping else i
                                for i in get_index_levels(df, level)], level)


def append_index_level(index, codes, level, name, order=False):
    """Append a level to a pd.MultiIndex"""
    new_index = pd.MultiIndex(
        codes=index.codes + [codes],
        levels=index.levels + [level],
        names=index.names + [name])
    if order:
        new_index = new_index.reorder_levels(order)
    return new_index
