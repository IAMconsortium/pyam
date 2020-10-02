import pandas as pd


def get_index_levels(df, level):
    """Return the category-values for a specific level"""
    return list(df.index.levels[df.index._get_level_number(level)])


def replace_index_values(df, level, mapping):
    """Replace one or several category-values at a specific level"""
    index = df.index.copy()
    n = index._get_level_number(level)

    # replace the levels
    _levels = [mapping[i] if i in mapping else i for i in index.levels[n]]
    _unique_levels = list(set(_levels))

    # if no duplicate levels exist after replace, set new levels and return
    if len(_levels) == len(_unique_levels):
        return index.set_levels(_levels, n)

    # if duplicate levels exist, re-map the codes
    levels_mapping = dict([(i, _unique_levels.index(lvl))
                           for i, lvl in enumerate(_levels)])
    _codes = [levels_mapping[i] for i in index.codes[n]]
    return index.set_codes(_codes, n).set_levels(_unique_levels, n)


def append_index_level(index, codes, level, name, order=False):
    """Append a level to a pd.MultiIndex"""
    new_index = pd.MultiIndex(
        codes=index.codes + [codes],
        levels=index.levels + [level],
        names=index.names + [name])
    if order:
        new_index = new_index.reorder_levels(order)
    return new_index
