import numpy as np
import pandas as pd

from pyam.exceptions import raise_data_error


def get_index_level_number(index, level):
    """Return the number of a specific level"""
    return index._get_level_number(level)


def get_index_levels(index, level):
    """Return the labels for a specific level"""

    if not isinstance(index, pd.Index):
        index = index.index  # assume that the arg `index` is a pd.DataFrame

    if isinstance(index, pd.MultiIndex):
        return list(index.levels[get_index_level_number(index, level)])

    # if index is one-dimensional, make sure that the "level" is the name
    if index.name != level:
        raise KeyError("Index does not have a level {level}")
    return list(index)


def get_index_levels_codes(df, level):
    """Return the category-values and codes for a specific level"""
    n = get_index_level_number(df.index, level)
    return df.index.levels[n], df.index.codes[n]


def get_keep_col(codes, matches):
    """Return boolean mask where *matches* appear in *codes*

    *matches* can be given as either:
    1. A boolean mask against the levels of a multiindex, or
    2. An subset of integers in *codes*
    """
    matches = np.asanyarray(matches)

    if np.issubdtype(matches.dtype, "bool"):
        (matches,) = np.where(matches)

    return np.isin(codes, matches)


def replace_index_values(df, name, mapping, rows=None):
    """Replace one or several category-values at a specific level (for specific rows)"""
    index = df if isinstance(df, pd.Index) else df.index

    n = get_index_level_number(index, name)

    # if replacing level values with a filter (by rows)
    if rows is not None and not all(rows):
        _levels = pd.Series(index.get_level_values(n))
        renamed_index = replace_index_values(index[rows], name, mapping)
        _levels[rows] = list(renamed_index.get_level_values(n))
        _unique_levels = pd.Index(_levels.unique())

        return append_index_level(
            index=index.droplevel(n),
            codes=_unique_levels.get_indexer(_levels),
            level=_unique_levels,
            name=name,
            order=index.names,
        )

    # else, replace the level values for the entire index dimension
    _levels = index.levels[n].map(lambda level: mapping.get(level, level))
    _unique_levels = _levels.unique()

    # if no duplicate levels exist after replace, set new levels and return
    if len(index.levels[n]) == len(_unique_levels):
        return index.set_levels(_levels, level=n)

    # if duplicate levels exist, re-map the codes
    level_mapping = _unique_levels.get_indexer(_levels)
    _codes = np.where(index.codes[n] != -1, level_mapping[index.codes[n]], -1)
    return index.set_codes(_codes, level=n).set_levels(_unique_levels, level=n)


def replace_index_labels(index, name, labels):
    """Replace the labels for a specific level"""

    n = get_index_level_number(index, name)
    codes = index.codes[n]
    return append_index_level(index.droplevel(n), codes, labels, name, index.names)


def append_index_col(index, values, name, order=False):
    """Append a list of `values` as a new column (level) to an `index`"""
    levels = pd.Index(values).unique()
    codes = levels.get_indexer(values)

    return append_index_level(index, codes, levels, name, order)


def append_index_level(index, codes, level, name, order=False):
    """Append a level to a pd.MultiIndex"""
    if isinstance(level, str):
        level = [level]
        codes = [codes] * len(index.codes[0])

    new_index = pd.MultiIndex(
        codes=index.codes + [codes],
        levels=index.levels + [level],
        names=index.names + [name],
    )
    if order:
        new_index = new_index.reorder_levels(order)
    return new_index


def verify_index_integrity(df):
    """Verify integrity of index

    Arguments
    ---------
    df : Union[pd.DataFrame, pd.Series, pd.Index]

    Raises
    ------
    ValueError
    """
    index = df if isinstance(df, pd.Index) else df.index
    if not index.is_unique:
        overlap = index[index.duplicated()].unique()

        raise_data_error(
            "Timeseries data has overlapping values", overlap.to_frame(index=False)
        )
