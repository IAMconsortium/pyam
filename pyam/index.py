import pandas as pd
import numpy as np

from .utils import _raise_data_error, isstr


def get_index_levels(index, level):
    """Return the category-values for a specific level"""

    if not isinstance(index, pd.Index):
        index = index.index  # assume that the arg `index` is a pd.DataFrame

    if isinstance(index, pd.MultiIndex):
        return list(index.levels[index._get_level_number(level)])

    # if index is one-dimensional, make sure that the "level" is the name
    if index.name != level:
        raise KeyError("Index does not have a level {level}")
    return list(index)


def get_index_levels_codes(df, level):
    """Return the category-values and codes for a specific level"""
    level = df.index._get_level_number(level)
    return df.index.levels[level], df.index.codes[level]


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


def replace_index_values(df, level, mapping, rows=None):
    """Replace one or several category-values at a specific level (for specific rows)"""
    index = df if isinstance(df, pd.Index) else df.index

    n = index._get_level_number(level)

    # if replacing level values with a filter (by rows)
    if rows is not None and not all(rows):
        _levels = pd.Series(index.get_level_values(n))
        renamed_index = replace_index_values(index[rows], level, mapping)
        _levels[rows] = list(renamed_index.get_level_values(n))
        _unique_levels = pd.Index(_levels.unique())

        return append_index_level(
            index=index.droplevel(n),
            codes=_unique_levels.get_indexer(_levels),
            level=_unique_levels,
            name=level,
            order=index.names,
        )

    # else, replace the level values for the entire index dimension
    _levels = index.levels[n].map(lambda l: mapping.get(l, l))
    _unique_levels = _levels.unique()

    # if no duplicate levels exist after replace, set new levels and return
    if len(index.levels[n]) == len(_unique_levels):
        return index.set_levels(_levels, n)

    # if duplicate levels exist, re-map the codes
    level_mapping = _unique_levels.get_indexer(_levels)
    _codes = np.where(index.codes[n] != -1, level_mapping[index.codes[n]], -1)
    return index.set_codes(_codes, level=n).set_levels(_unique_levels, level=n)


def append_index_col(index, values, name, order=False):
    """Append a list of `values` as a new column (level) to an `index`"""
    levels = pd.Index(values).unique()
    codes = levels.get_indexer(values)

    return append_index_level(index, codes, levels, name, order)


def append_index_level(index, codes, level, name, order=False):
    """Append a level to a pd.MultiIndex"""
    if isstr(level):
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

        _raise_data_error(
            "Timeseries data has overlapping values", overlap.to_frame(index=False)
        )
