#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from pyam import filter_by_meta
from pyam.utils import isstr

describe_cols = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']


class Statistics(object):
    """This class provides a wrapper for descriptive statistics
    of IAMC-style timeseries data.

    Parameters
    ----------
    df: pyam.IamDataFrame
        an IamDataFrame from which to retrieve metadata for grouping, filtering
    groupby: str or dict
        a column of `df.meta` to be used for the groupby feature,
        or a dictionary of `{column: list}`, where `list` is used for ordering
    """
    def __init__(self, df, groupby=None):
        self.df = df
        # check that specifications for the `groupby` feature are valid
        self.col = None
        self.groupby = None
        if isstr(groupby):
            self.col = groupby
            self.groupby = {groupby: None}
        elif isinstance(groupby, dict) and len(groupby) == 1:
            self.col = list(groupby.keys())[0]
            self.groupby = groupby
        elif groupby is not None:
            raise ValueError('arg `{}` not valid `groupby`'.format(groupby))
        if self.col is not None and self.col not in df.meta.columns:
            raise ValueError('column `{}` not in `df.meta`'.format(self.col))

        self.stats = None
        self.rows = []
        self.headers = []

    def describe(self, data, header, subheader=None):
        """Filter `data` by arguments of this SummaryStats instance,
        then apply `pd.describe()` and format the statistics

        Parameters
        ----------
        data : pd.DataFrame or pd.Series
            data for which summary statistics should be computed
        header : str
            column name for descriptive statistics
        subheader : str, optional
            column name (level=1) if data is a unnamed `pd.Series`
        """
        if isinstance(data, pd.Series) and data.name is None:
            data.name = subheader or ''
        data = pd.DataFrame(data)

        if self.groupby is not None:
            args = dict(data=data, df=self.df, **self.groupby, join_meta=True)
            _stats = filter_by_meta(**args).groupby('category').describe()
            _stats['name'] = self.col
            _stats = _stats.set_index('name', append=True).swaplevel()
            _stats.index.names = ['', '']
            # order rows by groupby-columns if available
            if self.groupby[self.col] is not None:
                _stats = _stats.reindex(index=self.groupby[self.col], level=1)
            _stats = pd.concat([_stats], keys=[header], names=[''], axis=1)

        if self.stats is None:
            self.stats = _stats
        else:
            # replace or join statistics
            intersect = self.stats.columns.intersection(_stats.columns)
            self.stats.loc[:, intersect] = _stats.loc[:, intersect]
            diff = _stats.columns.difference(self.stats.columns)
            self.stats = self.stats.join(_stats.loc[:, diff])

    def summarize(self, interquartile=False):
        """Format the compiled statistics to a concise string output

        Parameter
        ---------
        interquartile : bool, default False
            return interquartile range if True, else max-min
        """
        upper = '75%' if interquartile else 'max'
        lower = '25%' if interquartile else 'min'
        return self.stats.apply(format_rows, upper=upper, lower=lower,
                                axis=1, raw=False)

# %% auxiliary functions


def format_rows(row, upper='max', lower='min'):
    """Format a row with `describe()` columns to a concise string"""
    index = row.index.droplevel(2).drop_duplicates()
    count_arg = dict(tuples=[('count', '')], names=[None, None])
    ret = pd.Series(index=pd.MultiIndex.from_tuples(**count_arg).append(index))

    row = row.sort_index()

    # get maximum of `count` and write to first entry of return series
    count = max(row.loc[(slice(None), slice(None), 'count')])
    ret.loc[('count', '')] = ('{:.0f}'.format(count)) if count > 1 else ''

    # format `describe()` columns to string output
    for i in index:
        x = row.loc[i]
        _count = x['count']
        if np.isnan(_count) or _count == 0:
            s = 'NA'
        elif _count > 1:
            s = '{:.2f} ({:.2f}, {:.2f})'.format(x['50%'], x[upper], x[lower])
        elif _count == 1:
            s = '{:.2f}'.format(x['50%'])
        # add count of this section as `[]` if different from count_max
        if _count < count:
            s += ' [{:.0f}]'.format(_count)
        ret.loc[i] = s

    return ret
