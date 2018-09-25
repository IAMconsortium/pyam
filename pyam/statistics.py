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
        then apply `pd.describe()` and format the statistics"""
        if isinstance(data, pd.Series):
            data.name = subheader or ''
            data = pd.DataFrame(data)

        if self.groupby is not None:
            args = dict(data=data, df=self.df, **self.groupby, join_meta=True)
            stats = filter_by_meta(**args).groupby('category').describe()
            stats['name'] = self.col
            stats = stats.set_index('name', append=True).swaplevel()
            stats.index.names = ['', '']
            # order rows by groupby-columns if available
            if self.groupby[self.col] is not None:
                stats = stats.reindex(index=self.groupby[self.col], level=1)

        return stats

    def _append_stats(self, other, name, header):
        if name not in self.rows:
            self.rows.append(name)
        if header not in self.headers:
            self.headers.append(header)

        if self.stats is None:
            self.stats = other
        else:
            self.stats = (
                    other.combine_first(self.stats)
                    .reindex(columns=self.headers, level=0)
                    .reindex(columns=describe_cols, level=2)
                    )

        # reorder groupby-for first (and second) index level (if necessary)
        if self.groupby is None:
            self.stats = self.stats.reindex(index=self.rows)
        else:
            self.stats = self.stats.reindex(index=self.rows, level=0)\
                .reindex(index=self.subrows, level=1)


# %% auxiliary functions


def format_rows(row, upper='max', lower='min'):
    """Format a row with `describe()` columns to a concise string"""
    index = row.index.droplevel(2).drop_duplicates()
    ret = pd.Series(index=pd.MultiIndex.from_tuples(tuples=[('count', '')],
                                                    names=['type', None])
                    .append(index))

    # get maximum of `count` and write to first entry of return series
    count = max(row.loc[(slice(None), slice(None), 'count')])
    ret.loc[('count', '')] = ('{:.0f}'.format(count)) if count > 1 else ''

    for i in index:
        x = row.loc[(i[0], i[1])]
        _count = x['count']
        if np.isnan(_count) or _count == 0:
            s = 'NA'
        elif _count > 1:
            s = '{:.2f} ({:.2f}, {:.2f})'.format(x['50%'], x[upper], x[lower])
        elif _count == 1:
            s = '{:.2f}'.format(x['50%'])

        # add count of this section in `[]` if different from count_max
        if _count < count:
            s += ' [{:.0f}]'.format(_count)

        ret.loc[(i[0], i[1])] = s

    return ret
