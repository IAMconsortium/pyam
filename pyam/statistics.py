#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import utils

describe_cols = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']


class Statistics(object):
    """This class provides a wrapper.
    """
    def __init__(self, df, groupby=None, **kwargs):
        self.df = df
        self.groupby = groupby
        self.filters = kwargs

        self.stats = None
        self.rows = []
        self.subrows = self.filters[groupby] if groupby else None
        self.headers = []

    def describe(self, name, data, header, subheader=None):
        """Filter `data` by arguments of this SummaryStats instance,
        then apply `pd.describe()` and format the output"""

        if isinstance(data, pd.Series):
            data.name = subheader or ''
            data = pd.DataFrame(data)

        _data = utils.filter_by_meta(data, self.df, **self.filters)

        if self.groupby is not None:
            _data = _data.groupby(self.groupby)

        # generate statistics for `data` dataframe and append to self.stats
        stats = describe(_data, row=name, col=header)
        self._append_stats(stats, name, header)

    def add_by_row(self, name, data, header, subheader=None, **kwargs):
        """Add data by row to the summary statistics"""
        if len(kwargs) > 1:
            msg = 'Filtering by more than one kwargs not supported!'
            raise NotImplementedError(msg)

        if isinstance(data, pd.Series):
            data.name = subheader or ''
            data = pd.DataFrame(data)

        data = utils.filter_by_meta(data, self.df, **kwargs)

        data['name'] = name
        data['type'] = header
        data[''] = '50%'

        cat = list(kwargs.keys())[0]

        data = (
                data
                .reset_index(drop=True)
                .set_index(['name', cat, '', 'type'])
                .reindex(index=kwargs[cat], level=1)
                .unstack().unstack()
                )
        data.columns = data.columns.swaplevel(i=0, j=1)
        self._append_stats(data, name, header)

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

    def format_as_table(self, upper='max', lower='min'):
        """Format the compiled statistics to a concise string output"""
        return self.stats.apply(format_rows, upper=upper, lower=lower,
                                axis=1, raw=False)


# %% auxiliary functions

def describe(x, row, col):
    """Wrapper for `pd.describe()` to get output in standardized format"""

    stats = x.describe()

    stats['name'] = row
    stats = stats.set_index('name', append=True).swaplevel()

    if isinstance(x, pd.DataFrame):
        cols = stats.index.get_level_values(1)
        stats = stats.unstack().reindex(columns=cols, level=1)

    stats['type'] = col
    stats.set_index(['type'], append=True, inplace=True)
    stats = stats.unstack()
    stats.columns = stats.columns.swaplevel().swaplevel(i=0, j=1)

    return stats


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
