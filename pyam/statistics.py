from copy import deepcopy
import numpy as np
import pandas as pd
from pyam import filter_by_meta, META_IDX
from pyam.utils import isstr, islistable


class Statistics(object):
    """This class generates descriptive statistics of timeseries data

    Parameters
    ----------
    df : IamDataFrame
        an IamDataFrame from which to retrieve meta indicators for grouping
        or filtering
    groupby : str or dict
        a column of `df.meta` to be used for `groupby`
        or a dictionary of `{column: list}`, where `list` is used for ordering
    filters : list of tuples
        arguments for filtering and describing, either `((index, dict)` or
        `((index[0], index[1]), dict)`; when also using `groupby`, index must
        have length 2.
    percentiles : list-like of numbers, optional
        The percentiles to get from :meth:`pandas.DataFrame.describe()`.
        All should fall between 0 and 1. The default is `[.25, .5, .75]`,
        which returns the 25th, 50th, and 75th percentiles.
    """

    def __init__(
        self, df, groupby=None, filters=None, rows=False, percentiles=[0.25, 0.5, 0.75]
    ):
        self.df = df
        self.idx_depth = None

        # assing `groupby` settings and check that specifications are valid
        self.col = None
        self.groupby = None
        if isstr(groupby):
            self.col = groupby
            self.groupby = {groupby: None}
        elif isinstance(groupby, dict) and len(groupby) == 1:
            self.col = list(groupby.keys())[0]
            self.groupby = groupby
            self.idx_depth = 2
        elif groupby is not None:
            raise ValueError("arg `{}` not valid `groupby`".format(groupby))
        if self.col is not None and self.col not in df.meta.columns:
            raise ValueError("column `{}` not in `df.meta`".format(self.col))

        # if neither groupby nor filters is given, use filters to describe all
        # and assume that rows are used
        if groupby is None and filters is None:
            self.filters = [("", {})]
            rows = True
        else:
            self.filters = filters if filters is not None else []

        # set lists to sort index and subindex
        self._idx = [] if self.col is None else [self.col]
        self._sub_idx = (
            self.groupby[self.col] or self.df[self.col].unique()
            if self.col is not None
            else []
        )
        self._headers, self._subheaders = ([], [])

        # assing `filters` settings and check that specifications are valid
        for (idx, _filter) in self.filters:
            # check that index in tuple is valid
            if isstr(idx):
                self._add_to_index(idx)
            else:
                if not (
                    isinstance(idx, tuple)
                    and len(idx) == 2
                    and isstr(idx[0])
                    or not isstr(idx[1])
                ):
                    raise ValueError("`{}` is not a valid index".format(idx))
                self._add_to_index(idx[0], idx[1])
            # check that filters in tuple are valid
            if not isinstance(_filter, dict):
                raise ValueError("`{}` is not a valid filter".format(_filter))
            elif not (set(_filter) - set(META_IDX)).issubset(df.meta):
                raise ValueError(
                    "column `{}` not in `df.meta`".format(
                        set(_filter) - set(META_IDX) - set(df.meta)
                    )
                )

        self.stats = None
        self.rows = [] if rows else None

        # percentiles for passing to `pandas.describe()`
        self.percentiles = list(percentiles)
        self._describe_cols = (
            ["count", "mean", "std", "min"]
            + ["{:.0%}".format(i) for i in self.percentiles]
            + ["max"]
        )

    def _add_to_index(self, idx, sub_idx=None):
        # assign index depth if not set
        if self.idx_depth is None:
            self.idx_depth = 1 if sub_idx is None else 2
        # check that index matches depth
        if self.groupby is not None and sub_idx is None:
            msg = "if `groupby` is used, index `{}` must have format `{}`"
            raise ValueError(msg.format(idx, "(idx0, idx1)"))
        if self.idx_depth == 1 and sub_idx is not None:
            raise ValueError(
                "index depth set to 1, found `({}, {})`".format(idx, sub_idx)
            )
        if self.idx_depth == 2 and sub_idx is None:
            raise ValueError("index depth set to 2, found `({})`".format(idx))

        # append to lists for sorting index
        if idx not in self._idx:
            self._idx.append(idx)
        if self.idx_depth == 2 and sub_idx not in self._sub_idx:
            self._sub_idx.append(sub_idx)

    def _add_to_header(self, header, subheader):
        if header not in self._headers:
            self._headers.append(header)
        if islistable(subheader):
            for s in subheader:
                if s not in self._subheaders:
                    self._subheaders.append(s)
        elif subheader not in self._subheaders:
            self._subheaders.append(subheader)

    def add(self, data, header, row=None, subheader=None):
        """Filter 'data' by arguments of this Statistics instance,

        Apply :meth:`pandas.DataFrame.describe()` and format the statistics

        Parameters
        ----------
        data : pandas.DataFrame or pandas.Series
            data for which summary statistics should be computed
        header : str
            column name for descriptive statistics
        row : str
            row name for descriptive statistics
            (required if :class:`Statistics(rows=True) <Statistics>`)
        subheader : str, optional
            column name (level=1) if data is a unnamed :class:`pandas.Series`
        """
        # verify validity of specifications
        if self.rows is not None and row is None:
            raise ValueError("row specification required")
        if self.rows is None and row is not None:
            raise ValueError("row arg illegal for this `Statistics` instance")
        if isinstance(data, pd.Series):
            if subheader is not None:
                data.name = subheader
            elif data.name is None:
                msg = "`data` must be named `pd.Series` or provide `subheader`"
                raise ValueError(msg)
            data = pd.DataFrame(data)

        if self.rows is not None and row not in self.rows:
            self.rows.append(row)

        _stats = None

        # describe with groupby feature
        if self.groupby is not None:
            filter_args = dict(data=data, df=self.df, join_meta=True)
            filter_args.update(self.groupby)
            _stats = (
                filter_by_meta(**filter_args)
                .groupby(self.col)
                .describe(percentiles=self.percentiles)
            )
            _stats = pd.concat([_stats], keys=[self.col], names=[""], axis=0)
            if self.rows:
                _stats["row"] = row
                _stats.set_index("row", append=True, inplace=True)
            _stats.index.names = [""] * 3 if self.rows else [""] * 2

        # describe with filter feature
        for (idx, _filter) in self.filters:
            filter_args = dict(data=data, df=self.df)
            filter_args.update(_filter)
            _stats_f = filter_by_meta(**filter_args).describe(
                percentiles=self.percentiles
            )
            _stats_f = pd.DataFrame(_stats_f.unstack()).T
            if self.idx_depth == 1:
                levels = [[idx]]
            else:
                levels = [[idx[0]], [idx[1]]]
            lvls, lbls = (
                (levels, [[0]] * self.idx_depth)
                if not self.rows
                else (levels + [[row]], [[0]] * (self.idx_depth + 1))
            )
            _stats_f.index = pd.MultiIndex(levels=lvls, codes=lbls)
            _stats = _stats_f if _stats is None else _stats.append(_stats_f)

        # add header
        _stats = pd.concat([_stats], keys=[header], names=[""], axis=1)
        subheader = _stats.columns.get_level_values(1).unique()
        self._add_to_header(header, subheader)

        # set statistics
        if self.stats is None:
            self.stats = _stats
        else:
            self.stats = _stats.combine_first(self.stats)

    def reindex(self, copy=True):
        """Reindex the summary statistics dataframe"""
        ret = deepcopy(self) if copy else self

        ret.stats = ret.stats.reindex(index=ret._idx, level=0)
        if ret.idx_depth == 2:
            ret.stats = ret.stats.reindex(index=ret._sub_idx, level=1)
        if ret.rows is not None:
            ret.stats = ret.stats.reindex(index=ret.rows, level=ret.idx_depth)

        ret.stats = ret.stats.reindex(columns=ret._headers, level=0)
        ret.stats = ret.stats.reindex(columns=ret._subheaders, level=1)
        ret.stats = ret.stats.reindex(columns=ret._describe_cols, level=2)

        if copy:
            return ret

    def summarize(
        self, center="mean", fullrange=None, interquartile=None, custom_format="{:.2f}"
    ):
        """Format the compiled statistics to a concise string output

        Parameters
        ----------
        center : str, default `mean`
            what to return as 'center' of the summary: `mean`, `50%`, `median`
        fullrange : bool, default None
            return full range of data if True or `fullrange`, `interquartile`
            and `format_spec` are None
        interquartile : bool, default None
            return interquartile range if True
        custom_format : formatting specifications
        """
        # call `reindex()` to reorder index and columns
        self.reindex(copy=False)

        center = "median" if center == "50%" else center
        if fullrange is None and interquartile is None:
            fullrange = True
        return self.stats.apply(
            format_rows,
            center=center,
            fullrange=fullrange,
            interquartile=interquartile,
            custom_format=custom_format,
            axis=1,
            raw=False,
        )


# %% auxiliary functions


def format_rows(
    row, center, fullrange=None, interquartile=None, custom_format="{:.2f}"
):
    """Format a row with `describe()` columns to a concise string"""
    if (fullrange or 0) + (interquartile or 0) == 1:
        legend = "{} ({})".format(
            center, "max, min" if fullrange is True else "interquartile range"
        )
        index = row.index.droplevel(2).drop_duplicates()
        count_arg = dict(tuples=[("count", "")], names=[None, legend])
    else:
        msg = "displaying multiple range formats simultaneously not supported"
        raise NotImplementedError(msg)

    ret = pd.Series(
        index=pd.MultiIndex.from_tuples(**count_arg).append(index), dtype=float
    )

    row = row.sort_index()
    center = "50%" if center == "median" else center

    # get maximum of `count` and write to first entry of return series
    count = max(
        [i for i in row.loc[(slice(None), slice(None), "count")] if not np.isnan(i)]
    )
    ret.loc[("count", "")] = ("{:.0f}".format(count)) if count > 1 else ""

    # set upper and lower for the range
    upper, lower = ("max", "min") if fullrange is True else ("75%", "25%")

    # format `describe()` columns to string output
    for i in index:
        x = row.loc[i]
        _count = x["count"]
        if np.isnan(_count) or _count == 0:
            s = ""
        elif _count > 1:
            s = "{f} ({f}, {f})".format(f=custom_format).format(
                x[center], x[upper], x[lower]
            )
        elif _count == 1:
            s = "{f}".format(f=custom_format).format(x["50%"])
        # add count of this section as `[]` if different from count_max
        if 0 < _count < count:
            s += " [{:.0f}]".format(_count)
        ret.loc[i] = s

    return ret
