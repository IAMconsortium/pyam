import copy
import os
import six
import warnings
import itertools

import numpy as np
import pandas as pd

try:
    import ixmp
    has_ix = True
except ImportError:
    has_ix = False

from pyam_analysis.utils import (
    logger,
    write_sheet,
    read_ix,
    read_file,
    format_data,
    pattern_match,
    years_match,
    META_IDX,
    IAMC_IDX
)
from pyam_analysis.timeseries import fill_series
from pyam_analysis import plotting


class IamDataFrame(object):
    """This class is a wrapper for dataframes following the IAMC data convention.
    It provides a number of diagnostic features (including validation of values,
    completeness of variables provided) as well as a number of visualization and
    plotting tools.
    """

    def __init__(self, data, **kwargs):
        """Initialize an instance of an IamDataFrame

        Parameters
        ----------
        data: ixmp.TimeSeries, ixmp.Scenario, pandas.DataFrame or data file
            an instance of an TimeSeries or Scenario (requires `ixmp`),
            or pandas.DataFrame or data file with IAMC-format data columns
        """
        # import data from pandas.DataFrame or read from source
        if isinstance(data, pd.DataFrame):
            self.data = format_data(data)
        elif has_ix and isinstance(data, ixmp.TimeSeries):
            self.data = read_ix(data, **kwargs)
        else:
            self.data = read_file(data, **kwargs)

        # define a dataframe for categorization and other meta-data
        self.meta = self.data[META_IDX].drop_duplicates().set_index(META_IDX)
        self.reset_exclude()

    def __getitem__(self, key):
        _key_check = [key] if isinstance(key, six.string_types) else key
        if set(_key_check).issubset(self.meta.columns):
            return self.meta.__getitem__(key)
        else:
            return self.data.__getitem__(key)

    def __len__(self):
        return self.data.__len__()

    def head(self, *args, **kwargs):
        """Identical to pd.DataFrame.head() operating on data"""
        return self.data.head(*args, **kwargs)

    def append(self, other, inplace=False, **kwargs):
        """Import or read timeseries data and append to IamDataFrame

        Parameters
        ----------
        other: pyam-analysis.IamDataFrame, ixmp.TimeSeries, ixmp.Scenario,
        pandas.DataFrame or data file
            an IamDataFrame, TimeSeries or Scenario (requires `ixmp`),
            or pandas.DataFrame or data file with IAMC-format data columns
        inplace : bool, default False
            if True, do operation inplace and return None
        """
        ret = copy.deepcopy(self) if not inplace else self

        if not isinstance(other, IamDataFrame):
            other = IamDataFrame(other)

        # check that any model/scenario is not yet included in IamDataFrame
        ret.meta = ret.meta.append(other.meta, verify_integrity=True)

        # add new data
        ret.data = ret.data.append(other.data).reset_index(drop=True)

        if not inplace:
            return ret

    def pivot_table(self, index, columns, values='value',
                    aggfunc='count', fill_value=None, style=None):
        """Returns a pivot table

        Parameters
        ----------
        index: str or list of strings
            rows for Pivot table
        columns: str or list of strings
            columns for Pivot table
        values: str, default 'value'
            dataframe column to aggregate or count
        exclude_cat: None or list of strings, default ['exclude']
            exclude all scenarios from the listed categories
        aggfunc: str or function, default 'count'
            function used for aggregation,
            accepts 'count', 'mean', and 'sum'
        fill_value: scalar, default None
            value to replace missing values with
        style: str, default None
            output style for pivot table formatting
            accepts 'highlight_not_max', 'heatmap'
        """
        index = [index] if isinstance(index, six.string_types) else index
        columns = [columns] if isinstance(
            columns, six.string_types) else columns

        df = self.data

        # allow 'aggfunc' to be passed as string for easier user interface
        if isinstance(aggfunc, six.string_types):
            if aggfunc == 'count':
                df = self.data.groupby(index + columns, as_index=False).count()
                fill_value = 0
            elif aggfunc == 'mean':
                df = self.data.groupby(index + columns, as_index=False).mean()\
                    .round(2)
                aggfunc = np.sum
                fill_value = 0 if style == 'heatmap' else ""
            elif aggfunc == 'sum':
                aggfunc = np.sum
                fill_value = 0 if style == 'heatmap' else ""

        df = df.pivot_table(values=values, index=index, columns=columns,
                            aggfunc=aggfunc, fill_value=fill_value)
        return df

    def interpolate(self, year):
        """Interpolate missing values in timeseries (linear interpolation)

        Parameters
        ----------
        year: int
             year to be interpolated
        exclude_cat: None or list of strings, default ['exclude']
             exclude all scenarios from the listed categories
        """
        df = self.pivot_table(index=IAMC_IDX, columns=['year'],
                              values='value', aggfunc=np.sum)
        fill_values = df.apply(fill_series,
                               raw=False, axis=1, year=year)
        fill_values = fill_values.dropna().reset_index()
        fill_values = fill_values.rename(columns={0: "value"})
        fill_values['year'] = year
        self.data = self.data.append(fill_values)

    def as_pandas(self, with_metadata=False):
        """Return this as a pd.DataFrame

        Parameters
        ----------
        with_metadata : bool, default False
           if True, join data with existing metadata
        """
        df = self.data
        if with_metadata:
            df = (df
                  .set_index(META_IDX)
                  .join(self.meta)
                  .reset_index()
                  )
        return df

    def timeseries(self):
        """Returns a dataframe in the standard IAMC format
        """
        return self.data.pivot_table(index=IAMC_IDX,
                                     columns='year')['value']

    def reset_exclude(self):
        """Reset exclusion assignment for all scenarios to 'uncategorized'
        """
        self.meta['exclude'] = False


    def metadata(self, meta, name=None):
        """Add metadata columns as pandas Series or DataFrame

        Parameters
        ----------
        meta: pandas.Series, list, int or str
            column to be added to metadata
            (by `['model', 'scenario']` index if possible)
        name: str
            category column name (if not given by data series name)
        """
        if isinstance(meta, pd.Series) and \
                meta.index.names == ['model', 'scenario']:
            diff = meta.index.difference(self.meta.index)
            if not diff.empty:
                error = "adding metadata for non-existing scenarios '{}'!"
                raise ValueError(error.format(diff))
            meta = meta.to_frame(meta.name or name)
            self.meta = meta.combine_first(self.meta)
            return  # EXIT FUNCTION

        if isinstance(meta, pd.Series):
            name = meta.name or name
            meta = meta.tolist()

        self.meta[name] = meta

    def categorize(self, name, value, criteria,
                   color=None, marker=None, linestyle=None):
        """Assign scenarios to a category according to specific criteria
        or display the category assignment

        Parameters
        ----------
        name: str
            category column name
        value: str
            category identifier
        criteria: dict
            dictionary with variables mapped to applicable checks
            ('up' and 'lo' for respective bounds, 'year' for years - optional)
        color: str
            assign a color to this category for plotting
        marker: str
            assign a marker to this category for plotting
        linestyle: str
            assign a linestyle to this category for plotting
        """
        # add plotting run control
        for kind, arg in [('color', color), ('marker', marker),
                          ('linestyle', linestyle)]:
            if arg:
                plotting.run_control().update({kind: {name: {value: arg}}})

        if criteria == 'uncategorized':
            self.meta[name].fillna(value, inplace=True)
            msg = "{} of {} scenarios are uncategorized."
            logger().info(msg.format(np.sum(self.meta[name] == value),
                                     len(self.meta)))
            return  # EXIT FUNCTION

        # find all data that matches categorization
        idx = _meta_idx(_apply_criteria(self.data, criteria, in_range=True))

        # update metadata dataframe
        if len(idx) == 0:
            if name not in self.meta:
                self.meta[name] = None
            logger().info("No scenarios satisfy the criteria")
        else:
            self.meta.loc[idx, name] = value
            msg = "{} scenario(s) categorized as {} '{}'"
            logger().info(msg.format(len(idx), name, value))

    def validate(self, criteria={}, exclude=False):
        """Check which model/scenarios satisfy specific criteria

        Parameters
        ----------
        criteria: dict
           dictionary with variable keys and check values
            ('up' and 'lo' for respective bounds, 'year' for years - optional)
        exclude: bool, default False
            if true, exclude models and scenarios failing validation from further analysis
        """
        df = _apply_criteria(self.data, criteria, in_range=False)

        if exclude:
            idx = _meta_idx(df)
            self.meta.loc[idx, 'exclude'] = True

        if not df.empty:
            msg = '{} of {} data points to not satisfy the criteria'
            logger().info(msg.format(len(df), len(self.data)))

            if exclude and len(idx) > 0:
                logger().info('Non-valid scenarios will be excluded')

            return df

    def filter(self, filters, inplace=False):
        """Return a filtered IamDataFrame (i.e., a subset of current data)

        Parameters
        ----------
        filters: dict
            The following columns are available for filtering:
             - metadata columns: filter by category assignment in metadata
             - 'model', 'scenario', 'region': takes a string or list of strings
             - 'variable': takes a string or list of strings,
                where ``*`` can be used as a wildcard
             - 'level': the maximum "depth" of IAM variables (number of '|')
               (exluding the strings given in the 'variable' argument)
             - 'year': takes an integer, a list of integers or a range
                note that the last year of a range is not included,
                so ``range(2010,2015)`` is interpreted as ``[2010, ..., 2014]``
        inplace : bool, default False
            if True, do operation inplace and return None
        """
        keep = _apply_filters(self.data, self.meta, filters)
        ret = copy.deepcopy(self) if not inplace else self
        ret.data = ret.data[keep]

        idx = pd.MultiIndex.from_tuples(
            pd.unique(list(zip(ret.data['model'], ret.data['scenario']))),
            names=('model', 'scenario')
        )
        ret.meta = ret.meta.loc[idx]
        if not inplace:
            return ret

    def to_excel(self, excel_writer, sheet_name='data', index=False, **kwargs):
        """Write timeseries data to Excel using the IAMC template convention
        (wrapper for `pandas.DataFrame.to_excel()`)

        Parameters
        ----------
        excel_writer: string or ExcelWriter object
             file path or existing ExcelWriter
        sheet_name: string, default 'data'
            name of the sheet that will contain the (filtered) IamDataFrame
        index: boolean, default False
            write row names (index)
        """
        df = self.timeseries().reset_index()
        df = df.rename(columns={c: str(c).title() for c in df.columns})
        df.to_excel(excel_writer, sheet_name=sheet_name, index=index, **kwargs)

    def export_metadata(self, path):
        """Export metadata to Excel

        Parameters
        ----------
        path:
            path/filename for xlsx file of metadata export
        """
        writer = pd.ExcelWriter(path)
        write_sheet(writer, 'metadata', self.meta, index=True)
        writer.save()

    def load_metadata(self, path, *args, **kwargs):
        """Load metadata from previously exported instance of pyam_analysis

        Parameters
        ----------
        path:
            xlsx file with metadata exported from an instance of pyam_analysis
        """

        if not os.path.exists(path):
            raise ValueError("no metadata file '" + path + "' found!")

        if path.endswith('csv'):
            df = pd.read_csv(path, *args, **kwargs)
        else:
            df = pd.read_excel(path, *args, **kwargs)

        req_cols = ['model', 'scenario', 'exclude']
        if not set(req_cols).issubset(set(df.columns)):
            e = "metadata file '{}' does not have required columns ({})!"
            raise ValueError(e.format(path, req_cols))

        df.set_index(META_IDX, inplace=True)
        self.meta = df.combine_first(self.meta)

    def line_plot(self, *args, **kwargs):
        """Plot timeseries lines of existing data

        see pyam_analysis.plotting.line_plot() for all available options
        """
        df = self.as_pandas(with_metadata=True)
        ax, handles, labels = plotting.line_plot(df, *args, **kwargs)
        return ax


def _meta_idx(data):
    return data[META_IDX].set_index(META_IDX).index


def _apply_filters(data, meta, filters):
    keep = np.array([True] * len(data))

    # filter by columns and list of values
    for col, values in filters.items():
        if col in meta.columns:
            matches = pattern_match(meta[col], values)
            cat_idx = meta[matches].index
            keep_col = data[META_IDX].set_index(META_IDX).index.isin(cat_idx)

        elif col in ['model', 'scenario', 'region']:
            keep_col = pattern_match(data[col], values)

        elif col == 'variable':
            level = filters['level'] if 'level' in filters.keys() else None
            keep_col = pattern_match(data[col], values, True, level)

        elif col in ['year']:
            keep_col = years_match(data[col], values)

        elif col in ['level']:
            if 'variable' not in filters.keys():
                keep_col = pattern_match(data['variable'], '*',
                                         pseudo_regex=True, level=values)
            else:
                continue
        else:
            raise SystemError(
                'filter by column ' + col + ' not supported')
        keep = keep & keep_col
    return keep


def _apply_criteria(df, criteria, in_range=True):
    idxs = []
    for var, check in criteria.items():
        fail_idx = []
        where_idx = []
        _df = df[df['variable'] == var]
        where_idx.append(set(_df.index))
        up_op = _df['value'].__le__ if in_range else _df['value'].__gt__
        lo_op = _df['value'].__ge__ if in_range else _df['value'].__lt__
        for check_type, val in check.items():
            if check_type == 'up':
                fail_idx.append(set(_df.index[up_op(val)]))
            elif check_type == 'lo':
                fail_idx.append(set(_df.index[lo_op(val)]))
            elif check_type == 'year':
                where_idx.append(set(_df.index[_df['year'] == val]))
            else:
                raise ValueError(
                    "Unknown checking type: {}".format(check_type))
        where_idx = set.intersection(*where_idx)
        fail_idx = set.intersection(*fail_idx)
        idxs.append(where_idx & fail_idx)

    df = df.loc[itertools.chain(*idxs)]

    return df


def validate(df, *args, **kwargs):
    """Run validation checks on timeseries data

    Parameters
    ----------
    df: IamDataFrame instance
    args and kwargs: see IamDataFrame.validate() for details
    filters: dict, optional
        filter by model, scenario, region, variable, level, year, category
        see function 'filter()' for details
        filter by 'variable'/'year' is replaced by arguments of 'criteria'
        see function IamDataFrame.filter() for details
    """
    filters = kwargs.pop('filters', {})
    fdf = df.filter(filters)
    vdf = fdf.validate(*args, **kwargs)
    df.meta['exclude'] |= fdf.meta['exclude']  # update if any excluded
    return vdf


def categorize(df, *args, **kwargs):
    """Run validation checks on timeseries data

    Parameters
    ----------
    df: IamDataFrame instance
    args and kwargs: see IamDataFrame.categorize() for details
    filters: dict, optional
        filter by model, scenario, region, variable, level, year, category
        see function 'filter()' for details
        filter by 'variable'/'year' is replaced by arguments of 'criteria'
        see function IamDataFrame.filter() for details
    """
    filters = kwargs.pop('filters', {})
    fdf = df.filter(filters)
    fdf.categorize(*args, **kwargs)

    # update metadata
    name = args[0]
    if name in df.meta:
        df.meta[name].update(fdf.meta[name])
    else:
        df.meta[name] = fdf.meta[name]
