# -*- coding: utf-8 -*-
"""
Initial version based on
https://github.com/iiasa/ceds_harmonization_analysis by Matt Gidden
"""

import copy
import os
import six
import warnings

import re
import numpy as np
import pandas as pd

import pyam_analysis as iam

from pyam_analysis.utils import logger

from pyam_analysis import plotting

try:
    import seaborn as sns
except ImportError:
    pass

try:
    import ixmp
    has_ix = True
except ImportError:
    has_ix = False


# %% treatment of warnings, formatting of Jupyter noteook output

# formatting for warnings
def custom_formatwarning(msg, category, filename, lineno, line=''):
    # ignore everything except the message
    return str(msg) + '\n'


# in Jupyter notebooks: disable autoscroll, activate warnings
try:
    get_ipython().run_cell_magic(u'javascript', u'',
                                 u'IPython.OutputArea.prototype._should_scroll = function(lines) { return false; }')
    warnings.simplefilter('default')
    warnings.formatwarning = custom_formatwarning
except Exception:
    pass


# %%  default settings for column headers

MIN_IDX = ['model', 'scenario']
IAMC_IDX = ['model', 'scenario', 'region', 'variable', 'unit']
MAX_IDX = IAMC_IDX + ['year']


# %% class for working with IAMC-style timeseries data


class IamDataFrame(object):
    """This class is a wrapper for dataframes
    following the IAMC data convention.
    It provides a number of diagnostic features
    (including validation of values, completeness of variables provided)
    as well as a number of visualization and plotting tools."""

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
        self.meta = pd.DataFrame(return_index(self.data, MIN_IDX,
                                              drop_duplicates=True))
        self.reset_exclude()

    def __getitem__(self, key):
        _key_check = [key] if isinstance(key, six.string_types) else key
        if set(_key_check).issubset(self.meta.columns):
            return self.meta.__getitem__(key)
        else:
            return self.data.__getitem__(key)

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

    def export_metadata(self, path):
        """Export metadata to Excel

        Parameters
        ----------
        path:
            path/filename for xlsx file of metadata export
        """
        writer = pd.ExcelWriter(path)
        iam.utils.write_sheet(writer, 'metadata', self.meta)
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

        df.set_index(MIN_IDX, inplace=True)
        self.meta = df.combine_first(self.meta)

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

    def timeseries(self):
        """Returns a dataframe in the standard IAMC format
        """
        return self.data.pivot_table(index=IAMC_IDX,
                                     columns='year')['value']

    def validate(self, criteria={}, filters={},
                 exclude=False, silent=False):
        """Run validation checks on timeseries data

        Parameters
        ----------
        criteria: dict, optional
            dictionary of variables mapped to a dictionary of checks
            ('up' and 'lo' for respective bounds, 'year' for years - optional)
        filters: dict, optional
            filter by model, scenario, region, variable, level, year, category
            see function 'filter()' for details
            filter by 'variable'/'year' is replaced by arguments of 'criteria'
            see function _check() for details
        exclude_cat: list of strings, default ['exclude']
            exclude all scenarios from the listed categories from validation
        exclude: bool, default False
            models/scenarios failing the validation to be excluded from data
        silent: bool, default False
            if False, print a summary statement of validation
        """
        # get filtered data and meta-data
        df = self.data[self._filter_columns(filters)]
        idx = pd.MultiIndex.from_tuples(
            pd.unique(list(zip(df['model'], df['scenario']))),
            names=('model', 'scenario')
        )
        meta = self.meta.loc[idx].index
        count = len(meta)

        # if criteria is a string, check that each scenario has this variable
        if isinstance(criteria, str):
            data_filters = filters.copy()
            data_filters.update({'variable': criteria})
            idx = self.data[self._filter_columns(data_filters)]\
                .set_index(MIN_IDX).index

            df = pd.DataFrame(index=meta)
            df['keep'] = True
            if len(idx):
                df.loc[idx, 'keep'] = False
            df = df[df.keep].reset_index()[MIN_IDX]
        # else, loop over dictionary and perform checks
        else:
            df = pd.DataFrame()
            for var, check in criteria.items():
                df = df.append(self._check(var, check,
                                           filters, ret_true=False))
        if len(df):
            n = len(df)
            s = 'scenario' if count == 1 else 'scenarios'
            msg = '{} data points do not satisfy the criteria (out of {} {})'

            if exclude:
                idx = return_index(df, MIN_IDX)
                self.meta.loc[idx, 'exclude'] = True
                msg += ", categorized as 'exclude' in metadata"

            if not silent:
                logger().info(msg.format(n, count, s))

            return df

        elif not silent:
            logger().info('{} scenarios satisfy the criteria'.format(count))

    def categorize(self, name, value, criteria, filters={},
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
        filters: dict, optional
            filter by model, scenario, region, variable, level, year, category
            see function 'filter()' for details
            filter by 'variable'/'year' is replaced by arguments of 'criteria'
            see function _check() for details
        color: str
            assign a color to this category for plotting
        marker: str
            assign a marker to this category for plotting
        linestyle: str
            assign a linestyle to this category for plotting
        """
        # find all data that matches categorization
        cat_idx = self.meta.index
        for var, check in criteria.items():
            cat_idx = cat_idx.intersection(self._check(var, check,
                                                       filters).index)

        if len(cat_idx) == 0:
            logger().info("No scenarios satisfy the criteria")
            if name not in self.meta:
                self.meta[name] = None
            return

        # update metadata dataframe
        self.meta.loc[cat_idx, name] = value

        # add plotting run control
        for kind, arg in [('color', color), ('marker', marker),
                          ('linestyle', linestyle)]:
            if arg:
                plotting.run_control().update({kind: {name: {value: arg}}})

        n = len(cat_idx)
        s = 'scenario' if n == 1 else 'scenarios'
        logger().info("{} {} categorized as {} '{}'".format(n, s, name, value))

    def reset_exclude(self):
        """Reset exclusion assignment for all scenarios to 'uncategorized'
        """
        self.meta['exclude'] = False

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
        fill_values = df.apply(iam.fill_series, raw=False, axis=1, year=year)
        fill_values = fill_values.dropna().reset_index()
        fill_values = fill_values.rename(columns={0: "value"})
        fill_values['year'] = year
        self.data = self.data.append(fill_values)

    def _validate(self, criteria, passes=True):
        """Check which model/scenarios satisfy specific criteria

        Parameters
        ----------
        variable: str
            variable to be checked
        criteria: dict
            dictionary with variable keys and check values
            ('up' and 'lo' for respective bounds, 'year' for years - optional)
        filters: dict, optional
            filter by model, scenario, region, variable, level, year, category
            see function 'filter()' for details
            filter by 'variable'/'year' are replaced by arguments of 'check'
        passes: bool, default True
            if true, return models/scenarios passing the check;
            otherwise, return datatframe of all failed checks
        """
        is_true = np.array([True] * len(self.data))

        for check_type, val in check.items():
            if check_type == 'up':
                is_true = is_true & (
                    self.data.loc['variable', var].value <= val)

            if check_type == 'lo':
                is_true = is_true & (self.data.loc['variable', var] > val)

        if passes:
            # if assessing a criteria for one year only
            if ('year' in check) and isinstance(check['year'], int):
                return df.loc[is_true, ['model', 'scenario', 'year']]\
                    .drop_duplicates()\
                    .set_index(MIN_IDX)
            # if more than one year is filtered for, ensure that
            # the criteria are satisfied in every year
            else:
                num_yr = len(df.year.drop_duplicates())
                df_agg = df.loc[is_true, ['model', 'scenario', 'year']]\
                    .groupby(MIN_IDX).count()
                return pd.DataFrame(index=df_agg[df_agg.year == num_yr].index)
        else:
            return df[~is_true]

    def _filter_columns(self, filters):
        keep = np.array([True] * len(self.data))

        # filter by columns and list of values
        for col, values in filters.items():
            if col in self.meta.columns:
                matches = pattern_match(self.meta[col], values)
                cat_idx = self.meta[matches].index
                keep_col = return_index(self.data, MIN_IDX).isin(cat_idx)

            elif col in ['model', 'scenario', 'region']:
                keep_col = pattern_match(self.data[col], values)

            elif col == 'variable':
                level = filters['level'] if 'level' in filters.keys() else None
                keep_col = pattern_match(self.data[col], values, True, level)

            elif col in ['year']:
                keep_col = years_match(self.data[col], values)

            elif col in ['level']:
                if 'variable' not in filters.keys():
                    keep_col = pattern_match(self.data['variable'], '*',
                                             pseudo_regex=True, level=values)
                else:
                    continue
            else:
                raise SystemError(
                    'filter by column ' + col + ' not supported')
            keep = keep & keep_col
        return keep

    def _check(self, variable, check, filters={}, ret_true=True):
        """Check which model/scenarios satisfy specific criteria

        Parameters
        ----------
        variable: str
            variable to be checked
        check: dict
            dictionary with checks
            ('up' and 'lo' for respective bounds, 'year' for years - optional)
        filters: dict, optional
            filter by model, scenario, region, variable, level, year, category
            see function 'filter()' for details
            filter by 'variable'/'year' are replaced by arguments of 'check'
        ret_true: bool, default True
            if true, return models/scenarios passing the check;
            otherwise, return datatframe of all failed checks
        """
        if not filters:
            filters = {}
        if 'year' in check:
            filters['year'] = check['year']
        filters['variable'] = variable
        df = self.data[self._filter_columns(filters)]

        is_true = np.array([True] * len(df.value))

        for check_type, val in check.items():
            if check_type == 'up':
                is_true = is_true & (df.value <= val)

            if check_type == 'lo':
                is_true = is_true & (df.value > val)

        if ret_true:
            # if assessing a criteria for one year only
            if ('year' in check) and isinstance(check['year'], int):
                return df.loc[is_true, ['model', 'scenario', 'year']]\
                    .drop_duplicates()\
                    .set_index(MIN_IDX)
            # if more than one year is filtered for, ensure that
            # the criteria are satisfied in every year
            else:
                num_yr = len(df.year.drop_duplicates())
                df_agg = df.loc[is_true, ['model', 'scenario', 'year']]\
                    .groupby(MIN_IDX).count()
                return pd.DataFrame(index=df_agg[df_agg.year == num_yr].index)
        else:
            return df[~is_true]

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
        keep = self._filter_columns(filters)
        ret = copy.deepcopy(self) if not inplace else self
        ret.data = ret.data[keep]

        idx = pd.MultiIndex.from_tuples(
            pd.unique(list(zip(ret.data['model'], ret.data['scenario']))),
            names=('model', 'scenario')
        )
        ret.meta = ret.meta.loc[idx]
        if not inplace:
            return ret

    def head(self, *args, **kwargs):
        """Identical to pd.DataFrame.head() operating on data"""
        return self.data.head(*args, **kwargs)

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
                  .set_index(['model', 'scenario'])
                  .join(self.meta)
                  .reset_index()
                  )
        return df

    def line_plot(self, *args, **kwargs):
        """Plot timeseries lines of existing data

        see pyam_analysis.plotting.line_plot() for all available options
        """
        df = self.as_pandas(with_metadata=True)
        ax, handles, labels = plotting.line_plot(df, *args, **kwargs)
        return ax

# %% auxiliary function for reading data from snapshot file


def read_ix(ix, **kwargs):
    """Read timeseries data from an ix object

    Parameters
    ----------
    ix: ixmp.TimeSeries or ixmp.Scenario
        this option requires the ixmp package as a dependency
    regions: list
        list of regions to be loaded from the database snapshot
    """
    if isinstance(ix, ixmp.TimeSeries):
        df = ix.timeseries(iamc=False, **kwargs)
        df['model'] = ix.model
        df['scenario'] = ix.scenario
    else:
        error = 'arg ' + ix + ' not recognized as valid ixmp class'
        raise ValueError(error)

    return df


def read_file(fname, *args, **kwargs):
    """Read data from a snapshot file saved in the standard IAMC format
    or a table with year/value columns
    """
    if not os.path.exists(fname):
        raise ValueError("no data file '" + fname + "' found!")

    # read from database snapshot csv or xlsx
    if fname.endswith('csv'):
        df = pd.read_csv(fname, *args, **kwargs)
    else:
        df = pd.read_excel(fname, *args, **kwargs)

    return format_data(df)


def format_data(df):
    """Convert an imported dataframe and check all required columns"""

    # format columns to lower-case and check that all required columns exist
    df.rename(columns={c: str(c).lower() for c in df.columns}, inplace=True)
    if not set(IAMC_IDX).issubset(set(df.columns)):
        missing = list(set(IAMC_IDX) - set(df.columns))
        raise ValueError("missing required columns {}!".format(missing))

    # check whether data in IAMC style or year/value layout
    if 'value' not in df.columns:
        numcols = sorted(set(df.columns) - set(IAMC_IDX))
        df = pd.melt(df, id_vars=IAMC_IDX, var_name='year',
                     value_vars=numcols, value_name='value')
    df.year = pd.to_numeric(df.year)

    # drop NaN's
    df.dropna(inplace=True)

    return df


# %% auxiliary functions for data filtering

def style_df(df, style='heatmap'):
    if style == 'highlight_not_max':
        return df.style.apply(lambda s: ['' if v else 'background-color: yellow' for v in s == s.max()])
    if style == 'heatmap':
        cm = sns.light_palette("green", as_cmap=True)
        return df.style.background_gradient(cmap=cm)


def return_index(df, idx_cols, drop_duplicates=False):
    """set and return an index for a dataframe"""
    if drop_duplicates:
        return df[idx_cols].drop_duplicates().set_index(idx_cols)
    else:
        return df[idx_cols].set_index(idx_cols).index


def pattern_match(data, strings, pseudo_regex=False, level=None):
    """
    matching of model/scenario names, variables, regions, and categories
    to pseudo-regex (optional) for data filtering
    """
    matches = np.array([False] * len(data))

    if isinstance(strings, six.string_types):
        strings = [strings]

    for s in strings:
        regexp = str(s)
        if pseudo_regex:
            regexp = regexp.replace('|', '\\|').replace('*', '.*') + "$"
        pattern = re.compile(regexp)
        subset = filter(pattern.match, data)
        # check for depth by counting '|' after excluding the filter string
        if pseudo_regex and level is not None:
            pipe = re.compile('\\|')
            regexp = str(s).replace('*', '')
            depth = [len(pipe.findall(c.replace(regexp, ''))) <= level
                     for c in data]
            matches = matches | (data.isin(subset) & depth)
        else:
            matches = matches | data.isin(subset)

    return matches


def years_match(data, years):
    """
    matching of year columns for data filtering
    """
    if isinstance(years, int):
        return data == years
    elif isinstance(years, list) or isinstance(years, range):
        return data.isin(years)
    else:
        raise ValueError('filtering for years by ' + years + ' not supported,' +
                         'must be int, list or range')
