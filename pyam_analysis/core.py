# -*- coding: utf-8 -*-
"""
Initial version based on
https://github.com/iiasa/ceds_harmonization_analysis by Matt Gidden
"""

import copy
import os
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

mod_scen = ['model', 'scenario']
iamc_idx_cols = ['model', 'scenario', 'region', 'variable', 'unit']
all_idx_cols = iamc_idx_cols + ['year']


# %% class for working with IAMC-style timeseries data


class IamDataFrame(object):
    """This class is a wrapper for dataframes
    following the IAMC data convention.
    It provides a number of diagnostic features
    (including validation of values, completeness of variables provided)
    as well as a number of visualization and plotting tools."""

    def __init__(self, data, regions=None, variables=None, units=None,
                 years=None, **kwargs):
        """Initialize an instance of an IamDataFrame

        Parameters
        ----------
        data: ixmp.TimeSeries, ixmp.Scenario, pandas.DataFrame or data file
            an instance of an TimeSeries or Scenario (requires `ixmp`),
            or pandas.DataFrame or data file with IAMC-format data columns
        regions : list of strings
            filter by regions
        variables : list of strings
            filter by variables (only with ixmp objects)
        units : list of strings
            filter by units (only with ixmp objects)
        years : list of integers
            filter by years (only with ixmp objects)
        """
        # import data from pandas.DataFrame or read from source
        if isinstance(data, pd.DataFrame):
            self.data = format_data(data)
        elif has_ix and isinstance(data, ixmp.TimeSeries):
            self.data = read_ix(data, regions=regions, variables=variables,
                                units=units, years=years)
        else:
            self.data = read_data(data, regions, **kwargs)

        # define a dataframe for categorization and other meta-data
        self._meta = return_index(self.data, mod_scen,
                                  drop_duplicates=True)
        self.reset_category(True)

        # define a dictionary for category-color mapping
        self.cat_color = {'uncategorized': 'white', 'exclude': 'black'}
        self.col_count = 0

    def append(self, other, regions=None, variables=None, units=None,
               years=None, **kwargs):
        """Import or read timeseries data and append to IamDataFrame

        Parameters
        ----------
        other: pyam-analysis.IamDataFrame, ixmp.TimeSeries, ixmp.Scenario,
        pandas.DataFrame or data file
            an IamDataFrame, TimeSeries or Scenario (requires `ixmp`),
            or pandas.DataFrame or data file with IAMC-format data columns
        regions : list of strings
            filter by regions
        variables : list of strings
            filter by variables (only with ixmp objects)
        units : list of strings
            filter by units (only with ixmp objects)
        years : list of integers
            filter by years (only with ixmp objects)
        """
        new = copy.deepcopy(self)

        if isinstance(other, IamDataFrame):
            df = other.data
            meta = other._meta
            # TODO merge other.cat_color
        else:
            if isinstance(other, pd.DataFrame):
                df = format_data(other)
            elif has_ix and isinstance(other, ixmp.TimeSeries):
                df = read_ix(other, regions=regions, variables=variables,
                             units=units, years=years)
            elif os.path.isfile(other):
                df = read_data(other, regions, **kwargs)
            else:
                raise ValueError("arg '{}' not recognized as valid source"
                                 .format(other))
            meta = return_index(df, mod_scen, drop_duplicates=True)
            meta['category'] = 'uncategorized'

        # check that model/scenario is not yet included in this IamDataFrame
        new._meta = new._meta.append(meta, verify_integrity=True)

        # add new data
        new.data = new.data.append(df).reset_index(drop=True)
        return new

    def export_metadata(self, path):
        """Export metadata to Excel

        Parameters
        ----------
        path:
            path/filename for xlsx file of metadata export
        """
        writer = pd.ExcelWriter(path)
        meta = self.metadata(display='df')
        iam.utils.write_sheet(writer, 'categories', meta)
        colors = pd.DataFrame({'category': list(self.cat_color.keys()),
                               'color': list(self.cat_color.values())})
        iam.utils.write_sheet(writer, 'categories_color', colors)
        writer.save()

    def load_metadata(self, path, *args):
        """Load metadata from previously exported instance of pyam_analysis

        Parameters
        ----------
        path:
            xlsx file with metadata exported from an instance of pyam_analysis
        """

        if not os.path.exists(path):
            raise ValueError("no metadata file '" + path + "' found!")

        df = pd.read_excel(path, sheet_name='categories', *args)
        req_cols = ['model', 'scenario', 'category']
        if not set(req_cols).issubset(set(df.columns)):
            e = "metadata file '{}' does not have required columns ({})!"
            raise ValueError(e.format(path, req_cols))

        df.set_index(mod_scen, inplace=True)

        # check for metadata import of entries not existing in the data
        diff = df.index.difference(self._meta.index)
        if not diff.empty:
            w = 'failing to import {} metadata {} for non-existing scenario,' \
                ' see list of dropped scenarios below'
            e = 'entry' if len(diff == 1) else 'entries'
            warnings.warn(w.format(len(diff), e), Warning, stacklevel=0)
            dropped = df.loc[diff, 'category']
            df.drop(diff, inplace=True)

        # replace imported metadata for existing entries
        overlap = self._meta.index.intersection(df.index)
        conflict = ~(self._meta.loc[overlap].category == 'uncategorized')
        count_conflict = sum(conflict)
        if count_conflict:
            w = 'overwriting {} metadata {}'
            e = 'entry' if count_conflict == 1 else 'entries'
            warnings.warn(w.format(count_conflict, e), Warning, stacklevel=0)
        self._meta.drop(overlap, inplace=True)

        self._meta = self._meta.append(df)

        colors = pd.read_excel(path, sheet_name='categories_color', *args)
        for i in colors.index:
            row = colors.iloc[i]
            self.cat_color[row['category']] = row['color']

        # display imported model/scenario metadata for nonexisting data
        if not diff.empty:
            return pd.DataFrame(dropped)

    def models(self, filters={}):
        """Get a list of models filtered by specific characteristics

        Parameters
        ----------
        filters: dict, optional
            filter by model, scenario, region, variable, level, year, category
            see function _select() for details
        """
        return list(self._select(filters, ['model']).model)

    def scenarios(self, filters={}):
        """Get a list of scenarios filtered by specific characteristics

        Parameters
        ----------
        filters: dict, optional
            filter by model, scenario, region, variable, level, year, category
            see function _select() for details
        """
        return list(self._select(filters, ['scenario']).scenario)

    def regions(self, filters={}):
        """Get a list of regions filtered by specific characteristics

        Parameters
        ----------
        filters: dict, optional
            filter by model, scenario, region, variable, level, year, category
            see function _select() for details
        """
        return list(self._select(filters, ['region']).region)

    def variables(self, filters={}, include_units=False):
        """Get a list of variables filtered by specific characteristics

        Parameters
        ----------
        filters: dict, optional
            filter by model, scenario, region, variable, level, year, category
            see function _select() for details
        include_units: boolean, default False
            include the units
        """
        if include_units:
            x = self._select(filters, ['variable', 'unit'])
            x.sort_values(by='variable')
        else:
            x = list(self._select(filters, ['variable']).variable)
            x.sort()
        return x

    def pivot_table(self, index, columns, filters={}, values='value',
                    exclude_cat=['exclude'],
                    aggfunc='count', fill_value=None, style=None):
        """Returns a pivot table

        Parameters
        ----------
        index: str or list of strings
            rows for Pivot table
        columns: str or list of strings
            columns for Pivot table
        filters: dict, optional
            filter by model, scenario, region, variable, level, year, category
            see function _select() for details
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
        if isinstance(index, str):
            index = [index]
        if isinstance(columns, str):
            columns = [columns]

        cols = index + columns + [values]
        df = self._select(filters, cols, exclude_cat=exclude_cat)

        # allow 'aggfunc' to be passed as string for easier user interface
        if isinstance(aggfunc, str):
            if aggfunc == 'count':
                df = df.groupby(index + columns, as_index=False).count()
                fill_value = 0
            elif aggfunc == 'mean':
                df = df.groupby(index + columns, as_index=False).mean()\
                    .round(2)
                aggfunc = np.sum
                fill_value = 0 if style == 'heatmap' else ""
            elif aggfunc == 'sum':
                aggfunc = np.sum
                fill_value = 0 if style == 'heatmap' else ""

        df_pivot = df.pivot_table(values=values, index=index, columns=columns,
                                  aggfunc=aggfunc, fill_value=fill_value)
        if style == 'highlight_not_max':
            return df_pivot.style.apply(highlight_not_max)
        if style == 'heatmap':
            cm = sns.light_palette("green", as_cmap=True)
            return df_pivot.style.background_gradient(cmap=cm)
        else:
            return df_pivot

    def timeseries(self):
        """Returns a dataframe in the standard IAMC format
        """
        return self.data.pivot_table(index=iamc_idx_cols,
                                   columns='year')['value']

    def validate(self, criteria, filters={}, exclude_cat=['exclude'],
                 exclude=False, silent=False, display='heatmap'):
        """Run validation checks on timeseries data

        Parameters
        ----------
        criteria: str or dict
            string for variable name to be checked for existence
            or dictionary of variables mapped to a dictionary of checks
            ('up' and 'lo' for respective bounds, 'year' for years - optional)
        filters: dict, optional
            filter by model, scenario, region, variable, level, year, category
            see function _select() for details
            filter by 'variable'/'year' is replaced by arguments of 'criteria'
            see function _check() for details
        exclude_cat: list of strings, default ['exclude']
            exclude all scenarios from the listed categories from validation
        exclude: bool, default False
            models/scenarios failing the validation to be excluded from data
        silent: bool, default False
            if False, print a summary statement of validation
        display: str or None, default 'heatmap'
            display style of scenarios failing the validation
            (options: heatmap, list, df)
        """
        # get dataframe of meta-data, filter by model, scenario, category
        cols = ['model', 'scenario', 'category']
        meta = self.metadata(filters=filtered_dict(filters, cols),
                             exclude_cat=exclude_cat).index
        count = len(meta)

        # if criteria is a string, check that each scenario has this variable
        if isinstance(criteria, str):
            data_filters = filters.copy()
            data_filters.update({'variable': criteria})
            idx = self._select(data_filters, exclude_cat=exclude_cat,
                               idx_cols=mod_scen).index

            df = pd.DataFrame(index=meta)
            df['keep'] = True
            if len(idx):
                df.loc[idx, 'keep'] = False
            df = df[df.keep].reset_index()[mod_scen]
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
                idx = return_index(df, mod_scen)
                self._meta.loc[idx, 'category'] = 'exclude'
                msg += ", categorized as 'exclude' in metadata"

            if not silent:
                logger().info(msg.format(n, count, s))

            if isinstance(criteria, str):
                return df
            else:
                if display:
                    if display == 'heatmap':
                        df.set_index(all_idx_cols, inplace=True)
                        cm = sns.light_palette("green", as_cmap=True)
                        return df.style.background_gradient(cmap=cm)
                    else:
                        return return_df(df, display, all_idx_cols)
        elif not silent:
            logger().info('{} scenarios satisfy the criteria'.format(count))

    def category(self, name=None, criteria=None, filters={}, comment=None,
                 assign=True, color=None, display=None):
        """Assign scenarios to a category according to specific criteria
        or display the category assignment

        Parameters
        ----------
        name: str (optional)
            category name - if None, return a dataframe or pivot table
            of all categories mapped to models/scenarios
        criteria: dict, default None
            dictionary with variables mapped to applicable checks
            ('up' and 'lo' for respective bounds, 'year' for years - optional)
        filters: dict, optional
            filter by model, scenario, region, variable, level, year, category
            see function _select() for details
            filter by 'variable'/'year' is replaced by arguments of 'criteria'
            see function _check() for details
        comment: str
            a comment pertaining to the category
        assign: boolean, default True
            assign categorization to data (if false, display only)
        color: str
            assign a color to this category
        display: str or None, default None
            display style of scenarios assigned to this category
            (list, pivot, df - no display if None)
        """
        # for returning a list or pivot table of all categories or one specific
        if criteria is None:
            cat = self._meta.reset_index()
            if name:
                cat = cat[cat.category == name]
            for col, values in filters.items():
                cat = cat[keep_col_match(cat[col], values)]

            if display is not None:
                if display == 'pivot':
                    cat = cat.pivot(index='model', columns='scenario',
                                    values='category')
                    return cat.style.apply(color_by_cat,
                                           cat_col=self.cat_color, axis=None)
                else:
                    return return_df(cat, display,
                                     ['category', 'model', 'scenario'])

        # when criteria are provided, use them to assign a new category
        else:
            # TODO clear out existing assignments to that category?
            cat_idx = self._meta.index
            if criteria:
                for var, check in criteria.items():
                    cat_idx = cat_idx.intersection(self._check(var, check,
                                                               filters).index)
            # if criteria is empty, use all scenarios that satisfy 'filters'
            else:
                filter_idx = self._select(filters, idx_cols=mod_scen).index
                cat_idx = cat_idx.intersection(filter_idx)

            if len(cat_idx):
                # assign selected model/scenario to internal category mapping
                if assign:
                    self._meta.loc[cat_idx, 'category'] = name

                # assign a color to this category for pivot tables and plots
                if color:
                    self.cat_color[name] = color
                elif name not in self.cat_color:
                    self.cat_color[name] = sns.color_palette("hls",
                                                             8)[self.col_count]
                    self.col_count += 1

                n = len(cat_idx)
                s = 'scenario' if n == 1 else 'scenarios'
                logger().info("{} {} categorized as '{}'".format(n, s, name))

                # return the model/scenario as dataframe for visual output
                if display:
                    df = pd.DataFrame(index=cat_idx).reset_index()
                    return return_df(df, display, mod_scen)
            else:
                logger().info("No scenarios satisfy the criteria")

    def reset_category(self, reset_exclude=False):
        """Reset category assignment for all scenarios to 'uncategorized'

        Parameters
        ----------
        reset_exclude: boolean, default False
            reset the category for scenarios marked 'exclude
        """
        name = 'uncategorized'
        if reset_exclude:
            self._meta['category'] = name
        else:
            cat_idx = self._meta[self._meta['category'] != 'exclude'].index
            self._meta.loc[cat_idx, 'category'] = name

    def metadata(self, meta=None, name=None, filters={},
                 idx_cols=['model', 'scenario'],
                 exclude_cat=['exclude'], display='list'):
        """Show metadata or add metadata information

        Parameters
        ----------
        meta: dataframe or series, default None
            if provided, adds columns to the metadata
        name: str, default None
            if df is series, name of new metadata column
        filters: dict, optional
            filter by model, scenario, region, variable, level, year, category
            see function _select() for details
        idx_cols: list of str, default ['model', 'scenario']
            columns that are set as index of the returned dataframe (if 'list')
        exclude_cat: None or list of strings, default ['exclude']
            exclude all scenarios from the listed categories
        display: str, default 'list'
            accepts 'list' or 'df'
        """
        # if a dataframe or series is provided, add to metadata dataframe
        if meta is not None:
            if isinstance(meta, pd.Series):
                meta = meta.to_frame(name)
            for name, series in meta.iteritems():
                for idx, val in series.iteritems():
                    if len(idx) > 2:
                        idx = idx[0:2]
                    self._meta.loc[idx, name] = val

        # otherwise, return metadata
        else:
            meta = self._meta.reset_index()
            if exclude_cat is not None:
                meta = meta[~meta['category'].isin(exclude_cat)]
            for col, values in filters.items():
                meta = meta[keep_col_match(meta[col], values)]
            return return_df(meta, display, idx_cols)

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

    def interpolate(self, year, exclude_cat=['exclude']):
        """Interpolate missing values in timeseries (linear interpolation)

        Parameters
        ----------
        year: int
             year to be interpolated
        exclude_cat: None or list of strings, default ['exclude']
             exclude all scenarios from the listed categories
        """
        df = self.pivot_table(index=iamc_idx_cols, columns=['year'],
                              values='value', aggfunc=np.sum,
                              exclude_cat=exclude_cat)
        fill_values = df.apply(iam.fill_series, raw=False, axis=1, year=year)
        fill_values = fill_values.dropna().reset_index()
        fill_values = fill_values.rename(columns={0: "value"})
        fill_values['year'] = year
        self.data = self.data.append(fill_values)

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
            see function _select() for details
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
        df = self._select(filters)

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
                    .set_index(mod_scen)
            # if more than one year is filtered for, ensure that
            # the criteria are satisfied in every year
            else:
                num_yr = len(df.year.drop_duplicates())
                df_agg = df.loc[is_true, ['model', 'scenario', 'year']]\
                    .groupby(mod_scen).count()
                return pd.DataFrame(index=df_agg[df_agg.year == num_yr].index)
        else:
            return df[~is_true]

    def _filter_columns(self, filters):
        keep = np.array([True] * len(self.data))

        # filter by columns and list of values
        for col, values in filters.items():
            if col == 'category':
                cat_idx = self._meta[keep_col_match(self._meta['category'],
                                                    values)].index
                keep_col = return_index(self.data, mod_scen).isin(cat_idx)

            elif col in ['model', 'scenario', 'region']:
                keep_col = keep_col_match(self.data[col], values)

            elif col == 'variable':
                level = filters['level'] if 'level' in filters.keys() else None
                keep_col = keep_col_match(self.data[col], values, True, level)

            elif col in ['year']:
                keep_col = keep_col_yr(self.data[col], values)

            elif col in ['level']:
                if 'variable' not in filters.keys():
                    keep_col = keep_col_match(self.data['variable'], '*',
                                              True, values)
                else:
                    continue
            else:
                raise SystemError(
                    'filter by column ' + col + ' not supported')
            keep = keep & keep_col
        return keep

    def filter(self, filters, inplace=False):
        """Return a filtered IamDataFrame (i.e., a subset of current data)

        Parameters
        ----------
        filters: dict
            The following columns are available for filtering:
             - 'category': filter by category assignment in metadata
             - 'model', 'scenario', 'region': takes a string or list of strings
             - 'variable': takes a string or list of strings,
                where ``*`` can be used as a wildcard
             - 'level': the maximum "depth" of IAM variables (number of '|')
               (exluding the strings given in the 'variable' argument)
             - 'year': takes an integer, a list of integers or a range
                note that the last year of a range is not included,
                so ``range(2010,2015)`` is interpreted as ``[2010, ..., 2014]``
        inplace : bool, optional
            if True, operate on this object, otherwise return a copy
            default: False
        """
        keep = self._filter_columns(filters)
        ret = copy.deepcopy(self) if not inplace else self
        ret.data = ret.data[keep]

        idx = pd.MultiIndex.from_tuples(
            pd.unique(list(zip(ret.data['model'], ret.data['scenario']))),
            names=('model', 'scenario')
        )
        ret._meta = ret._meta.loc[idx]
        return ret

    def head(self, *args, **kwargs):
        """Identical to pd.DataFrame.head() operating on data"""
        return self.data.head(*args, **kwargs)

    def _select(self, filters={}, cols=None, idx_cols=None,
                exclude_cat=['exclude']):
        """Select a subset of the data (filter) and set an index

        Parameters
        ----------
        filters: dict, optional
            The following columns are available for filtering:
             - 'category': filter by category assignment in metadata
             - 'model', 'scenario', 'region': takes a string or list of strings
             - 'variable': takes a string or list of strings,
                where ``*`` can be used as a wildcard
             - 'level': the maximum "depth" of IAM variables (number of '|')
               (exluding the strings given in the 'variable' argument)
             - 'year': takes an integer, a list of integers or a range
                note that the last year of a range is not included,
                so ``range(2010,2015)`` is interpreted as ``[2010, ..., 2014]``
        cols: string or list
            columns returned for the dataframe, duplicates are dropped
        idx_cols: string or list
            columns that are set as index of the returned dataframe
        exclude_cat: None or list of strings, default ['exclude']
            exclude all scenarios from the listed categories
        """
        keep = self._filter_columns(filters)

        if exclude_cat is not None:
            idx = self._meta[~self._meta['category'].isin(exclude_cat)].index
            keep &= return_index(self.data, mod_scen).isin(idx)

        df = self.data[keep].copy()

        # select columns (and index columns), drop duplicates
        if cols is not None:
            if idx_cols:
                cols = cols + idx_cols
            df = df[cols].drop_duplicates()

        # set (or reset) index
        if idx_cols is not None:
            return df.set_index(idx_cols)
        else:
            return df.reset_index(drop=True)

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
                  .join(self.metadata())
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


def read_ix(ix, regions=None, variables=None, units=None, years=None):
    """Read timeseries data from an ix object

    Parameters
    ----------
    ix: ixmp.TimeSeries or ixmp.Scenario
        this option requires the ixmp package as a dependency
    regions: list
        list of regions to be loaded from the database snapshot
    """
    if isinstance(ix, ixmp.TimeSeries):
        df = ix.timeseries(iamc=False, regions=regions, variables=variables,
                           units=units, years=years)
        df['model'] = ix.model
        df['scenario'] = ix.scenario
    else:
        error = 'arg ' + ix + ' not recognized as valid ixmp class'
        raise ValueError(error)

    return df


def read_data(fname, regions=None, *args, **kwargs):
    """Read data from a snapshot file saved in the standard IAMC format
    or a table with year/value columns

    Parameters
    ----------
    fname: str
        a file with IAMC-style data
    regions: list
        list of regions to be loaded from the database snapshot
    """
    if not os.path.exists(fname):
        raise ValueError("no data file '" + fname + "' found!")

    # read from database snapshot csv or xlsx
    if fname.endswith('csv'):
        df = pd.read_csv(fname, *args, **kwargs)
    else:
        df = pd.read_excel(fname, *args, **kwargs)

    return(format_data(df))


def format_data(data, regions=None):
    """Convert an imported dataframe and check all required columns

    Parameters
    ----------
    data: pandas.DataFrame
        dataframe to be converted to an IamDataFrame
    regions: list
        list of regions to be loaded from the database snapshot
    """
    # format columns to lower-case and check that all required columns exist
    data = data.rename(columns={c: str(c).lower() for c in data.columns})
    if not set(iamc_idx_cols).issubset(set(data.columns)):
        missing = list(set(iamc_idx_cols) - set(data.columns))
        raise ValueError("missing required columns {}!".format(missing))

    # filter by selected regions
    if regions:
        data = data[keep_col_match(data['region'], regions)]

    # check whether data in IAMC style or year/value layout
    if 'value' not in data.columns:
        numcols = sorted(set(data.columns) - set(iamc_idx_cols))
        data = pd.melt(data, id_vars=iamc_idx_cols, var_name='year',
                       value_vars=numcols, value_name='value')
    data.year = pd.to_numeric(data.year)

    # drop NaN's
    data.dropna(inplace=True)

    return data


# %% auxiliary functions for data filtering


def return_df(df, display, idx_cols=None):
    """returns a dataframe with display options"""
    if display == 'df':
        return df.reset_index(drop=True)
    elif display == 'list':
        return df.set_index(idx_cols)
    else:
        warnings.warn("Display option '" + display + "' not supported!")


def return_index(df, idx_cols, drop_duplicates=False):
    """set and return an index for a dataframe"""
    if drop_duplicates:
        return df[idx_cols].drop_duplicates().set_index(idx_cols)
    else:
        return df[idx_cols].set_index(idx_cols).index


def keep_col_match(col, strings, pseudo_regex=False, level=None):
    """
    matching of model/scenario names, variables, regions, and categories
    to pseudo-regex (optional) for data filtering
    """
    keep_col = np.array([False] * len(col))

    if isinstance(strings, str):
        strings = [strings]

    for s in strings:
        regexp = str(s)
        if pseudo_regex:
            regexp = regexp.replace('|', '\\|').replace('*', '.*') + "$"
        pattern = re.compile(regexp)
        subset = filter(pattern.match, col)
        # check for depth by counting '|' after excluding the filter string
        if pseudo_regex and level is not None:
            pipe = re.compile('\\|')
            regexp = str(s).replace('*', '')
            depth = [len(pipe.findall(c.replace(regexp, ''))) <= level
                     for c in col]
            keep_col = keep_col | (col.isin(subset) & depth)
        else:
            keep_col = keep_col | col.isin(subset)

    return keep_col


def keep_col_yr(col, yrs):
    """
    matching of year columns for data filtering
    """
    if isinstance(yrs, int):
        return col == yrs
    elif isinstance(yrs, list) or isinstance(yrs, range):
        return col.isin(yrs)
    else:
        raise ValueError('filtering for years by ' + yrs + ' not supported,' +
                         'must be int, list or range')


def filtered_dict(dct, keys={}):
    """
    sub-select a dictionary by list of keys
    """
    filtered_dict = {}
    for key, values in dct.items():
        if key in keys:
            filtered_dict[key] = values
    return filtered_dict


# %% auxiliary functions for table and graph formatting

def color_by_cat(cat_df, cat_col):
    """
    returns a dataframe with style attributes
    """
    attr = np.where(cat_df, '',
                    'color: {0}; background-color: {0};'.format('lightgrey'))
    attr = np.where(cat_df == 'uncategorized',
                    'color: {0}; background-color: {0}'.format('white'), attr)
    for c in cat_col:
        attr = np.where(cat_df == c, 'color: {0}; background-color: {0}'
                        .format(cat_col[c]), attr)
    return pd.DataFrame(attr, index=cat_df.index, columns=cat_df.columns)


def pivot_has_elements(df, index, columns):
    """
    returns a pivot table with existing index-columns combinations highlighted
    """
    df.reset_index(inplace=True)
    df['has_elements'] = (np.array([True] * len(df)))
    df_pivot = df.pivot_table(values='has_elements',
                              index=index, columns=columns, fill_value='')
    return df_pivot.style.applymap(highlight_has_element)


def highlight_has_element(val):
    """
    highlights table cells green if value is True
    """
    color = 'green' if val else 'white'
    return 'color: {0}; background-color: {0}'.format(color)


def highlight_not_max(s):
    '''
    highlight the maximum in a Series using yellow background
    '''
    is_max = s == s.max()
    return ['' if v else 'background-color: yellow' for v in is_max]
