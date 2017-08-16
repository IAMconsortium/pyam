# -*- coding: utf-8 -*-
"""
Initial version based on
https://github.com/iiasa/ceds_harmonization_analysis by Matt Gidden
"""

import os
import warnings

import re
import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import mpld3
    import seaborn as sns
except Exception:
    pass

# ignore warnings
warnings.filterwarnings('ignore')

try:
    import ixmp
    has_ix = True
except Exception:
    has_ix = False

# disable autoscroll in Jupyter notebooks
try:
    get_ipython().run_cell_magic(u'javascript', u'',
                                 u'IPython.OutputArea.prototype._should_scroll = function(lines) { return false; }')
except Exception:
    pass

# default settings for column headers
iamc_idx_cols = ['model', 'scenario', 'region', 'variable', 'unit']
all_idx_cols = iamc_idx_cols + ['year']

# %% class for working with IAMC-style timeseries data


class IamDataFrame(object):
    """This class is a wrapper for dataframes
    following the IAMC data convention.
    It provides a number of diagnostic features
    (including validation of values, completeness of variables provided)
    as well as a number of visualization and plotting tools."""

    def __init__(self, ix=None, path=None, file=None, ext='csv',
                 regions=None):
        """Initialize an instance of an IamDataFrame

        Parameters
        ----------
        ix: IxTimeseriesObject or IxDataStructure
            an instance of an IxTimeseriesObject or IxDataStructure
            (this option requires the ixmp package as a dependency)
        path: str
            the folder path where the data file is located
            (if reading in data from a snapshot csv or xlsx file)
        file: str
            the folder path where the data file is located
            (if reading in data from a snapshot csv or xlsx file)
        ext: str
            snapshot file extension
            (if reading in data from a snapshot csv or xlsx file)
        regions: list
            list of regions to be imported
        """
        # copy-constructor
        if ix is not None and isinstance(ix, IamDataFrame):
            self.data = ix.data
            self._meta = ix._meta

        # read data from source
        else:
            if ix is not None:
                self.data = read_ix(ix, regions)
            elif file and ext:
                self.data = read_data(path, file, ext, regions)

            # define a dataframe for categorization and other meta-data
            self._meta = return_index(self.data, ['model', 'scenario'],
                                      drop_duplicates=True)
            self.reset_category(True)

        # define a dictionary for category-color mapping
        self.cat_color = {'uncategorized': 'white', 'exclude': 'black'}
        self.col_count = 0

    def append(self, ix=None, path=None, file=None, ext='csv',  regions=None):
        """Read timeseries data and append to IamDataFrame

        Parameters
        ----------
        ix: IxTimeseriesObject or IxDataStructure
            an instance of an IxTimeseriesObject or IxDataStructure
            (this option requires the ixmp package as a dependency)
        path: str
            the folder path where the data file is located
            (if reading in data from a snapshot csv or xlsx file)
        file: str
            the folder path where the data file is located
            (if reading in data from a snapshot csv or xlsx file)
        ext: str
            snapshot file extension
            (if reading in data from a snapshot csv or xlsx file)
        regions: list
            list of regions to be imported
        """
        new = IamDataFrame(ix=self)

        if ix is not None:
            df = read_ix(ix, regions)
        elif file and ext:
            df = read_data(path, file, ext, regions)

        # check that model/scenario is not yet included in this IamDataFrame
        meta = return_index(df, ['model', 'scenario'], drop_duplicates=True)
        meta['category'] = 'uncategorized'
        new._meta = new._meta.append(meta, verify_integrity=True)

        # add new timeseries to data and append to metadata
        new.data = new.data.append(df)
        return new

    def models(self, filters={}):
        """Get a list of models filtered by specific characteristics

        Parameters
        ----------
        filters: dict, optional
            filter by model, scenario, region, variable, year, or category
            see function select() for details
        """
        return list(self.select(filters, ['model']).model)

    def scenarios(self, filters={}):
        """Get a list of scenarios filtered by specific characteristics

        Parameters
        ----------
        filters: dict, optional
            filter by model, scenario, region, variable, year, or category
            see function select() for details
        """
        return list(self.select(filters, ['scenario']).scenario)

    def regions(self, filters={}):
        """Get a list of regions filtered by specific characteristics

        Parameters
        ----------
        filters: dict, optional
            filter by model, scenario, region, variable, year, or category
            see function select() for details
        """
        return list(self.select(filters, ['region']).region)

    def variables(self, filters={}, include_units=False):
        """Get a list of variables filtered by specific characteristics

        Parameters
        ----------
        filters: dict, optional
            filter by model, scenario, region, variable, or year
            see function select() for details
        """
        return list(self.select(filters, ['variable']).variable)

    def pivot_table(self, index, columns, filters={}, values=None,
                    aggfunc='count', fill_value=None, style=None):
        """Returns a pivot table

        Parameters
        ----------
        index: str or list of strings
            rows for Pivot table
        columns: str or list of strings
            columns for Pivot table
        filters: dict, optional
            filter by model, scenario, region, variable, year, or category
            see function select() for details
        values: str, optional
            dataframe column to aggregate or count
        aggfunc: str or function, default 'count'
            function used for aggregation,
            accepts 'count', 'mean', and 'sum'
        fill_value: scalar, default None
            value to replace missing values with
        style: str, default None
            output style for pivot table formatting
            accepts 'highlight_not_max', 'heatmap'
        """
        if not values:
            return pivot_has_elements(self.select(filters, index+columns),
                                      index=index, columns=columns)
        else:
            cols = index + columns + [values]
            df = self.select(filters, cols)

            # allow 'aggfunc' to be passed as string for easier user interface
            if isinstance(aggfunc, str):
                if aggfunc == 'count':
                    df = df.groupby(index+columns, as_index=False).count()
                    fill_value = 0
                elif aggfunc == 'mean':
                    df = df.groupby(index+columns, as_index=False).mean()\
                                                                  .round(2)
                    aggfunc = np.sum
                    fill_value = 0 if style == 'heatmap' else ""
                elif aggfunc == 'sum':
                    aggfunc = np.sum
                    fill_value = 0 if style == 'heatmap' else ""

            df_pivot = df.pivot_table(values=values, index=index,
                                      columns=columns, aggfunc=aggfunc,
                                      fill_value=fill_value)
            if style == 'highlight_not_max':
                return df_pivot.style.apply(highlight_not_max)
            if style == 'heatmap':
                cm = sns.light_palette("green", as_cmap=True)
                return df_pivot.style.background_gradient(cmap=cm)
            else:
                return df_pivot

    def timeseries(self, filters={}):
        """Returns a dataframe in the standard IAMC format

        Parameters
        ----------
        filters: dict, optional
            filter by model, scenario, region, variable, year, or category
            see function select() for details
        """
        return self.select(filters).pivot_table(index=iamc_idx_cols,
                                                columns='year')['value']

    def validate(self, criteria, filters={}, exclude=False, display='heatmap'):
        """Run validation checks on timeseries data

        Parameters
        ----------
        criteria: dict
            dictionary of variables mapped to a dictionary of checks
            ('up' and 'lo' for respective bounds, 'year' for years - optional)
        filters: dict, optional
            filter by model, scenario, region, or category
            (variables & years are replaced by the other arguments)
            see function select() for details
        exclude: bool
            models/scenarios failing the validation to be excluded from data
        display: str or None, default 'heatmap'
            display style of scenarios failing the validation
            (options: heatmap, list, df)
        """
        df = pd.DataFrame()
        for var, check in criteria.items():
            df = df.append(self.check(var, check,
                                      filters, ret_true=False))
        if len(df):
            n = str(len(df))
            if exclude:
                idx = return_index(df, ['model', 'scenario'])
                self._meta.loc[idx, 'category'] = 'exclude'
                print(n + " data points do not satisfy the criteria, " +
                      "categorized as 'exclude' in metadata")
            else:
                print(n + " data points do not satisfy the criteria")

            if display:
                if display == 'heatmap':
                    df.set_index(all_idx_cols, inplace=True)
                    cm = sns.light_palette("green", as_cmap=True)
                    return df.style.background_gradient(cmap=cm)
                else:
                    return return_df(df, display, all_idx_cols)
        else:
            print("All models and scenarios satisfy the criteria")

    def category(self, name=None, criteria=None, filters={}, comment=None,
                 assign=True, color=None, display=None):
        """Assign scenarios to a category according to specific criteria
        or display the category assignment

        Parameters
        ----------
        name: str (optional)
            category name - if None, return a dataframe or pivot table
            of all categories mapped to models/scenarios
        criteria: dict
            dictionary with variables mapped to applicable checks
            ('up' and 'lo' for respective bounds, 'year' for years - optional)
        filters: dict, optional
            filter by model, scenario, region, or category
            (variables & years are replaced by args in criteria)
            see function select() for details
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
        if not criteria:
            cat = self._meta.reset_index()
            if name:
                cat = cat[cat.category == name]
            for col, values in filters.items():
                cat = cat[keep_col_match(cat[col], values)]

            if display:
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
            for var, check in criteria.items():
                cat_idx = cat_idx.intersection(self.check(var, check,
                                                          filters).index)
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

                n = str(len(cat_idx))
                print(n + " scenarios categorized as '" + name + "'")

                # return the model/scenario as dataframe for visual output
                if display:
                    df = pd.DataFrame(index=cat_idx).reset_index()
                    return return_df(df, display, ['model', 'scenario'])
            else:
                print("No scenario satisfies the criteria")

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
                 idx_cols=['model', 'scenario', 'category'],
                 exclude_cat=['exclude'], display='list'):
        """Show metadata or add metadata information

        Parameters
        ----------
        meta: dataframe or series, default None
            if provided, adds columns to the metadata
        name: str, default None
            if df is series, name of new metadata column
        filters: dict, optional
            filter by model, scenario or category
        idx_cols: list of str, default ['model', 'scenario', 'category']
            columns that are set as index of the returned dataframe (if 'list')
        display: str, default 'list'
            accepts 'list' or 'df'
        exclude_cat: None or list of strings, default ['exclude']
            exclude all scenarios from the listed categories
        """
        # if a dataframe or series is provided, add to metadata dataframe
        if meta is not None:
            if isinstance(meta, pd.Series):
                meta = meta.to_frame(name)
            for name, series in meta.iteritems():
                for idx, val in series.iteritems():
                    self._meta.loc[idx, name] = val

        # otherwise, return metadata
        else:
            meta = self._meta.reset_index()
            if exclude_cat is not None:
                meta = meta[~meta['category'].isin(exclude_cat)]
            for col, values in filters.items():
                meta = meta[keep_col_match(meta[col], values)]
            return return_df(meta, display, idx_cols)

    def check(self, variable, check, filters=None, ret_true=True):
        """Check which model/scenarios satisfy specific criteria

        Parameters
        ----------
        variable: str
            variable to be checked
        check: dict
            dictionary with checks
            ('up' and 'lo' for respective bounds, 'year' for years - optional)
        filters: dict, optional
            filter by model, scenario, region, or category
            (variables & years are replaced by arguments of 'check')
            see function select() for details
        ret_true: bool, default True
            if true, return models/scenarios passing the check;
            otherwise, return datatframe of all failed checks
        """
        if not filters:
            filters = {}
        if 'year' in check:
            filters['year'] = check['year']
        filters['variable'] = variable
        df = self.select(filters)

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
                              .set_index(['model', 'scenario'])
            # if more than one year is filtered for, ensure that
            # the criteria are satisfied in every year
            else:
                num_yr = len(df.year.drop_duplicates())
                df_agg = df.loc[is_true, ['model', 'scenario', 'year']]\
                    .groupby(['model', 'scenario']).count()
                return pd.DataFrame(index=df_agg[df_agg.year == num_yr].index)
        else:
            return df[~is_true]

    def select(self, filters={}, cols=None, idx_cols=None,
               exclude_cat=['exclude']):
        """Select a subset of the data (filter) and set an index

        Parameters
        ----------
        filters: dict, optional
            The following columns are available for filtering:
             - 'category': filter by category assignment in meta-data
             - 'model', 'scenario', 'region': takes a string or list of strings
             - 'variable': takes a string or list of strings,
                where ``*`` can be used as a wildcard
             - 'year': takes an integer, a list of integers or a range
                (note that the last year of a range is not included,
                so ``range(2010,2015)`` is interpreted
                as ``[2010, 2011, 2012, 2013, 2014]``)
        cols: string or list
            columns returned for the dataframe, duplicates are dropped
        idx_cols: string or list
            columns that are set as index of the returned dataframe
        exclude_cat: None or list of strings, default ['exclude']
            exclude all scenarios from the listed categories
        """
        if exclude_cat is not None:
            idx = self._meta[~self._meta['category'].isin(exclude_cat)].index
            keep = return_index(self.data, ['model', 'scenario']).isin(idx)
        else:
            keep = np.array([True] * len(self.data))

        # filter by columns and list of values
        for col, values in filters.items():
            if col == 'category':
                cat_idx = self._meta[keep_col_match(self._meta['category'],
                                                    values)].index
                keep_col = return_index(self.data, ['model', 'scenario'])\
                    .isin(cat_idx)

            elif col in ['model', 'scenario', 'region']:
                keep_col = keep_col_match(self.data[col], values)

            elif col == 'variable':
                keep_col = keep_col_match(self.data[col], values, True)

            elif col in ['year']:
                keep_col = keep_col_yr(self.data[col], values)

            else:
                raise SystemError(
                        'filter by column ' + col + ' not supported')
            keep = keep & keep_col

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

    def plot_lines(self, filters={}, idx_cols=None, color_by_cat=False,
                   save=None, interactive_plots=True, return_ax=False):
        """Simple line plotting feature

        Parameters
        ----------
        filters: dict, optional
            filter by model, scenario, region, variable, year, or category
            see function select() for details
        idx_cols: str or list of strings, optional
            list of index columns to display
            (summing over non-selected columns)
        color_by_cat: boolean, default False
            use category coloring scheme, replace full legend by category
        save: str, optional
             filename for export of figure (as png)
        interactive_plots: boolean
            use mpld3 for interactive plots (mouse-over)
        return_ax: boolean, optional, default False
            return the 'axes()' object of the plot
            (interactive_plots deactivated)
        """
        if return_ax:
            interactive_plots = False
        if interactive_plots:
            mpld3.enable_notebook()
        else:
            mpld3.disable_notebook()

        if not idx_cols:
            idx_cols = iamc_idx_cols
        cols = idx_cols + ['year', 'value']

        # select data, drop 'uncategorized' if option color_by_cat is selected
        if color_by_cat:
            df = self.select(filters, cols,
                             exclude_cat=['exclude', 'uncategorized'])
        else:
            df = self.select(filters, cols)

        # pivot dataframe for use by matplotlib, start plot
        df = df.pivot_table(values='value', index=['year'], columns=idx_cols,
                            aggfunc=np.sum)
        plt.cla()
        ax = plt.axes()

        # drop index columns if it only has one level
        # shift to title or y-axis (if unit) for more elegant figures
        title = None
        i = 0
        for col in df.columns.names:
            if len(df.columns.levels[i]) == 1:
                level = str(df.columns.levels[i][0])
                df.columns = df.columns.droplevel(i)
                if col == 'unit':
                    plt.ylabel(level)
                elif title is not None:
                    title = '{} - {}: {}'.format(title, col, level)
                else:
                    title = '{}: {}'.format(col, level)
            else:
                i += 1

        for (col, data) in df.iteritems():
            if color_by_cat:
                color = self.cat_color[self._meta.loc[col[0:2]].category]
                lines = ax.plot(data, color=color)
            else:
                lines = ax.plot(data)

            if interactive_plots:
                if isinstance(col, tuple):
                    label = col[0]
                    for i in range(1, len(col)):
                        label = '{} - {}'.format(label, col[i])
                else:
                    label = col
                tooltips = mpld3.plugins.LineLabelTooltip(lines[0], label)
                mpld3.plugins.connect(plt.gcf(), tooltips)

            # only show the legend if no more than 12 rows
            if len(df) <= 12:
                ax.legend(loc='best', framealpha=0.0)

        plt.title(title)
        plt.xlabel('Years')
        if save:
            plt.savefig(save)

        if interactive_plots:
            return mpld3.display()
        else:
            plt.show()

        if return_ax:
            return ax

# %% auxiliary function for reading data from snapshot file


def read_ix(ix, regions=None):
    """Read timeseries data from an ix object

    Parameters
    ----------
    ix: IxTimeseriesObject or IxDataStructure
        an instance of an IxTimeseriesObject or IxDataStructure
        (this option requires the ixmp package as a dependency)
    regions: list
        list of regions to be loaded from the database snapshot
    """
    if not has_ix:
        error = 'this option depends on the ixmp package'
        raise SystemError(error)
    if isinstance(ix, ixmp.ixDatastructure):
        df = ix.timeseries()
        df['model'] = ix.model
        df['scenario'] = ix.scenario
    else:
        error = 'arg ' + ix + ' not recognized as valid ix object'
        raise SystemError(error)

    return df


def read_data(path=None, file=None, ext='csv', regions=None):
    """Read data from a snapshot file saved in the standard IAMC format

    Parameters
    ----------
    path: str
        the folder path where the data file is located
    file: str
        the folder path where the data file is located
    ext: str
        snapshot file extension
    regions: list
        list of regions to be loaded from the database snapshot
    """
    if path is not None:
        fname = '{}/{}.{}'.format(path, file, ext)
    else:
        fname = '{}.{}'.format(file, ext)

    if not os.path.exists(fname):
        raise SystemError("no snapshot file '" + fname + "' found!")

    # read from database snapshot csv
    if ext == 'csv':
        try:
            df = pd.read_csv(fname)
        except UnicodeDecodeError:
            df = pd.read_csv(fname, encoding='ISO-8859-1')
        df = df.rename(columns={c: str(c).lower() for c in df.columns})

        # filter by selected regions
        if regions:
            df = df[df['region'].isin(regions)]

        # transpose dataframe by year column
        idx = iamc_idx_cols
        numcols = sorted(set(df.columns) - set(idx))
        df = pd.melt(df, id_vars=idx, var_name='year',
                     value_vars=numcols, value_name='value')
        df.year = pd.to_numeric(df.year)

        # drop NaN's
        df.dropna(inplace=True)
    else:
        raise SystemError('file type ' + ext + ' is not supported')

    return df

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


def keep_col_match(col, strings, pseudo_regex=False):
    """
    matching of model/scenario names, variables, regions, and categories
    to pseudo-regex (optional) for data filtering
    """
    keep_col = np.array([False] * len(col))

    if isinstance(strings, str):
        strings = [strings]

    for s in strings:
        if pseudo_regex:
            s = s.replace('|', '\\|').replace('*', '.*') + "$"
        pattern = re.compile(s)
        subset = filter(pattern.match, col)
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
        raise SystemError('filtering for years by ' + yrs + ' not supported,' +
                          'must be int, list or range')

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
    highlight the maximum in a Series yellow.
    '''
    is_max = s == s.max()
    return ['' if v else 'background-color: yellow' for v in is_max]
