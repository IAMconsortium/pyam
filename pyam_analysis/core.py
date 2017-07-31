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
import matplotlib.pyplot as plt
import seaborn as sns

# ignore warnings
warnings.filterwarnings('ignore')

# disable autoscroll in Jupyter notebooks
try:
    get_ipython().run_cell_magic(u'javascript', u'',
                                 u'IPython.OutputArea.prototype._should_scroll = function(lines) { return false; }')
except:
    pass

iamc_idx_cols = ['model', 'scenario', 'region', 'variable', 'unit']
all_idx_cols = iamc_idx_cols + ['year']

# %% class for working with IAMC-style timeseries data


class IamDataFrame(object):
    """This class is a wrapper for dataframes
    following the IAMC data convention."""

    def __init__(self, mp=None, path=None, file=None, ext='csv',
                 regions=None):
        """Initialize an instance of an IamDataFrame

        Parameters
        ----------
        mp: ixPlatform
            an instance of an ix modeling platform (ixmp)
            (if initializing from an ix platform)
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
        if mp:
            raise SystemError('connection to ix platform not yet supported')
        elif file and ext:
            self.data = read_data(path, file, ext, regions)

        # define a dataframe for categorization and other meta-data
        self.cat = self.data[['model', 'scenario']].drop_duplicates()\
            .set_index(['model', 'scenario'])
        self.reset_category()

        # define a dictionary for category-color mapping
        self.cat_color = {'uncategorized': 'white', 'exclude': 'black'}
        self.col_count = 0

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

    def variables(self, filters={}):
        """Get a list of variables filtered by specific characteristics

        Parameters
        ----------
        filters: dict, optional
            filter by model, scenario, region, variable, or year
            see function select() for details
        """
        return list(self.select(filters, ['variable']).variable)

    def pivot_table(self, index, columns, filters={}, values=None,
                    aggfunc='count', fill_value=None,
                    style=None):
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
            output style for dataframe formatting
            accepts 'highlight_not_max', 'heat_map'
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
                    fill_value = ""
                elif aggfunc == 'sum':
                    aggfunc = np.sum
                    fill_value = ""

            df_pivot = df.pivot_table(values=values, index=index,
                                      columns=columns, aggfunc=aggfunc,
                                      fill_value=fill_value)
            if style == 'highlight_not_max':
                return df_pivot.style.apply(highlight_not_max)
            if style == 'heat_map':
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
            display style of scenarios failing the validation (heatmap, list)
        """
        df = pd.DataFrame()
        for var, check in criteria.items():
            df = df.append(self.check(var, check,
                                      filters, ret_true=False))
        if len(df):
            if exclude:
                idx = return_index(df, ['model', 'scenario'])
                self.cat.loc[idx, 'category'] = 'exclude'

            n = str(len(df))
            print(n + " scenarios do not satisfy the criteria")
            if display == 'heatmap':
                df.set_index(all_idx_cols, inplace=True)
                cm = sns.light_palette("green", as_cmap=True)
                return df.style.background_gradient(cmap=cm)
            elif display == 'list':
                return df
        else:
            print("All models and scenarios satisfy the criteria")

    def category(self, name=None, criteria=None, filters=None, comment=None,
                 assign=True, color=None, display=list):
        """Assign scenarios to a category according to specific criteria

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
            display style of scenarios assigned to this category (list, pivot)
            (no display if None)
        """
        # for returning a list or pivot table of all categories or one specific
        if not criteria:
            cat = self.cat.reset_index()
            if name:
                cat = cat[cat.category == name]
            if filters:
                for col in ['model', 'scenario']:
                    if col in filters:
                        cat = cat[keep_col_match(cat[col], filters[col])]

            if display == 'list':
                return pd.DataFrame(
                        index=return_index(cat,
                                           ['category', 'model', 'scenario']))
            elif display == 'pivot':
                cat = cat.pivot(index='model', columns='scenario',
                                values='category')
                return cat.style.apply(color_by_cat,
                                       cat_col=self.cat_color, axis=None)

        # when criteria are provided, use them to assign a new category
        else:
            # TODO clear out existing assignments to that category?
            cat = self.cat.index
            for var, check in criteria.items():
                cat = cat.intersection(self.check(var, check,
                                                  filters).index)

            df = pd.DataFrame(index=cat)
            if len(df):
                # assign selected model/scenario to internal category mapping
                if assign:
                    self.cat.loc[cat, 'category'] = name

                # assign a color to this category for pivot tables and plots
                if color:
                    self.cat_color[name] = color
                elif name not in self.cat_color:
                    self.cat_color[name] = sns.color_palette("hls",
                                                             8)[self.col_count]
                    self.col_count += 1

                # return the model/scenario as dataframe for visual output
                if display:
                    print("The following scenarios are categorized as '" +
                          name + "':")
                    if display == 'list':
                        return df
                    elif display == 'pivot':
                        return pivot_has_elements(df, 'model', 'scenario')
                else:
                    n = str(len(df))
                    print(n + " scenarios categorized as '" + name + "'")
            else:
                print("No scenario satisfies the criteria")

    def reset_category(self):
        """Reset category assignment for all scenarios to 'uncategorized'"""
        self.cat['category'] = 'uncategorized'

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
        ret_true: bool
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

    def select(self, filters={}, cols=None, idx_cols=None):
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
            Columns returned for the dataframe, duplicates are dropped
        idx_cols: string or list
            Columns that are set as index of the returned dataframe
        """

        # filter by columns and list of values
        keep = np.array([True] * len(self.data))

        for col, values in filters.items():
            if col == 'category':
                cat = self.cat[keep_col_match(self.cat['category'], values)]
                keep_col = self.data.set_index(['model', 'scenario'])\
                    .index.isin(cat.index)

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
        if cols:
            if idx_cols:
                cols = cols + idx_cols
            df = df[cols].drop_duplicates()

        # set (or reset) index
        if idx_cols:
            return df.set_index(idx_cols)
        else:
            return df.reset_index(drop=True)

    def plot_lines(self, filters={}, idx_cols=None, color_by_cat=False,
                   save=None, ret_ax=False):
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
        ret_ax: boolean, optional, default False
            return the 'axes()' object of the plot
        """
        if not idx_cols:
            idx_cols = iamc_idx_cols
        df = self.pivot_table(['year'], idx_cols, filters, values='value',
                              aggfunc=np.sum, style=None)
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
                elif title:
                    title = '{} - {}: {}'.format(title, col, level)
                else:
                    title = '{}: {}'.format(col, level)
            else:
                i += 1

        if color_by_cat:
            for (col, data) in df.iteritems():
                color = self.cat_color[self.cat.loc[col[0:2]].category]
                data.plot(color=color)
        else:
            for (col, data) in df.iteritems():
                data.plot()
            ax.legend(loc='best', framealpha=0.0)

        plt.title(title)
        plt.xlabel('Years')
        if save:
            plt.savefig(save)
        plt.show()

        if ret_ax:
            return ax

# %% auxiliary function for reading data from snapshot file


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
    if path:
        fname = '{}/{}.{}'.format(path, file, ext)
    else:
        fname = '{}.{}'.format(file, ext)

    if not os.path.exists(fname):
        raise SystemError("no snapshot file '" + fname + "' found!")

    # read from database snapshot csv
    if ext == 'csv':
        df = pd.read_csv(fname)
        df = (df.rename(columns={c: str(c).lower() for c in df.columns}))

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


def return_index(df, idx_cols):
    """set and return an index for a dataframe"""
    return df[idx_cols].set_index(idx_cols).index


def keep_col_match(col, strings, pseudo_regex=False):
    """
    matching of model/scenario names and variables to pseudo-regex (optional)
    for data filtering
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
