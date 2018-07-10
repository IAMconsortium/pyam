import copy
import importlib
import itertools
import os
import sys
import warnings

import numpy as np
import pandas as pd

try:
    import ixmp
    has_ix = True
except ImportError:
    has_ix = False

from pyam import plotting

from pyam.logger import logger
from pyam.run_control import run_control
from pyam.utils import (
    write_sheet,
    read_ix,
    read_files,
    read_pandas,
    format_data,
    pattern_match,
    years_match,
    isstr,
    islistable,
    META_IDX,
    YEAR_IDX,
    IAMC_IDX,
    SORT_IDX,
    LONG_IDX,
)
from pyam.timeseries import fill_series


class IamDataFrame(object):
    """This class is a wrapper for dataframes following the IAMC format.
    It provides a number of diagnostic features (including validation of data,
    completeness of variables provided) as well as a number of visualization
    and plotting tools.
    """

    def __init__(self, data, **kwargs):
        """Initialize an instance of an IamDataFrame

        Parameters
        ----------
        data: ixmp.TimeSeries, ixmp.Scenario, pd.DataFrame or data file
            an instance of an TimeSeries or Scenario (requires `ixmp`),
            or pd.DataFrame or data file with IAMC-format data columns.

            Special support is provided for data files downloaded directly from
            IIASA SSP and RCP databases. If you run into any problems loading
            data, please make an issue at:
            https://github.com/IAMconsortium/pyam/issues
        """
        # import data from pd.DataFrame or read from source
        if isinstance(data, pd.DataFrame):
            self.data = format_data(data.copy())
        elif has_ix and isinstance(data, ixmp.TimeSeries):
            self.data = read_ix(data, **kwargs)
        else:
            self.data = read_files(data, **kwargs)

        # define a dataframe for categorization and other metadata indicators
        self.meta = self.data[META_IDX].drop_duplicates().set_index(META_IDX)
        self.reset_exclude()

        # execute user-defined code
        if 'exec' in run_control():
            self._execute_run_control()

    def __getitem__(self, key):
        _key_check = [key] if isstr(key) else key
        if set(_key_check).issubset(self.meta.columns):
            return self.meta.__getitem__(key)
        else:
            return self.data.__getitem__(key)

    def __setitem__(self, key, value):
        _key_check = [key] if isstr(key) else key
        if set(_key_check).issubset(self.meta.columns):
            return self.meta.__setitem__(key, value)
        else:
            return self.data.__setitem__(key, value)

    def __len__(self):
        return self.data.__len__()

    def _execute_run_control(self):
        for module_block in run_control()['exec']:
            fname = module_block['file']
            functions = module_block['functions']

            dirname = os.path.dirname(fname)
            if dirname:
                sys.path.append(dirname)

            module = os.path.basename(fname).split('.')[0]
            mod = importlib.import_module(module)
            for func in functions:
                f = getattr(mod, func)
                f(self)

    def head(self, *args, **kwargs):
        """Identical to pd.DataFrame.head() operating on data"""
        return self.data.head(*args, **kwargs)

    def tail(self, *args, **kwargs):
        """Identical to pd.DataFrame.tail() operating on data"""
        return self.data.tail(*args, **kwargs)

    def models(self):
        """Get a list of models"""
        return pd.Series(self.meta.index.levels[0])

    def scenarios(self):
        """Get a list of scenarios"""
        return pd.Series(self.meta.index.levels[1])

    def regions(self):
        """Get a list of regions"""
        return pd.Series(self.data['region'].unique(), name='region')

    def variables(self, include_units=False):
        """Get a list of variables

        Parameters
        ----------
        include_units: boolean, default False
            include the units
        """
        if include_units:
            return self.data[['variable', 'unit']].drop_duplicates()\
                .reset_index(drop=True).sort_values('variable')
        else:
            return pd.Series(self.data.variable.unique(), name='variable')

    def append(self, other, inplace=False, **kwargs):
        """Import or read timeseries data and append to IamDataFrame

        Parameters
        ----------
        other: pyam.IamDataFrame, ixmp.TimeSeries, ixmp.Scenario,
        pd.DataFrame or data file
            an IamDataFrame, TimeSeries or Scenario (requires `ixmp`),
            or pd.DataFrame or data file with IAMC-format data columns
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
        aggfunc: str or function, default 'count'
            function used for aggregation,
            accepts 'count', 'mean', and 'sum'
        fill_value: scalar, default None
            value to replace missing values with
        style: str, default None
            output style for pivot table formatting
            accepts 'highlight_not_max', 'heatmap'
        """
        index = [index] if isstr(index) else index
        columns = [columns] if isstr(columns) else columns

        df = self.data

        # allow 'aggfunc' to be passed as string for easier user interface
        if isstr(aggfunc):
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
        """
        df = self.pivot_table(index=IAMC_IDX, columns=['year'],
                              values='value', aggfunc=np.sum)
        # drop year-rows where values are already defined
        if year in df.columns:
            df = df[np.isnan(df[year])]
        fill_values = df.apply(fill_series,
                               raw=False, axis=1, year=year)
        fill_values = fill_values.dropna().reset_index()
        fill_values = fill_values.rename(columns={0: "value"})
        fill_values['year'] = year
        self.data = self.data.append(fill_values, ignore_index=True)

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
        return (
            self.data
            .pivot_table(index=IAMC_IDX, columns='year')
            .value  # column name
            .rename_axis(None, axis=1)
        )

    def reset_exclude(self):
        """Reset exclusion assignment for all scenarios to `exclude: False`"""
        self.meta['exclude'] = False

    def set_meta(self, meta, name=None, index=None):
        """Add metadata columns as pd.Series or list

        Parameters
        ----------
        meta: pd.Series, list, int or str
            column to be added to metadata
            (by `['model', 'scenario']` index if possible)
        name: str
            meta column name (if not given by meta pd.Series.name)
        index: pyam.IamDataFrame or pd.MultiIndex, optional
            index to be used for setting meta column
        """
        if not name and not hasattr(meta, 'name'):
            raise ValueError('Must pass a name or use a pd.Series')

        # use meta.index if index arg is an IamDataFrame
        if index is not None and isinstance(index, IamDataFrame):
            index = index.meta.index

        # create pd.Series from index and meta if provided
        _meta = meta if index is None else pd.Series(data=meta, index=index,
                                                     name=name)

        try:
            append_by_idx = set(META_IDX).issubset(_meta.index.names)
        except AttributeError:
            append_by_idx = False

        # if not possible to append by index
        if not append_by_idx:
            if islistable(meta):
                self.meta[name] = list(meta)
            else:
                self.meta[name] = meta
            return  # EXIT FUNCTION

        # else append by index
        name = name or _meta.name

        # reduce index dimensions to model-scenario only
        _meta = (
            _meta
            .reset_index()
            .reindex(columns=META_IDX + [name])
            .set_index(META_IDX)
        )

        # raise error if index is not unique
        if _meta.index.duplicated().any():
            raise ValueError("non-unique ['model', 'scenario'] index!")

        # check if trying to add model-scenario index not existing in self
        diff = _meta.index.difference(self.meta.index)
        if not diff.empty:
            error = "adding metadata for non-existing scenarios '{}'!"
            raise ValueError(error.format(diff))

        self._new_meta_column(name, meta)
        self.meta[name] = _meta[name].combine_first(self.meta[name])

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
                run_control().update({kind: {name: {value: arg}}})

        # find all data that matches categorization
        rows = _apply_criteria(self.data, criteria,
                               in_range=True, return_test='all')
        idx = _meta_idx(rows)

        if len(idx) == 0:
            logger().info("No scenarios satisfy the criteria")
            return  # EXIT FUNCTION

        # update metadata dataframe
        self._new_meta_column(name, value)
        self.meta.loc[idx, name] = value
        msg = '{} scenario{} categorized as `{}: {}`'
        logger().info(msg.format(len(idx), '' if len(idx) == 1 else 's',
                                 name, value))

    def _new_meta_column(self, name, value):
        """Add a metadata column, set to `uncategorized` if str else np.nan"""
        if name is None:
            raise ValueError('cannot add a meta column {}'.format(name))
        if name not in self.meta:
            self.meta[name] = 'uncategorized' if isstr(value) else np.nan

    def require_variable(self, variable, unit=None, year=None,
                         exclude_on_fail=False):
        """Check whether all scenarios have a required variable

        Parameters
        ----------
        variable: str
            required variable
        unit: str, default None
            name of unit (optional)
        years: int or list, default None
            years (optional)
        exclude: bool, default False
            flag scenarios missing the required variables as `exclude: True`
        """
        criteria = {'variable': variable}
        if unit:
            criteria.update({'unit': unit})
        if year:
            criteria.update({'year': year})

        keep = _apply_filters(self.data, self.meta, criteria)
        idx = self.meta.index.difference(_meta_idx(self.data[keep]))

        n = len(idx)
        if n == 0:
            logger().info('All scenarios have the required variable `{}`'
                          .format(variable))
            return

        msg = '{} scenario does not include required variable `{}`' if n == 1 \
            else '{} scenarios do not include required variable `{}`'

        if exclude_on_fail:
            self.meta.loc[idx, 'exclude'] = True
            msg += ', marked as `exclude: True` in metadata'

        logger().info(msg.format(n, variable))
        return pd.DataFrame(index=idx).reset_index()

    def validate(self, criteria={}, exclude_on_fail=False):
        """Validate scenarios using criteria on timeseries values

        Parameters
        ----------
        criteria: dict
           dictionary with variable keys and check values
            ('up' and 'lo' for respective bounds, 'year' for years)
        exclude_on_fail: bool, default False
            flag scenarios failing validation as `exclude: True`
        """
        df = _apply_criteria(self.data, criteria, in_range=False)

        if not df.empty:
            msg = '{} of {} data points to not satisfy the criteria'
            logger().info(msg.format(len(df), len(self.data)))

            if exclude_on_fail and len(df) > 0:
                self._exclude_on_fail(df)

            return df

    def rename(self, mapping, inplace=False):
        """Rename and aggregate column entries using groupby.sum()

        Parameters
        ----------
        mapping: dict
            for each column where entries should be renamed, provide current
            name and target name
            {<column name>: {<current_name_1>: <target_name_1>,
                             <current_name_2>: <target_name_2>}}
        inplace: bool, default False
            if True, do operation inplace and return None
        """
        ret = copy.deepcopy(self) if not inplace else self
        for col in mapping:
            if col not in ['region', 'variable', 'unit']:
                raise ValueError('renaming by {} not supported!'.format(col))
            ret.data.loc[:, col] = self.data.loc[:, col].replace(mapping[col])

        ret.data = ret.data.groupby(LONG_IDX).sum().reset_index()
        if not inplace:
            return ret

    def convert_unit(self, conversion_mapping, inplace=False):
        """Converts units based on provided unit conversion factors

        Parameters
        ----------
        conversion_mapping: dict
            for each unit for which a conversion should be carried out,
            provide current unit and target unit and conversion factor
            {<current unit>: [<target unit>, <conversion factor>]}
        inplace: bool, default False
            if True, do operation inplace and return None
        """
        ret = copy.deepcopy(self) if not inplace else self
        for current_unit, (new_unit, factor) in conversion_mapping.items():
            factor = pd.to_numeric(factor)
            where = ret.data['unit'] == current_unit
            ret.data.loc[where, 'value'] *= factor
            ret.data.loc[where, 'unit'] = new_unit
        if not inplace:
            return ret

    def check_aggregate(self, variable, components=None, units=None,
                        exclude_on_fail=False, multiplier=1, **kwargs):
        """Check whether the timeseries data match the aggregation
        of components or sub-categories

        Parameters
        ----------
        variable: str
            variable to be checked for matching aggregation of sub-categories
        components: list of str, default None
            list of variables, defaults to all sub-categories of `variable`
        units: str or list of str, default None
            filter variable and components for given unit(s)
        exclude_on_fail: boolean, default False
            flag scenarios failing validation as `exclude: True`
        multiplier: number, default 1
            factor when comparing variable and sum of components
        kwargs: passed to `np.isclose()`
        """
        # default components to all variables one level below `variable`
        if components is None:
            components = self.filter(variable='{}|*'.format(variable),
                                     level=0).variables()

        # filter and groupby data, use `pd.Series.align` for machting index
        df_variable, df_components = (
            _aggregate_by_variables(self.data, variable, units)
            .align(_aggregate_by_variables(self.data, components, units))
        )

        # use `np.isclose` for checking match
        diff = df_variable[~np.isclose(df_variable, multiplier * df_components,
                                       **kwargs)]

        if len(diff):
            msg = '{} of {} data points are not aggregates of components'
            logger().info(msg.format(len(diff), len(df_variable)))

            if exclude_on_fail:
                self._exclude_on_fail(diff.index.droplevel([2, 3]))

            return diff.unstack().rename_axis(None, axis=1)

    def _exclude_on_fail(self, df):
        """Assign a selection of scenarios as `exclude: True` in meta"""
        idx = df if isinstance(df, pd.MultiIndex) else _meta_idx(df)
        self.meta.loc[idx, 'exclude'] = True
        logger().info('{} non-valid scenario{} will be excluded'
                      .format(len(idx), '' if len(idx) == 1 else 's'))

    def filter(self, filters=None, keep=True, inplace=False, **kwargs):
        """Return a filtered IamDataFrame (i.e., a subset of current data)

        Parameters
        ----------
        keep: bool, default True
            keep all scenarios satisfying the filters (if True) or the inverse
        inplace: bool, default False
            if True, do operation inplace and return None
        filters by kwargs or dict (deprecated):
            The following columns are available for filtering:
             - metadata columns: filter by category assignment in metadata
             - 'model', 'scenario', 'region', 'variable', 'unit':
               string or list of strings, where ``*`` can be used as a wildcard
             - 'level': the maximum "depth" of IAM variables (number of '|')
               (exluding the strings given in the 'variable' argument)
             - 'year': takes an integer, a list of integers or a range
                note that the last year of a range is not included,
                so ``range(2010,2015)`` is interpreted as ``[2010, ..., 2014]``
        """
        if filters is not None:
            warnings.warn(
                '`filters` keyword argument in filters() is deprecated and will be removed in the next release')
            kwargs.update(filters)

        _keep = _apply_filters(self.data, self.meta, kwargs)
        _keep = _keep if keep else ~_keep
        ret = copy.deepcopy(self) if not inplace else self
        ret.data = ret.data[_keep]

        idx = pd.MultiIndex.from_tuples(
            pd.unique(list(zip(ret.data['model'], ret.data['scenario']))),
            names=('model', 'scenario')
        )
        if len(idx) == 0:
            logger().warning('Filtered IamDataFrame is empty!')

        ret.meta = ret.meta.loc[idx]
        if not inplace:
            return ret

    def col_apply(self, col, func, *args, **kwargs):
        """Apply a function to a column

        Parameters
        ----------
        col: string
            column in either data or metadata
        func: functional
            function to apply
        """
        if col in self.data:
            self.data[col] = self.data[col].apply(func, *args, **kwargs)
        else:
            self.meta[col] = self.meta[col].apply(func, *args, **kwargs)

    def _to_file_format(self):
        """Return a dataframe suitable for writing to a file"""
        df = self.timeseries().reset_index()
        df = df.rename(columns={c: str(c).title() for c in df.columns})
        return df

    def to_csv(self, path, index=False, **kwargs):
        """Write data to a csv file

        Parameters
        ----------
        index: boolean, default False
            write row names (index)
        """
        self._to_file_format().to_csv(path, index=False, **kwargs)

    def to_excel(self, path=None, writer=None, sheet_name='data', index=False,
                 **kwargs):
        """Write timeseries data to Excel using the IAMC template convention
        (wrapper for `pd.DataFrame.to_excel()`)

        Parameters
        ----------
        excel_writer: string or ExcelWriter object
             file path or existing ExcelWriter
        sheet_name: string, default 'data'
            name of the sheet that will contain the (filtered) IamDataFrame
        index: boolean, default False
            write row names (index)
        """
        if (path is None and writer is None) or \
           (path is not None and writer is not None):
            raise ValueError('Only one of path and writer must have a value')
        if writer is None:
            writer = pd.ExcelWriter(path)
        self._to_file_format().to_excel(writer, sheet_name=sheet_name,
                                        index=index, **kwargs)

    def export_metadata(self, path):
        """Export metadata to Excel

        Parameters
        ----------
        path: string
            path/filename for xlsx file of metadata export
        """
        writer = pd.ExcelWriter(path)
        write_sheet(writer, 'metadata', self.meta, index=True)
        writer.save()

    def load_metadata(self, path, *args, **kwargs):
        """Load metadata from previously exported instance of pyam

        Parameters
        ----------
        path: string
            xlsx file with metadata exported from an instance of pyam
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

    def line_plot(self, x='year', y='value', **kwargs):
        """Plot timeseries lines of existing data

        see pyam.plotting.line_plot() for all available options
        """
        df = self.as_pandas(with_metadata=True)

        # pivot data if asked for explicit variable name
        variables = df['variable'].unique()
        if x in variables or y in variables:
            keep_vars = set([x, y]) & set(variables)
            df = df[df['variable'].isin(keep_vars)]
            idx = list(set(df.columns) - set(['value']))
            df = (df
                  .reset_index()
                  .set_index(idx)
                  .value  # df -> series
                  .unstack(level='variable')  # keep_vars are columns
                  .rename_axis(None, axis=1)  # rm column index name
                  .reset_index()
                  .set_index(META_IDX)
                  )
            if x != 'year' and y != 'year':
                df = df.drop('year', axis=1)  # years causes NaNs

        ax, handles, labels = plotting.line_plot(df, x=x, y=y, **kwargs)
        return ax

    def stack_plot(self, *args, **kwargs):
        """Plot timeseries stacks of existing data

        see pyam.plotting.stack_plot() for all available options
        """
        df = self.as_pandas(with_metadata=True)
        ax = plotting.stack_plot(df, *args, **kwargs)
        return ax

    def bar_plot(self, *args, **kwargs):
        """Plot timeseries bars of existing data

        see pyam.plotting.bar_plot() for all available options
        """
        df = self.as_pandas(with_metadata=True)
        ax = plotting.bar_plot(df, *args, **kwargs)
        return ax

    def pie_plot(self, *args, **kwargs):
        """Plot a pie chart

        see pyam.plotting.pie_plot() for all available options
        """
        df = self.as_pandas(with_metadata=True)
        ax = plotting.pie_plot(df, *args, **kwargs)
        return ax

    def map_regions(self, map_col, agg=None, copy_col=None, fname=None,
                    region_col=None, inplace=False):
        """Plot regional data for a single model, scenario, variable, and year

        see pyam.plotting.region_plot() for all available options

        Parameters
        ----------
        map_col: string
            The column used to map new regions to. Common examples include
            iso and 5_region.
        agg: string, optional
            Perform a data aggregation. Options include: sum.
        copy_col: string, optional
            Copy the existing region data into a new column for later use.
        fname: string, optional
            Use a non-default region mapping file
        region_col: string, optional
            Use a non-default column name for regions to map from.
        inplace : bool, default False
            if True, do operation inplace and return None
        """
        models = self.meta.index.get_level_values('model').unique()
        fname = fname or run_control()['region_mapping']['default']
        mapping = read_pandas(fname).rename(str.lower, axis='columns')
        map_col = map_col.lower()

        ret = copy.deepcopy(self) if not inplace else self
        _df = ret.data
        columns_orderd = _df.columns

        # merge data
        dfs = []
        for model in models:
            df = _df[_df['model'] == model]
            _col = region_col or '{}.REGION'.format(model)
            _map = mapping.rename(columns={_col.lower(): 'region'})
            _map = _map[['region', map_col]].dropna().drop_duplicates()

            if copy_col is not None:
                df[copy_col] = df['region']

            df = (df
                  .merge(_map, on='region')
                  .drop('region', axis=1)
                  .rename(columns={map_col: 'region'})
                  )
            dfs.append(df)
        df = pd.concat(dfs)

        # perform aggregations
        if agg == 'sum':
            df = df.groupby(LONG_IDX).sum().reset_index()

        ret.data = (df
                    .reindex(columns=columns_orderd)
                    .sort_values(SORT_IDX)
                    .reset_index(drop=True)
                    )
        if not inplace:
            return ret

    def region_plot(self, **kwargs):
        """Plot regional data for a single model, scenario, variable, and year

        see pyam.plotting.region_plot() for all available options
        """
        df = self.as_pandas(with_metadata=True)
        ax = plotting.region_plot(df, **kwargs)
        return ax


def _meta_idx(data):
    return data[META_IDX].drop_duplicates().set_index(META_IDX).index


def _aggregate_by_variables(df, variables, units=None):
    variables = [variables] if isstr(variables) else variables
    df = df[df.variable.isin(variables)]

    if units is not None:
        units = [units] if isstr(units) else units
        df = df[df.unit.isin(units)]

    return df.groupby(YEAR_IDX).sum()['value']


def _apply_filters(data, meta, filters):
    keep = np.array([True] * len(data))

    # filter by columns and list of values
    for col, values in filters.items():
        if col in meta.columns:
            matches = pattern_match(meta[col], values)
            cat_idx = meta[matches].index
            keep_col = data[META_IDX].set_index(META_IDX).index.isin(cat_idx)

        elif col in ['model', 'scenario', 'region', 'unit']:
            keep_col = pattern_match(data[col], values)

        elif col == 'variable':
            level = filters['level'] if 'level' in filters.keys() else None
            keep_col = pattern_match(data[col], values, level)

        elif col == 'year':
            keep_col = years_match(data[col], values)

        elif col == 'level':
            if 'variable' not in filters.keys():
                keep_col = pattern_match(data['variable'], '*', level=values)
            else:
                continue
        else:
            raise ValueError(
                'filter by column ' + col + ' not supported')
        keep &= keep_col

    return keep


def _check_rows(rows, check, in_range=True, return_test='any'):
    """Check all rows to be in/out of a certain range and provide testing on
    return values based on provided conditions

    Parameters
    ----------
    rows: pd.DataFrame
        data rows
    check: dict
        dictionary with possible values of "up", "lo", and "year"
    in_range: bool, optional
        check if values are inside or outside of provided range
    return_test: str, optional
        possible values:
            - 'any': default, return scenarios where check passes for any entry
            - 'all': test if all values match checks, if not, return empty set
    """
    valid_checks = set(['up', 'lo', 'year'])
    if not set(check.keys()).issubset(valid_checks):
        msg = 'Unknown checking type: {}'
        raise ValueError(msg.format(check.keys() - valid_checks))

    where_idx = set(rows.index[rows['year'] == check['year']]) \
        if 'year' in check else set(rows.index)
    rows = rows.loc[list(where_idx)]

    up_op = rows['value'].__le__ if in_range else rows['value'].__gt__
    lo_op = rows['value'].__ge__ if in_range else rows['value'].__lt__

    check_idx = []
    for (bd, op) in [('up', up_op), ('lo', lo_op)]:
        if bd in check:
            check_idx.append(set(rows.index[op(check[bd])]))

    if return_test is 'any':
        ret = where_idx & set.union(*check_idx)
    elif return_test == 'all':
        ret = where_idx if where_idx == set.intersection(*check_idx) else set()
    else:
        raise ValueError('Unknown return test: {}'.format(return_test))
    return ret


def _apply_criteria(df, criteria, **kwargs):
    """Apply criteria individually to every model/scenario instance"""
    idxs = []
    for var, check in criteria.items():
        _df = df[df['variable'] == var]
        for group in _df.groupby(META_IDX):
            grp_idxs = _check_rows(group[-1], check, **kwargs)
            idxs.append(grp_idxs)
    df = df.loc[itertools.chain(*idxs)]
    return df


def validate(df, criteria={}, exclude_on_fail=False, **kwargs):
    """Validate scenarios using criteria on timeseries values

    Parameters
    ----------
    df: IamDataFrame instance
    args: see `IamDataFrame.validate()` for details
    kwargs: passed to `df.filter()`
    """
    fdf = df.filter(**kwargs)
    if len(fdf.data) > 0:
        vdf = fdf.validate(criteria=criteria, exclude_on_fail=exclude_on_fail)
        df.meta['exclude'] |= fdf.meta['exclude']  # update if any excluded
        return vdf


def require_variable(df, variable, unit=None, year=None, exclude_on_fail=False,
                     **kwargs):
    """Check whether all scenarios have a required variable

    Parameters
    ----------
    df: IamDataFrame instance
    args: see `IamDataFrame.require_variable()` for details
    kwargs: passed to `df.filter()`
    """
    fdf = df.filter(**kwargs)
    if len(fdf.data) > 0:
        vdf = fdf.require_variable(variable=variable, unit=unit, year=year,
                                   exclude_on_fail=exclude_on_fail)
        df.meta['exclude'] |= fdf.meta['exclude']  # update if any excluded
        return vdf


def categorize(df, name, value, criteria,
               color=None, marker=None, linestyle=None, **kwargs):
    """Assign scenarios to a category according to specific criteria
    or display the category assignment

    Parameters
    ----------
    df: IamDataFrame instance
    args: see `IamDataFrame.categorize()` for details
    kwargs: passed to `df.filter()`
    """
    fdf = df.filter(**kwargs)
    fdf.categorize(name=name, value=value, criteria=criteria, color=color,
                   marker=marker, linestyle=linestyle)

    # update metadata
    if name in df.meta:
        df.meta[name].update(fdf.meta[name])
    else:
        df.meta[name] = fdf.meta[name]


def check_aggregate(df, variable, components=None, units=None,
                    exclude_on_fail=False, multiplier=1, **kwargs):
    """Check whether the timeseries values match the aggregation
    of sub-categories

    Parameters
    ----------
    df: IamDataFrame instance
    args: see IamDataFrame.check_aggregate() for details
    kwargs: passed to `df.filter()`
    """
    fdf = df.filter(**kwargs)
    if len(fdf.data) > 0:
        vdf = fdf.check_aggregate(variable=variable, components=components,
                                  units=units, exclude_on_fail=exclude_on_fail,
                                  multiplier=multiplier)
        df.meta['exclude'] |= fdf.meta['exclude']  # update if any excluded
        return vdf


def filter_by_meta(data, df, join_meta=False, **kwargs):
    """Filter by and join meta columns from an IamDataFrame to a pd.DataFrame

    Parameters
    ----------
    data: pd.DataFrame instance
        DataFrame to which meta columns are to be joined,
        index or columns must include `['model', 'scenario']`
    df: IamDataFrame instance
        IamDataFrame from which meta columns are filtered and joined (optional)
    join_meta: bool, default False
        join selected columns from `df.meta` on `data`
    kwargs:
        meta columns to be joined, where `col=...` applies filters
        by the given arguments (using `utils.pattern_match()`) and `col=None`
        joins the column without filtering
    """
    if not set(META_IDX).issubset(data.index.names + list(data.columns)):
        raise ValueError('missing required index dimensions or columns!')

    meta = df.meta[list(kwargs)].copy()

    # filter meta by columns
    keep = np.array([True] * len(meta))
    for col, values in kwargs.items():
        if values is not None:
            keep_col = pattern_match(meta[col], values)
            keep &= keep_col
    meta = meta[keep]

    # set the data index to META_IDX and apply filtered meta index
    data = data.copy()
    idx = list(data.index.names) if not data.index.names == [None] else None
    data = data.reset_index().set_index(META_IDX).loc[meta.index]

    # join meta (optional), reset index to format as input arg
    data = data.join(meta) if join_meta else data
    data = data.reset_index().set_index(idx or 'index')
    if idx is None:
        data.index.name = None

    return data
