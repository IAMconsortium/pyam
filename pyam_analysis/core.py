import copy
import os
import six
import itertools

import numpy as np
import pandas as pd

try:
    import ixmp
    has_ix = True
except ImportError:
    has_ix = False

from pyam_analysis import plotting

from pyam_analysis.logger import logger
from pyam_analysis.run_control import run_control
from pyam_analysis.utils import (
    isstr,
    write_sheet,
    read_ix,
    read_files,
    read_pandas,
    format_data,
    pattern_match,
    years_match,
    isstr,
    META_IDX,
    IAMC_IDX
)
from pyam_analysis.timeseries import fill_series


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
        data: ixmp.TimeSeries, ixmp.Scenario, pd.DataFrame or data file
            an instance of an TimeSeries or Scenario (requires `ixmp`),
            or pd.DataFrame or data file with IAMC-format data columns
        """
        # import data from pd.DataFrame or read from source
        if isinstance(data, pd.DataFrame):
            self.data = format_data(data.copy())
        elif has_ix and isinstance(data, ixmp.TimeSeries):
            self.data = read_ix(data, **kwargs)
        else:
            self.data = read_files(data, **kwargs)

        # define a dataframe for categorization and other meta-data
        self.meta = self.data[META_IDX].drop_duplicates().set_index(META_IDX)
        self.reset_exclude()

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

    def head(self, *args, **kwargs):
        """Identical to pd.DataFrame.head() operating on data"""
        return self.data.head(*args, **kwargs)

    def tail(self, *args, **kwargs):
        """Identical to pd.DataFrame.tail() operating on data"""
        return self.data.tail(*args, **kwargs)

    def append(self, other, inplace=False, **kwargs):
        """Import or read timeseries data and append to IamDataFrame

        Parameters
        ----------
        other: pyam-analysis.IamDataFrame, ixmp.TimeSeries, ixmp.Scenario,
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
        """Reset exclusion assignment for all scenarios to `exclude: False`
        """
        self.meta['exclude'] = False

    def metadata(self, meta, name=None):
        """Add metadata columns as pd.Series or list

        Parameters
        ----------
        meta: pd.Series, list, int or str
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
            meta = meta.to_frame(name or meta.name)
            self.meta = meta.combine_first(self.meta)
            return  # EXIT FUNCTION

        if isinstance(meta, pd.Series):
            name = name or meta.name
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
                run_control().update({kind: {name: {value: arg}}})

        if criteria == 'uncategorized':
            self.meta[name].fillna(value, inplace=True)
            msg = "{} of {} scenarios are uncategorized."
            logger().info(msg.format(np.sum(self.meta[name] == value),
                                     len(self.meta)))
            return  # EXIT FUNCTION

        # find all data that matches categorization
        rows = _apply_criteria(self.data, criteria,
                               in_range=True, return_test='all')
        idx = _meta_idx(rows)

        # update metadata dataframe
        if name not in self.meta:
            self.meta[name] = np.nan
        if len(idx) == 0:
            logger().info("No scenarios satisfy the criteria")
        else:
            self.meta.loc[idx, name] = value
            msg = "{} scenario{} categorized as {}: '{}'"
            logger().info(msg.format(len(idx), '' if len(idx) == 1 else 's',
                                     name, value))

    def require_variable(self, variable, unit=None, year=None, exclude=False):
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

        if len(idx) == 0:
            logger().info('All scenarios have the required variable')
            return

        msg = '{} scenario{} to not include required variables'

        if exclude:
            self.meta.loc[idx, 'exclude'] = True
            msg += ', marked as `exclude=True` in metadata'

        logger().info(msg.format(len(idx), '' if len(idx) == 1 else 's'))
        return pd.DataFrame(index=idx).reset_index()

    def validate(self, criteria={}, exclude=False):
        """Validate scenarios using criteria on timeseries values

        Parameters
        ----------
        criteria: dict
           dictionary with variable keys and check values
            ('up' and 'lo' for respective bounds, 'year' for years)
        exclude: bool, default False
            flag scenarios failing validation as `exclude: True`
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

    def filter(self, filters, keep=True, inplace=False):
        """Return a filtered IamDataFrame (i.e., a subset of current data)

        Parameters
        ----------
        filters: dict
            The following columns are available for filtering:
             - metadata columns: filter by category assignment in metadata
             - 'model', 'scenario', 'region', 'variable', 'unit':
               string or list of strings, where ``*`` can be used as a wildcard
             - 'level': the maximum "depth" of IAM variables (number of '|')
               (exluding the strings given in the 'variable' argument)
             - 'year': takes an integer, a list of integers or a range
                note that the last year of a range is not included,
                so ``range(2010,2015)`` is interpreted as ``[2010, ..., 2014]``
        inplace : bool, default False
            if True, do operation inplace and return None
        """
        _keep = _apply_filters(self.data, self.meta, filters)
        _keep = _keep if keep else ~_keep
        ret = copy.deepcopy(self) if not inplace else self
        ret.data = ret.data[_keep]

        idx = pd.MultiIndex.from_tuples(
            pd.unique(list(zip(ret.data['model'], ret.data['scenario']))),
            names=('model', 'scenario')
        )
        ret.meta = ret.meta.loc[idx]
        if not inplace:
            return ret

    def col_apply(self, col, func):
        self.data[col] = self.data[col].apply(func)

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

    def to_excel(self, path=None, writer=None, sheet_name='data', index=False, **kwargs):
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
        self._to_file_format().to_excel(writer, sheet_name=sheet_name, index=index, **kwargs)

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
        """Load metadata from previously exported instance of pyam_analysis

        Parameters
        ----------
        path: string
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

    def bar_plot(self, *args, **kwargs):
        """Plot timeseries bars of existing data

        see pyam_analysis.plotting.bar_plot() for all available options
        """
        df = self.as_pandas(with_metadata=True)
        ax = plotting.bar_plot(df, *args, **kwargs)
        return ax

    def pie_plot(self, *args, **kwargs):
        """Plot a pie chart

        see pyam_analysis.plotting.pie_plot() for all available options
        """
        df = self.as_pandas(with_metadata=True)
        ax = plotting.pie_plot(df, *args, **kwargs)
        return ax

    def region_plot(self, map_regions=False, map_col='iso', **kwargs):
        """Plot regional data for a single model, scenario, variable, and year

        see pyam_analysis.plotting.region_plot() for all available options

        Parameters
        ----------
        map_regions: boolean or string, default False
            Apply a mapping from existing regions to regions to plot. If True, 
            the mapping will be searched in known locations (e.g., if 
            registered with `run_control()`). If a path to a file is provided,
            that file will be used. Files must have a "region" column of 
            existing regions and a mapping column of regions to be mapped to.
        map_col: string, default 'iso'
            The column used to map new regions to. 
        """
        df = self.as_pandas(with_metadata=True)
        if map_regions:
            if map_regions is True:
                model = df['model'].unique()[0]
                fname = run_control()['region_mapping'][model]
            elif os.path.exists(map_regions):
                fname = map_regions
            else:
                raise ValueError(
                    'Unknown region mapping: {}'.format(map_regions))

            mapping = read_pandas(fname)
            df = (df
                  .merge(mapping, on='region')
                  .rename(columns={map_col: 'region', 'region': 'label'})
                  )

        ax = plotting.region_plot(df, **kwargs)
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
            raise SystemError(
                'filter by column ' + col + ' not supported')
        keep &= keep_col

    return keep


def _check_rows(rows, check, in_range=True, return_test='any'):
    """Check all rows to be in/out of a certain range and provide testing on return
    values based on provided conditions

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
    check_idx = set.intersection(*check_idx)

    if return_test is 'any':
        ret = where_idx & check_idx
    elif return_test == 'all':
        ret = where_idx if where_idx == check_idx else set()
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


def validate(df, *args, **kwargs):
    """Validate scenarios using criteria on timeseries values

    Parameters
    ----------
    df: IamDataFrame instance
    args and kwargs: see IamDataFrame.validate() for details
    filters: dict, optional
        filter by data & metadata columns, see function 'filter()' for details,
        filtering by 'variable'/'year' is replaced by arguments of 'criteria'
    """
    filters = kwargs.pop('filters', {})
    fdf = df.filter(filters)
    vdf = fdf.validate(*args, **kwargs)
    df.meta['exclude'] |= fdf.meta['exclude']  # update if any excluded
    return vdf


def require_variable(df, *args, **kwargs):
    """Check whether all scenarios have a required variable

    Parameters
    ----------
    df: IamDataFrame instance
    args and kwargs: see IamDataFrame.require_variable() for details
    filters: dict, optional
        filter by data & metadata columns, see function 'filter()' for details
    """

    filters = kwargs.pop('filters', {})
    fdf = df.filter(filters)
    vdf = fdf.require_variable(*args, **kwargs)
    df.meta['exclude'] |= fdf.meta['exclude']  # update if any excluded
    return vdf


def categorize(df, *args, **kwargs):
    """Assign scenarios to a category according to specific criteria
    or display the category assignment

    Parameters
    ----------
    df: IamDataFrame instance
    args and kwargs: see IamDataFrame.categorize() for details
    filters: dict, optional
        filter by data & metadata columns, see function 'filter()' for details,
        filtering by 'variable'/'year' is replaced by arguments of 'criteria'
    """
    filters = kwargs.pop('filters', {})
    fdf = df.filter(filters)
    fdf.categorize(*args, **kwargs)

    # update metadata
    name = args[0] if len(args) else kwargs['name']
    if name in df.meta:
        df.meta[name].update(fdf.meta[name])
    else:
        df.meta[name] = fdf.meta[name]
