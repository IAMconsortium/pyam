import copy
import importlib
import itertools
import logging
import os
import sys

import numpy as np
import pandas as pd

from pathlib import Path
from tempfile import TemporaryDirectory

try:
    from datapackage import Package
    HAS_DATAPACKAGE = True
except ImportError:
    Package = None
    HAS_DATAPACKAGE = False

try:
    import ixmp
    ixmp.TimeSeries
    has_ix = True
except (ImportError, AttributeError):
    has_ix = False

from pyam import plotting
from pyam.run_control import run_control
from pyam.utils import (
    write_sheet,
    read_file,
    read_pandas,
    format_data,
    format_time_col,
    merge_meta,
    get_keep_col,
    find_depth,
    pattern_match,
    years_match,
    month_match,
    hour_match,
    day_match,
    datetime_match,
    isstr,
    islistable,
    print_list,
    META_IDX,
    YEAR_IDX,
    IAMC_IDX,
    SORT_IDX,
    ILLEGAL_COLS
)
from pyam.read_ixmp import read_ix
from pyam.timeseries import fill_series
from pyam.plotting import mpl_args_to_meta_cols
from pyam._aggregate import _aggregate, _aggregate_region, _aggregate_time,\
    _aggregate_recursive, _group_and_agg
from pyam.units import convert_unit
from pyam.index import get_index_levels, append_index_level
from pyam.logging import deprecation_warning

logger = logging.getLogger(__name__)


class IamDataFrame(object):
    """Scenario timeseries data following the IAMC data format

    The class provides a number of diagnostic features (including validation of
    data, completeness of variables provided), processing tools (e.g.,
    unit conversion), as well as visualization and plotting tools.

    Parameters
    ----------
    data : :class:`pandas.DataFrame`, :class:`ixmp.Scenario`,
            or file-like object as str or :class:`pathlib.Path`
        Scenario timeseries data following the IAMC data format or
        a supported variation as pandas object, a path to a file,
        or a scenario of an ixmp instance.
    meta : :class:`pandas.DataFrame`, optional
        A dataframe with suitable 'meta' indicators for the new instance.
        The index will be downselected to scenarios present in `data`.
    kwargs
        If `value=<col>`, melt column `<col>` to 'value' and use `<col>` name
        as 'variable'; or mapping of required columns (:code:`IAMC_IDX`) to
        any of the following:

        - one column in `data`
        - multiple columns, to be concatenated by :code:`|`
        - a string to be used as value for this column

    Notes
    -----
    A :class:`pandas.DataFrame` can have the required dimensions
    as columns or index.
    R-style integer column headers (i.e., `X2015`) are acceptable.

    When initializing an :class:`IamDataFrame` from an xlsx file,
    |pyam| will per default look for the sheets 'data' and 'meta' to
    populate the respective tables. Custom sheet names can be specified with
    kwargs :code:`sheet_name` ('data') and :code:`meta_sheet_name` ('meta').
    Calling the class with :code:`meta_sheet_name=False` will
    skip the import of the 'meta' table.

    When initializing an :class:`IamDataFrame` from an object that is already
    an :class:`IamDataFrame` instance, the new object will be hard-linked to
    all attributes of the original object - so any changes on one object
    (e.g., with :code:`inplace=True`) may also modify the other object!
    This is intended behaviour and consistent with pandas but may be confusing
    for those who are not used to the pandas/Python universe.
    """
    def __init__(self, data, meta=None, **kwargs):
        """Initialize an instance of an IamDataFrame"""
        if isinstance(data, IamDataFrame):
            if kwargs:
                msg = 'Invalid arguments `{}` for initializing an IamDataFrame'
                raise ValueError(msg.format(kwargs))
            for attr, value in data.__dict__.items():
                setattr(self, attr, value)
        else:
            self._init(data, meta, **kwargs)

    def _init(self, data, meta=None, **kwargs):
        """Process data and set attributes for new instance"""
        # pop kwarg for meta_sheet_name (prior to reading data from file)
        meta_sheet = kwargs.pop('meta_sheet_name', 'meta')

        # cast data from pandas
        if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
            _data = format_data(data.copy(), **kwargs)
        # read data from ixmp Platform instance
        elif has_ix and isinstance(data, ixmp.TimeSeries):
            # TODO read meta indicators from ixmp
            _data = read_ix(data, **kwargs)
        else:
            if islistable(data):
                raise ValueError(
                    'Initializing from list is not supported, '
                    'use `IamDataFrame.append()` or `pyam.concat()`'
                )
            # read from file
            try:
                data = Path(data)  # casting str or LocalPath to Path
                is_file = data.is_file()
            except TypeError:  # `data` cannot be cast to Path
                is_file = False

            if is_file:
                logger.info('Reading file `{}`'.format(data))
                _data = read_file(data, **kwargs)
            # if not a readable file...
            else:
                msg = 'IamDataFrame constructor not properly called!'
                raise ValueError(msg)

        _df, self.time_col, self.extra_cols = _data
        self._LONG_IDX = IAMC_IDX + [self.time_col] + self.extra_cols
        # cast time_col to desired format
        self._data = _df.set_index(self._LONG_IDX).value

        # define `meta` dataframe for categorization & quantitative indicators
        self.meta = pd.DataFrame(index=_make_index(self._data))
        self.reset_exclude()

        # merge meta dataframe (if given in kwargs)
        if meta is not None:
            self.meta = merge_meta(meta.loc[_make_index(self.data)],
                                   self.meta, ignore_meta_conflict=True)

        # if initializing from xlsx, try to load `meta` table from file
        if meta_sheet and isinstance(data, Path) and data.suffix == '.xlsx':
            excel_file = pd.ExcelFile(data)
            if meta_sheet in excel_file.sheet_names:
                self.load_meta(excel_file, sheet_name=meta_sheet)

        # add time domain and extra-cols as attributes
        if self.time_col == 'year':
            setattr(self, 'year', get_index_levels(self._data, 'year'))
        else:
            setattr(self, 'time', pd.Index(
                get_index_levels(self._data, 'time')))
        for c in self.extra_cols:
            setattr(self, c, get_index_levels(self._data, c))

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
        deprecation_warning('Please use `set_meta` or `rename`.',
                            'Item assignment')
        _key_check = [key] if isstr(key) else key
        if set(_key_check).issubset(self.meta.columns):
            self.meta.__setitem__(key, value)
        else:
            df = self.data
            df.__setitem__(key, value)
            self.data = df

    def __len__(self):
        return self.data.__len__()

    def __repr__(self):
        return self.info()

    def info(self, n=80, meta_rows=5, memory_usage=False):
        """Print a summary of the object index dimensions and meta indicators

        Parameters
        ----------
        n : int
            The maximum line length
        meta_rows : int
            The maximum number of meta indicators printed
        """
        # concatenate list of index dimensions and levels
        info = f'{type(self)}\nIndex dimensions:\n'
        c1 = max([len(i) for i in self._LONG_IDX]) + 1
        c2 = n - c1 - 5
        info += '\n'.join(
            [f' * {i:{c1}}: {print_list(get_index_levels(self._data, i), c2)}'
             for i in META_IDX])

        # concatenate list of index of _data (not in META_IDX)
        info += '\nTimeseries data coordinates:\n'
        info += '\n'.join(
            [f'   {i:{c1}}: {print_list(get_index_levels(self._data, i), c2)}'
             for i in self._LONG_IDX if i not in META_IDX])

        # concatenate list of (head of) meta indicators and levels/values
        def print_meta_row(m, t, lst):
            _lst = print_list(lst, n - len(m) - len(t) - 7)
            return f'   {m} ({t}) {_lst}'

        info += '\nMeta indicators:\n'
        info += '\n'.join(
            [print_meta_row(m, t, self.meta[m].unique())
             for m, t in zip(self.meta.columns[0:meta_rows],
                             self.meta.dtypes[0:meta_rows])])
        # print `...` if more than `meta_rows` columns
        if len(self.meta.columns) > meta_rows:
            info += '\n   ...'

        # add info on size (optional)
        if memory_usage:
            size = self._data.memory_usage() + sum(self.meta.memory_usage())
            info += f'\nMemory usage: {size} bytes'

        return info

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

    @property
    def index(self):
        """Return all model-scenario combinations as :class:`pandas.MultiIndex`

        The index allows to loop over the available model-scenario combinations
        using:

        .. code-block:: python

            for model, scenario in df.index:
                ...
        """
        return self.meta.index

    @property
    def model(self):
        """Return the list of (unique) model names"""
        return get_index_levels(self.meta, 'model')

    @property
    def scenario(self):
        """Return the list of (unique) scenario names"""
        return get_index_levels(self.meta, 'scenario')

    @property
    def region(self):
        """Return the list of (unique) regions"""
        return get_index_levels(self._data, 'region')

    @property
    def variable(self):
        """Return the list of (unique) variables"""
        return get_index_levels(self._data, 'variable')

    @property
    def unit(self):
        """Return the list of (unique) units"""
        return get_index_levels(self._data, 'unit')

    @property
    def data(self):
        """Return the timeseries data as a long :class:`pandas.DataFrame`"""
        if self.empty:  # reset_index fails on empty with `datetime` column
            return pd.DataFrame([], columns=self._LONG_IDX + ['value'])
        return self._data.reset_index()

    @data.setter
    def data(self, df):
        """Set the timeseries data from a long :class:`pandas.DataFrame`"""
        self._data = format_time_col(df, self.time_col)\
            .set_index(self._LONG_IDX).value

    def copy(self):
        """Make a deepcopy of this object

        See :func:`copy.deepcopy` for details.
        """
        return copy.deepcopy(self)

    def head(self, *args, **kwargs):
        """Identical to :meth:`pandas.DataFrame.head()` operating on data"""
        return self.data.head(*args, **kwargs)

    def tail(self, *args, **kwargs):
        """Identical to :meth:`pandas.DataFrame.tail()` operating on data"""
        return self.data.tail(*args, **kwargs)

    @property
    def empty(self):
        """Indicator whether this object is empty"""
        return self._data.empty

    def equals(self, other):
        """Test if two objects contain the same data and meta indicators

        This function allows two IamDataFrame instances to be compared against
        each other to see if they have the same timeseries data and meta
        indicators. nan's in the same location of the meta table are considered
        equal.

        Parameters
        ----------
        other : IamDataFrame
            the other :class:`IamDataFrame` to be compared with `self`
        """
        if not isinstance(other, IamDataFrame):
            raise ValueError('`other` is not an `IamDataFrame` instance')

        if compare(self, other).empty and self.meta.equals(other.meta):
            return True
        else:
            return False

    def models(self):
        """Get a list of models"""
        # TODO: deprecate in release >=0.10
        deprecation_warning('Use the attribute `model` instead.')
        return pd.Series(self.meta.index.levels[0])

    def scenarios(self):
        """Get a list of scenarios"""
        # TODO: deprecate in release >=0.10
        deprecation_warning('Use the attribute `scenario` instead.')
        return pd.Series(self.meta.index.levels[1])

    def regions(self):
        """Get a list of regions"""
        # TODO: deprecate in release >=0.10
        deprecation_warning('Use the attribute `region` instead.')
        return pd.Series(get_index_levels(self._data, 'region'), name='region')

    def variables(self, include_units=False):
        """Get a list of variables

        Parameters
        ----------
        include_units : boolean, default False
            include the units
        """
        if not include_units:
            _var = 'variable'
            deprecation_warning('Use the attribute `variable` instead.')
            return pd.Series(get_index_levels(self._data, _var), name=_var)

        # else construct dataframe from variable and unit levels
        return (
            pd.DataFrame(zip(self._data.index.get_level_values('variable'),
                             self._data.index.get_level_values('unit')),
                         columns=['variable', 'unit'])
            .drop_duplicates().sort_values('variable').reset_index(drop=True)
        )

    def append(self, other, ignore_meta_conflict=False, inplace=False,
               **kwargs):
        """Append any IamDataFrame-like object to this object

        Indicators in `other.meta` that are not in `self.meta` are merged.
        Missing values are set to `NaN`.
        Conflicting `data` rows always raise a `ValueError`.

        Parameters
        ----------
        other : IamDataFrame, ixmp.Scenario, pandas.DataFrame or data file
            Any object castable as IamDataFrame to be appended
        ignore_meta_conflict : bool, default False
            If False and `other` is an IamDataFrame, raise an error if
            any meta columns present in `self` and `other` are not identical.
        inplace : bool, default False
            If True, do operation inplace and return None
        kwargs
            Passed to :class:`IamDataFrame(other, **kwargs) <IamDataFrame>`
            if `other` is not already an IamDataFrame

        Returns
        -------
        IamDataFrame
            If *inplace* is :obj:`False`.
        None
            If *inplace* is :obj:`True`.

        Raises
        ------
        ValueError
            If time domain or other timeseries data index dimension don't match
        """
        if not isinstance(other, IamDataFrame):
            other = IamDataFrame(other, **kwargs)
            ignore_meta_conflict = True

        if self.time_col != other.time_col:
            raise ValueError('Incompatible time format (`year` vs. `time`)')

        if self._data.index.names != other._data.index.names:
            raise ValueError('Incompatible timeseries data index dimensions')

        ret = self.copy() if not inplace else self

        # merge `meta` tables
        ret.meta = merge_meta(ret.meta, other.meta, ignore_meta_conflict)

        # append other.data (verify integrity for no duplicates)
        _data = ret._data.append(other._data, verify_integrity=True)

        # merge extra columns in `data` and set `self._LONG_IDX`
        ret.extra_cols += [i for i in other.extra_cols
                           if i not in ret.extra_cols]
        ret._LONG_IDX = IAMC_IDX + [ret.time_col] + ret.extra_cols
        ret._data = _data.sort_index()

        if not inplace:
            return ret

    def pivot_table(self, index, columns, values='value',
                    aggfunc='count', fill_value=None, style=None):
        """Returns a pivot table

        Parameters
        ----------
        index : str or list of str
            rows for Pivot table
        columns : str or list of str
            columns for Pivot table
        values : str, default 'value'
            dataframe column to aggregate or count
        aggfunc : str or function, default 'count'
            function used for aggregation,
            accepts 'count', 'mean', and 'sum'
        fill_value : scalar, default None
            value to replace missing values with
        style : str, default None
            output style for pivot table formatting
            accepts 'highlight_not_max', 'heatmap'
        """
        index = [index] if isstr(index) else index
        columns = [columns] if isstr(columns) else columns

        if values != 'value':
            raise ValueError("This method only supports `values='value'`!")

        df = self._data

        # allow 'aggfunc' to be passed as string for easier user interface
        if isstr(aggfunc):
            if aggfunc == 'count':
                df = self._data.groupby(index + columns).count()
                fill_value = 0
            elif aggfunc == 'mean':
                df = self._data.groupby(index + columns).mean()\
                    .round(2)
                fill_value = 0 if style == 'heatmap' else ""
            elif aggfunc == 'sum':
                df = self._data.groupby(index + columns).sum()
                fill_value = 0 if style == 'heatmap' else ""

        df = df.unstack(level=columns, fill_value=fill_value)
        return df

    def interpolate(self, time, inplace=None, **kwargs):
        """Interpolate missing values in the timeseries data

        This method uses :meth:`pandas.DataFrame.interpolate`,
        which applies linear interpolation by default

        Parameters
        ----------
        time : int or datetime, or list-like thereof
             Year or :class:`datetime.datetime` to be interpolated.
             This must match the datetime/year format of `self`.
        inplace : bool, optional
            if True, do operation inplace and return None
        kwargs
            passed to :meth:`pandas.DataFrame.interpolate`
        """
        # TODO deprecate and add kwarg inplace=False in release >= 0.10
        if inplace is None:
            deprecation_warning(
                'Behavior of `interpolate` will change to `inplace=False` '
                'as default in a future release. Set the kwarg explicitly '
                'to avoid this warning. Use `inplace=True` to keep current '
                'behavior.'
            )
            inplace = True
        ##

        # setup
        ret = self.copy() if not inplace else self
        interp_kwargs = dict(method='slinear', axis=1)
        interp_kwargs.update(kwargs)
        time = list(time) if islistable(time) else [time]
        # TODO - have to explicitly cast to numpy datetime to sort later,
        # could enforce as we do for year below
        if self.time_col == 'time':
            time = list(map(np.datetime64, time))
        elif not all(isinstance(x, int) for x in time):
            raise ValueError(
                'The `time` argument `{}` contains non-integers'.format(time)
            )
        old_cols = list(ret[ret.time_col].unique())
        columns = np.sort(np.unique(old_cols + time))

        # calculate a separate dataframe with full interpolation
        df = ret.timeseries()
        newdf = (
            df
            .reindex(columns=columns)
            .interpolate(**interp_kwargs)
        )

        # replace only columns asked for
        for col in time:
            df[col] = newdf[col]

        # replace underlying data object
        # TODO naming time_col could be done in timeseries()
        df.columns.name = ret.time_col
        df = df.stack()  # long-data to pd.Series
        df.name = 'value'
        ret._data = df.sort_index()

        if not inplace:
            return ret

    def swap_time_for_year(self, inplace=False):
        """Convert the `time` column to `year`.

        Parameters
        ----------
        inplace : bool, default False
            if True, do operation inplace and return None

        Raises
        ------
        ValueError
            "time" is not a column of `self.data`
        """
        if "time" not in self.data:
            raise ValueError("time column must be datetime to use this method")

        ret = self.copy() if not inplace else self

        _data = ret.data
        _data["year"] = _data["time"].apply(lambda x: x.year)
        _data = _data.drop("time", axis="columns")
        ret._LONG_IDX = [v if v != "time" else "year" for v in ret._LONG_IDX]

        if any(_data[ret._LONG_IDX].duplicated()):
            error_msg = ('swapping time for year will result in duplicate '
                         'rows in `data`!')
            raise ValueError(error_msg)
        ret._data = _data.set_index(IAMC_IDX)

        if not inplace:
            return ret

    def as_pandas(self, meta_cols=True):
        """Return object as a pandas.DataFrame

        Parameters
        ----------
        meta_cols : list, default None
            join `data` with all `meta` columns if True (default)
            or only with columns in list, or return copy of `data` if False
        """
        # merge data and (downselected) meta, or return copy of data
        if meta_cols:
            meta_cols = self.meta.columns if meta_cols is True else meta_cols
            return (
                self.data
                .set_index(META_IDX)
                .join(self.meta[meta_cols])
                .reset_index()
            )
        else:
            return self.data.copy()

    def timeseries(self, iamc_index=False):
        """Returns `data` as :class:`pandas.DataFrame` in wide format

        Parameters
        ----------
        iamc_index : bool, default False
            if True, use `['model', 'scenario', 'region', 'variable', 'unit']`;
            else, use all 'data' columns

        Raises
        ------
        ValueError
            `IamDataFrame` is empty
        ValueError
            reducing to IAMC-index yields an index with duplicates
        """
        if self.empty:
            raise ValueError('this `IamDataFrame` is empty')

        df = self._data.unstack(level=self.time_col).rename_axis(None, axis=1)

        if df.index.has_duplicates:
            raise ValueError('timeseries object has duplicates in index ',
                             'use `iamc_index=False`')
        return df

    def reset_exclude(self):
        """Reset exclusion assignment for all scenarios to `exclude: False`"""
        self.meta['exclude'] = False

    def set_meta(self, meta, name=None, index=None):
        """Add meta indicators as pandas.Series, list or value (int/float/str)

        Parameters
        ----------
        meta : pandas.Series, list, int, float or str
            column to be added to 'meta'
            (by `['model', 'scenario']` index if possible)
        name : str, optional
            meta column name (defaults to meta `pandas.Series.name`);
            either `meta.name` or the name kwarg must be defined
        index : IamDataFrame, pandas.DataFrame or pandas.MultiIndex, optional
            index to be used for setting meta column (`['model', 'scenario']`)
        """
        # check that name is valid and doesn't conflict with data columns
        if (name or (hasattr(meta, 'name') and meta.name)) in [None, False]:
            raise ValueError('Must pass a name or use a named pd.Series')
        name = name or meta.name
        if name in self._data.index.names:
            raise ValueError(f'Column {name} already exists in `data`!')
        if name in ILLEGAL_COLS:
            raise ValueError(f'Name {name} is illegal for meta indicators!')

        # check if meta has a valid index and use it for further workflow
        if hasattr(meta, 'index') and hasattr(meta.index, 'names') \
                and set(META_IDX).issubset(meta.index.names):
            index = meta.index

        # if no valid index is provided, add meta as new column `name` and exit
        if index is None:
            self.meta[name] = list(meta) if islistable(meta) else meta
            return  # EXIT FUNCTION

        # use meta.index if index arg is an IamDataFrame
        if isinstance(index, IamDataFrame):
            index = index.meta.index
        # turn dataframe to index if index arg is a DataFrame
        if isinstance(index, pd.DataFrame):
            index = index.set_index(META_IDX).index
        if not isinstance(index, pd.MultiIndex):
            raise ValueError('index cannot be coerced to pd.MultiIndex')

        # raise error if index is not unique
        if index.duplicated().any():
            raise ValueError("non-unique ['model', 'scenario'] index!")

        # create pd.Series from meta, index and name if provided
        meta = pd.Series(data=meta, index=index, name=name)

        # reduce index dimensions to model-scenario only
        meta = (
            meta
            .reset_index()
            .reindex(columns=META_IDX + [name])
            .set_index(META_IDX)
        )

        # check if trying to add model-scenario index not existing in self
        diff = meta.index.difference(self.meta.index)
        if not diff.empty:
            msg = 'Adding meta for non-existing scenarios:\n{}'
            raise ValueError(msg.format(diff))

        self._new_meta_column(name)
        self.meta[name] = meta[name].combine_first(self.meta[name])

    def set_meta_from_data(self, name, method=None, column='value', **kwargs):
        """Add meta indicators from downselected timeseries data of self

        Parameters
        ----------
        name : str
            column name of the 'meta' table
        method : function, optional
            method for aggregation
            (e.g., :func:`numpy.max <numpy.ndarray.max>`);
            required if downselected data do not yield unique values
        column : str, optional
            the column from `data` to be used to derive the indicator
        kwargs
            passed to :meth:`filter` for downselected data
        """
        _data = self.filter(**kwargs).data
        if method is None:
            meta = _data.set_index(META_IDX)[column]
        else:
            meta = _data.groupby(META_IDX)[column].apply(method)
        self.set_meta(meta, name)

    def categorize(self, name, value, criteria,
                   color=None, marker=None, linestyle=None):
        """Assign scenarios to a category according to specific criteria

        Parameters
        ----------
        name : str
            column name of the 'meta' table
        value : str
            category identifier
        criteria : dict
            dictionary with variables mapped to applicable checks
            ('up' and 'lo' for respective bounds, 'year' for years - optional)
        color : str
            assign a color to this category for plotting
        marker : str
            assign a marker to this category for plotting
        linestyle : str
            assign a linestyle to this category for plotting
        """
        # add plotting run control
        for kind, arg in [('color', color), ('marker', marker),
                          ('linestyle', linestyle)]:
            if arg:
                run_control().update({kind: {name: {value: arg}}})
        # find all data that matches categorization
        rows = _apply_criteria(self._data, criteria,
                               in_range=True, return_test='all')
        idx = _make_index(rows)

        if len(idx) == 0:
            logger.info("No scenarios satisfy the criteria")
            return  # EXIT FUNCTION

        # update meta dataframe
        self._new_meta_column(name)
        self.meta.loc[idx, name] = value
        msg = '{} scenario{} categorized as `{}: {}`'
        logger.info(msg.format(len(idx), '' if len(idx) == 1 else 's',
                                 name, value))

    def _new_meta_column(self, name):
        """Add a column to meta if it doesn't exist, set value to nan"""
        if name is None:
            raise ValueError('cannot add a meta column `{}`'.format(name))
        if name not in self.meta:
            self.meta[name] = np.nan

    def require_variable(self, variable, unit=None, year=None,
                         exclude_on_fail=False):
        """Check whether all scenarios have a required variable

        Parameters
        ----------
        variable : str
            required variable
        unit : str, default None
            name of unit (optional)
        year : int or list, default None
            check whether the variable exists for ANY of the years (if a list)
        exclude_on_fail : bool, default False
            flag scenarios missing the required variables as `exclude: True`
        """
        criteria = {'variable': variable}
        if unit:
            criteria.update({'unit': unit})
        if year:
            criteria.update({'year': year})

        keep = self._apply_filters(**criteria)
        idx = self.meta.index.difference(_meta_idx(self.data[keep]))

        n = len(idx)
        if n == 0:
            logger.info('All scenarios have the required variable `{}`'
                          .format(variable))
            return

        msg = '{} scenario does not include required variable `{}`' if n == 1 \
            else '{} scenarios do not include required variable `{}`'

        if exclude_on_fail:
            self.meta.loc[idx, 'exclude'] = True
            msg += ', marked as `exclude: True` in `meta`'

        logger.info(msg.format(n, variable))
        return pd.DataFrame(index=idx).reset_index()

    def validate(self, criteria={}, exclude_on_fail=False):
        """Validate scenarios using criteria on timeseries values

        Returns all scenarios which do not match the criteria and prints a log
        message, or returns None if all scenarios match the criteria.

        When called with `exclude_on_fail=True`, scenarios not
        satisfying the criteria will be marked as `exclude=True`.

        Parameters
        ----------
        criteria : dict
           dictionary with variable keys and validation mappings
            ('up' and 'lo' for respective bounds, 'year' for years)
        exclude_on_fail : bool, optional
            flag scenarios failing validation as `exclude: True`

        Returns
        -------
        :class:`pandas.DataFrame`
            All data points that do not satisfy the criteria.
        None
            If all scenarios satisfy the criteria.
        """
        df = _apply_criteria(self._data, criteria, in_range=False)

        if not df.empty:
            msg = '{} of {} data points do not satisfy the criteria'
            logger.info(msg.format(len(df), len(self.data)))

            if exclude_on_fail and len(df) > 0:
                self._exclude_on_fail(df)
            return df.reset_index()

    def rename(self, mapping=None, inplace=False, append=False,
               check_duplicates=True, **kwargs):
        """Rename and aggregate columns using `groupby().sum()` on values

        When renaming models or scenarios, the uniqueness of the index must be
        maintained, and the function will raise an error otherwise.

        Renaming is only applied to any data row that matches for all
        columns given in `mapping`. Renaming can only be applied to the `model`
        and `scenario` columns, or to other data columns simultaneously.

        Parameters
        ----------
        mapping : dict or kwargs
            mapping of column name to rename-dictionary of that column

            .. code-block:: python

               dict(<column_name>: {<current_name_1>: <target_name_1>,
                                    <current_name_2>: <target_name_2>})

            or kwargs as `column_name={<current_name_1>: <target_name_1>, ...}`
        inplace : bool, default False
            if True, do operation inplace and return None
        append : bool, default False
            append renamed timeseries to `self` and return None;
            else return new `IamDataFrame`
        check_duplicates: bool, default True
            check whether conflict between existing and renamed data exists.
            If True, raise ValueError; if False, rename and merge
            with :meth:`groupby().sum() <pandas.core.groupby.GroupBy.sum>`.
        """
        # combine `mapping` arg and mapping kwargs, ensure no rename conflicts
        mapping = mapping or {}
        duplicate = set(mapping).intersection(kwargs)
        if duplicate:
            msg = 'conflicting rename args for columns `{}`'.format(duplicate)
            raise ValueError(msg)
        mapping.update(kwargs)

        # determine columns that are not `model` or `scenario`
        data_cols = set(self._LONG_IDX) - set(META_IDX)

        # changing index and data columns can cause model-scenario mismatch
        if any(i in mapping for i in META_IDX)\
                and any(i in mapping for i in data_cols):
            msg = 'Renaming index and data cols simultaneously not supported!'
            raise ValueError(msg)

        # translate rename mapping to `filter()` arguments
        filters = {col: _from.keys() for col, _from in mapping.items()}

        # if append is True, downselect and append renamed data
        if append:
            df = self.filter(**filters)
            # note that `append(other, inplace=True)` returns None
            return self.append(df.rename(mapping), inplace=inplace)

        # if append is False, iterate over rename mapping and do groupby
        ret = self.copy() if not inplace else self

        # renaming is only applied where a filter matches for all given columns
        rows = ret._apply_filters(**filters)
        idx = ret.meta.index.isin(_make_index(ret.data[rows]))

        # if `check_duplicates`, do the rename on a copy until after the check
        # _data = ret.data.copy() if check_duplicates else ret.data
        # TODO reactivate this avoidance of creating a copy
        _data = ret.data

        # apply renaming changes
        for col, _mapping in mapping.items():
            if col in META_IDX:
                _index = pd.DataFrame(index=ret.meta.index).reset_index()
                _index.loc[idx, col] = _index.loc[idx, col].replace(_mapping)
                if _index.duplicated().any():
                    raise ValueError('Renaming to non-unique `{}` index!'
                                     .format(col))
                ret.meta.index = _index.set_index(META_IDX).index
            elif col not in data_cols:
                raise ValueError('Renaming by `{}` not supported!'.format(col))
            _data.loc[rows, col] = _data.loc[rows, col].replace(_mapping)

        # check if duplicates exist between the renamed and not-renamed data
        if check_duplicates:
            merged = (
                _data.loc[rows, self._LONG_IDX].drop_duplicates().append(
                    _data.loc[~rows, self._LONG_IDX].drop_duplicates())
            )
            if any(merged.duplicated()):
                msg = 'Duplicated rows between original and renamed data!\n{}'
                conflict_rows = merged.loc[merged.duplicated(), self._LONG_IDX]
                raise ValueError(msg.format(conflict_rows.drop_duplicates()))

        # merge using `groupby().sum()`
        ret.data = _data.groupby(ret._LONG_IDX).sum().reset_index()

        if not inplace:
            return ret

    def convert_unit(self, current, to, factor=None, registry=None,
                     context=None, inplace=False):
        r"""Convert all data having *current* units to new units.

        If *factor* is given, existing values are multiplied by it, and the
        *to* units are assigned to the 'unit' column.

        Otherwise, the :mod:`pint` package is used to convert from *current* ->
        *to* units without an explicit conversion factor. Pint natively handles
        conversion between any standard (SI) units that have compatible
        dimensionality, such as exajoule to terawatt-hours, :code:`EJ -> TWh`,
        or tonne per year to gram per second, :code:`t / yr -> g / sec`.

        The default *registry* includes additional unit definitions relevant
        for integrated assessment models and energy systems analysis, via the
        `iam-units <https://github.com/IAMconsortium/units>`_ package.
        This registry can also be accessed directly, using:

        .. code-block:: python

            from iam_units import registry

        When using this registry, *current* and *to* may contain the symbols of
        greenhouse gas (GHG) species, such as 'CO2e', 'C', 'CH4', 'N2O',
        'HFC236fa', etc., as well as lower-case aliases like 'co2' supported by
        :mod:`pyam`. In this case, *context* must be the name of a specific
        global warming potential (GWP) metric supported by :mod:`iam_units`,
        e.g. 'AR5GWP100' (optionally prefixed by 'gwp_', e.g. 'gwp_AR5GWP100').

        Rows with units other than *current* are not altered.

        Parameters
        ----------
        current : str
            Current units to be converted.
        to : str
            New unit (to be converted to) or symbol for target GHG species. If
            only the GHG species is provided, the units (e.g. :code:`Mt /
            year`) will be the same as `current`, and an expression combining
            units and species (e.g. 'Mt CO2e / yr') will be placed in the
            'unit' column.
        factor : value, optional
            Explicit factor for conversion without `pint`.
        registry : :class:`pint.UnitRegistry`, optional
            Specific unit registry to use for conversion. Default: the
            `iam-units <https://github.com/IAMconsortium/units>`_ registry.
        context : str or :class:`pint.Context`, optional
            (Name of) the context to use in conversion.
            Required when converting between GHG species using GWP metrics,
            unless the species indicated by *current* and *to* are the same.
        inplace : bool, optional
            Whether to return a new IamDataFrame.

        Returns
        -------
        IamDataFrame
            If *inplace* is :obj:`False`.
        None
            If *inplace* is :obj:`True`.

        Raises
        ------
        pint.UndefinedUnitError
            if attempting a GWP conversion but *context* is not given.
        pint.DimensionalityError
            without *factor*, when *current* and *to* are not compatible units.
        """
        # Handle user input
        # Check that (only) either factor or registry/context is provided
        if factor and any([registry, context]):
            raise ValueError('use either `factor` or `pint.UnitRegistry`')

        # new standard method, remove this comment when deprecating above
        return convert_unit(self, current, to, factor, registry, context,
                            inplace)

    def normalize(self, inplace=False, **kwargs):
        """Normalize data to a specific data point

        Note: Currently only supports normalizing to a specific time.

        Parameters
        ----------
        inplace : bool, optional
            if :obj:`True`, do operation inplace and return None
        kwargs
            the column and value on which to normalize (e.g., `year=2005`)
        """
        if len(kwargs) > 1 or self.time_col not in kwargs:
            raise ValueError('Only time(year)-based normalization supported')
        ret = self.copy() if not inplace else self
        df = ret.data
        # change all below if supporting more in the future
        cols = self.time_col
        value = kwargs[self.time_col]
        x = df.set_index(IAMC_IDX)
        x['value'] /= x[x[cols] == value]['value']
        ret.data = x.reset_index()
        if not inplace:
            return ret

    def aggregate(self, variable, components=None, method='sum',
                  recursive=False, append=False):
        """Aggregate timeseries components or sub-categories within each region

        Parameters
        ----------
        variable : str or list of str
            variable(s) for which the aggregate will be computed
        components : list of str, optional
            list of variables to aggregate, defaults to all sub-categories
            of `variable`
        method : func or str, optional
            method to use for aggregation,
            e.g. :func:`numpy.mean`, :func:`numpy.sum`, 'min', 'max'
        recursive : bool, optional
            iterate recursively over all subcategories of `variable`
        append : bool, optional
            append the aggregate timeseries to `self` and return None,
            else return aggregate timeseries as new :class:`IamDataFrame`

        Notes
        -----
        The aggregation function interprets any missing values
        (:any:`numpy.nan`) for individual components as 0.
        """

        if recursive is True:
            if components is not None:
                msg = 'Recursive aggregation cannot take explicit components'
                raise ValueError(msg)
            _df = _aggregate_recursive(self, variable, method=method)
        else:
            _df = _aggregate(self, variable, components=components,
                             method=method)

        # return None if there is nothing to aggregate
        if _df is None:
            return None

        # else, append to `self` or return as `IamDataFrame`
        if append is True:
            self.append(_df, inplace=True)
        else:
            return IamDataFrame(_df, meta=self.meta)

    def check_aggregate(self, variable, components=None, method='sum',
                        exclude_on_fail=False, multiplier=1, **kwargs):
        """Check whether a timeseries matches the aggregation of its components

        Parameters
        ----------
        variable : str or list of str
            variable(s) checked for matching aggregation of sub-categories
        components : list of str, default None
            list of variables, defaults to all sub-categories of `variable`
        method : func or str, optional
            method to use for aggregation,
            e.g. :func:`numpy.mean`, :func:`numpy.sum`, 'min', 'max'
        exclude_on_fail : bool, optional
            flag scenarios failing validation as `exclude: True`
        multiplier : number, optional
            factor when comparing variable and sum of components
        kwargs : arguments for comparison of values
            passed to :func:`numpy.isclose`
        """
        # compute aggregate from components, return None if no components
        df_components = _aggregate(self, variable, components, method)
        if df_components is None:
            return

        # filter and groupby data, use `pd.Series.align` for matching index
        rows = self._apply_filters(variable=variable)
        df_var, df_components = (
            _group_and_agg(self.data[rows], [], method)
            .align(df_components)
        )

        # use `np.isclose` for checking match
        rows = ~np.isclose(df_var, multiplier * df_components, **kwargs)

        # if aggregate and components don't match, return inconsistent data
        if sum(rows):
            msg = '`{}` - {} of {} rows are not aggregates of components'
            logger.info(msg.format(variable, sum(rows), len(df_var)))

            if exclude_on_fail:
                self._exclude_on_fail(_meta_idx(df_var[rows].reset_index()))

            return pd.concat([df_var[rows], df_components[rows]], axis=1,
                             keys=(['variable', 'components']))

    def aggregate_region(self, variable, region='World', subregions=None,
                         components=False, method='sum', weight=None,
                         append=False):
        """Aggregate a timeseries over a number of subregions

        This function allows to add variable sub-categories that are only
        defined at the `region` level by setting `components=True`

        Parameters
        ----------
        variable : str or list of str
            variable(s) to be aggregated
        region : str, default 'World'
            region to which data will be aggregated
        subregions : list of str
            list of subregions, defaults to all regions other than `region`
        components : bool or list of str, optional
            variables at the `region` level to be included in the aggregation
            (ignored if False); if `True`, use all sub-categories of `variable`
            included in `region` but not in any of the `subregions`;
            or explicit list of variables
        method : func or str, optional
            method to use for aggregation,
            e.g. :func:`numpy.mean`, :func:`numpy.sum`, 'min', 'max'
        weight : str, default None
            variable to use as weight for the aggregation
            (currently only supported with `method='sum'`)
        append : bool, default False
            append the aggregate timeseries to `self` and return None,
            else return aggregate timeseries as new :class:`IamDataFrame`
        """
        _df = _aggregate_region(self, variable, region=region,
                                subregions=subregions, components=components,
                                method=method, weight=weight)

        # return None if there is nothing to aggregate
        if _df is None:
            return None

        # else, append to `self` or return as `IamDataFrame`
        if append is True:
            if not _df.empty:
                self.append(_df, region=region, inplace=True)
        else:
            if _df.empty:
                return _empty_iamframe(self._LONG_IDX + ['value'])
            return IamDataFrame(_df, region=region, meta=self.meta)

    def check_aggregate_region(self, variable, region='World', subregions=None,
                               components=False, method='sum', weight=None,
                               exclude_on_fail=False, **kwargs):
        """Check whether a timeseries matches the aggregation across subregions

        Parameters
        ----------
        variable : str or list of str
            variable(s) to be checked for matching aggregation of subregions
        region : str, default 'World'
            region to be checked for matching aggregation of subregions
        subregions : list of str
            list of subregions, defaults to all regions other than `region`
        components : bool or list of str, default False
            variables at the `region` level to be included in the aggregation
            (ignored if False); if `True`, use all sub-categories of `variable`
            included in `region` but not in any of the `subregions`;
            or explicit list of variables
        method : func or str, optional
            method to use for aggregation,
            e.g. :func:`numpy.mean`, :func:`numpy.sum`, 'min', 'max'
        weight : str, optional
            variable to use as weight for the aggregation
            (currently only supported with `method='sum'`)
        exclude_on_fail : boolean, optional
            flag scenarios failing validation as `exclude: True`
        kwargs : arguments for comparison of values
            passed to :func:`numpy.isclose`
        """
        # compute aggregate from subregions, return None if no subregions
        df_subregions = _aggregate_region(self, variable, region, subregions,
                                          components, method, weight)

        if df_subregions is None:
            return

        # filter and groupby data, use `pd.Series.align` for matching index
        rows = self._apply_filters(region=region, variable=variable)
        if not rows.any():
            msg = 'variable `{}` does not exist in region `{}`'
            logger.info(msg.format(variable, region))
            return

        df_region, df_subregions = (
            _group_and_agg(self.data[rows], 'region')
            .align(df_subregions)
        )

        # use `np.isclose` for checking match
        rows = ~np.isclose(df_region, df_subregions, **kwargs)

        # if region and subregions don't match, return inconsistent data
        if sum(rows):
            msg = '`{}` - {} of {} rows are not aggregates of subregions'
            logger.info(msg.format(variable, sum(rows), len(df_region)))

            if exclude_on_fail:
                self._exclude_on_fail(_meta_idx(df_region[rows].reset_index()))

            _df = pd.concat(
                [pd.concat([df_region[rows], df_subregions[rows]], axis=1,
                           keys=(['region', 'subregions']))],
                keys=[region], names=['region'])
            _df.index = _df.index.reorder_levels(self._LONG_IDX)
            return _df

    def aggregate_time(self, variable, column='subannual', value='year',
                       components=None, method='sum', append=False):
        """Aggregate a timeseries over a subannual time resolution

         Parameters
         ----------
         variable : str or list of str
             variable(s) to be aggregated
         column : str, optional
             the data column to be used as subannual time representation
         value : str, optional
             the name of the aggregated (subannual) time
         components : list of str
             subannual timeslices to be aggregated; defaults to all subannual
             timeslices other than `value`
         method : func or str, optional
             method to use for aggregation,
             e.g. :func:`numpy.mean`, :func:`numpy.sum`, 'min', 'max'
         append : bool, optional
             append the aggregate timeseries to `self` and return None,
             else return aggregate timeseries as new :class:`IamDataFrame`
         """
        _df = _aggregate_time(self, variable, column=column, value=value,
                              components=components, method=method)

        # return None if there is nothing to aggregate
        if _df is None:
            return None

        # else, append to `self` or return as `IamDataFrame`
        if append is True:
            self.append(_df, inplace=True)
        else:
            return IamDataFrame(_df, meta=self.meta)

    def downscale_region(self, variable, region='World', subregions=None,
                         proxy=None, weight=None, append=False):
        """Downscale a timeseries to a number of subregions

        Parameters
        ----------
        variable : str or list of str
            variable(s) to be downscaled
        region : str, optional
            region from which data will be downscaled
        subregions : list of str, optional
            list of subregions, defaults to all regions other than `region`
            (if using `proxy`) or `region` index (if using `weight`)
        proxy : str, optional
            variable (within the :class:`IamDataFrame`) to be used as proxy
            for regional downscaling
        weight : class:`pandas.DataFrame`, optional
            dataframe with time dimension as columns (year or
            :class:`datetime.datetime`) and regions[, model, scenario] as index
        append : bool, optional
            append the downscaled timeseries to `self` and return None,
            else return downscaled data as new IamDataFrame
        """
        if proxy is not None and weight is not None:
            raise ValueError(
                'Using both `proxy` and `weight` arguments is not valid')
        elif proxy is not None:
            # get default subregions if not specified and select data from self
            subregions = subregions or self._all_other_regions(region)
            rows = self._apply_filters(variable=proxy, region=subregions)
            cols = self._get_cols(['region', self.time_col])
            _proxy = self.data[rows].set_index(cols).value
        elif weight is not None:
            # downselect weight to subregions or remove `region` from index
            if subregions is not None:
                rows = weight.index.isin(subregions, level='region')
            else:
                rows = ~weight.index.isin([region], level='region')
            _proxy = weight[rows].stack()
        else:
            raise ValueError(
                'Either a `proxy` or `weight` argument is required')

        _value = (
            self.data[self._apply_filters(variable=variable, region=region)]
            .set_index(self._get_cols(['variable', 'unit', self.time_col]))
            .value
        )

        # compute downscaled data
        _total = _proxy.groupby(self.time_col).sum()
        _data = _value * _proxy / _total

        if append is True:
            self.append(_data, inplace=True)
        else:
            return IamDataFrame(_data, meta=self.meta)

    def _all_other_regions(self, region, variable=None):
        """Return list of regions other than `region` containing `variable`"""
        rows = self._apply_filters(variable=variable)
        return set(self.data[rows].region) - set([region])

    def _variable_components(self, variable, level=0):
        """Get all components (sub-categories) of a variable for a given level

        If `level=0`, for `variable='foo'`, return `['foo|bar']`, but don't
        include `'foo|bar|baz'`, which is a sub-sub-category. If `level=None`,
        all variables below `variable` in the hierarchy are returned."""
        var_list = pd.Series(self.data.variable.unique())
        return var_list[pattern_match(var_list, '{}|*'.format(variable),
                                      level=level)]

    def _get_cols(self, cols):
        """Return a list of columns of `self.data`"""
        return META_IDX + cols + self.extra_cols

    def check_internal_consistency(self, components=False, **kwargs):
        """Check whether a scenario ensemble is internally consistent

        We check that all variables are equal to the sum of their sectoral
        components and that all the regions add up to the World total. If
        the check is passed, None is returned, otherwise a DataFrame of
        inconsistent variables is returned.

        Note: at the moment, this method's regional checking is limited to
        checking that all the regions sum to the World region. We cannot
        make this more automatic unless we store how the regions relate,
        see `this issue <https://github.com/IAMconsortium/pyam/issues/106>`_.

        Parameters
        ----------
        kwargs : arguments for comparison of values
            passed to :func:`numpy.isclose`
        components : bool, optional
            passed to :meth:`check_aggregate_region` if `True`, use all
            sub-categories of each `variable` included in `World` but not in
            any of the subregions; if `False`, only aggregate variables over
            subregions
        """
        lst = []
        for variable in self.variables():
            diff_agg = self.check_aggregate(variable, **kwargs)
            if diff_agg is not None:
                lst.append(diff_agg)

            diff_regional = (
                self.check_aggregate_region(variable, components=components,
                                            **kwargs)
            )
            if diff_regional is not None:
                lst.append(diff_regional)

        if len(lst):
            _df = pd.concat(lst, sort=True).sort_index()
            return _df[[c for c in ['variable', 'components', 'region',
                                    'subregions'] if c in _df.columns]]

    def _exclude_on_fail(self, df):
        """Assign a selection of scenarios as `exclude: True` in meta"""
        idx = df if isinstance(df, pd.MultiIndex) else _make_index(df)
        self.meta.loc[idx, 'exclude'] = True
        logger.info('{} non-valid scenario{} will be excluded'
                      .format(len(idx), '' if len(idx) == 1 else 's'))

    def filter(self, keep=True, inplace=False, **kwargs):
        """Return a (copy of a) filtered (downselected) IamDataFrame

        Parameters
        ----------
        keep : bool, optional
            keep all scenarios satisfying the filters (if True) or the inverse
        inplace : bool, optional
            if True, do operation inplace and return None
        filters by kwargs:
            The following columns are available for filtering:
             - 'meta' columns: filter by string value of that column
             - 'model', 'scenario', 'region', 'variable', 'unit':
               string or list of strings, where `*` can be used as a wildcard
             - 'level': the maximum "depth" of IAM variables (number of '|')
               (excluding the strings given in the 'variable' argument)
             - 'year': takes an integer (int/np.int64), a list of integers or
                a range. Note that the last year of a range is not included,
               so `range(2010, 2015)` is interpreted as `[2010, ..., 2014]`
             - arguments for filtering by `datetime.datetime` or np.datetime64
               ('month', 'hour', 'time')
             - 'regexp=True' disables pseudo-regexp syntax in `pattern_match()`
        """
        if not isinstance(keep, bool):
            msg = '`filter(keep={}, ...)` is not valid, must be boolean'
            raise ValueError(msg.format(keep))

        _keep = self._apply_filters(**kwargs)
        _keep = _keep if keep else ~_keep
        ret = self.copy() if not inplace else self
        # TODO remove cast to list after refactoring `_apply_filters()`
        ret._data = ret._data[list(_keep)]
        ret._data.index = ret._data.index.remove_unused_levels()

        idx = _make_index(ret._data)
        if len(idx) == 0:
            logger.warning('Filtered IamDataFrame is empty!')
        ret.meta = ret.meta.loc[idx]
        if not inplace:
            return ret

    def _apply_filters(self, **filters):
        """Determine rows to keep in data for given set of filters

        Parameters
        ----------
        filters : dict
            dictionary of filters of the format (`{col: values}`);
            uses a pseudo-regexp syntax by default,
            but accepts `regexp: True` in the dictionary to use regexp directly
        """
        regexp = filters.pop('regexp', False)
        keep = np.array([True] * len(self))

        # filter by columns and list of values
        for col, values in filters.items():
            # treat `_apply_filters(col=None)` as no filter applied
            if values is None:
                continue

            if col in self.meta.columns:
                matches = pattern_match(self.meta[col], values, regexp=regexp,
                                        has_nan=True)
                cat_idx = self.meta[matches].index
                keep_col = _make_index(self._data, unique=False).isin(cat_idx)

            elif col == 'variable':
                level = filters['level'] if 'level' in filters else None
                col_values = pd.Series(get_index_levels(self._data, col))
                where = pattern_match(col_values, values, level, regexp)

                keep_col = get_keep_col(self._data, col_values[where], col)

            elif col == 'year':
                _data = self.data[col] if self.time_col != 'time' \
                    else self.data['time'].apply(lambda x: x.year)
                keep_col = years_match(_data, values)

            elif col == 'month' and self.time_col == 'time':
                keep_col = month_match(self.data['time']
                                           .apply(lambda x: x.month),
                                       values)

            elif col == 'day' and self.time_col == 'time':
                if isinstance(values, str):
                    wday = True
                elif isinstance(values, list) and isinstance(values[0], str):
                    wday = True
                else:
                    wday = False

                if wday:
                    days = self.data['time'].apply(lambda x: x.weekday())
                else:  # ints or list of ints
                    days = self.data['time'].apply(lambda x: x.day)

                keep_col = day_match(days, values)

            elif col == 'hour' and self.time_col == 'time':
                keep_col = hour_match(self.data['time']
                                          .apply(lambda x: x.hour),
                                      values)

            elif col == 'time' and self.time_col == 'time':
                keep_col = datetime_match(self.data[col], values)

            elif col == 'level':
                if 'variable' not in filters.keys():
                    v = 'variable'
                    col_values = pd.Series(get_index_levels(self._data, v))
                    where = find_depth(col_values, level=values)
                    keep_col = get_keep_col(self._data, col_values[where], v)
                else:
                    continue

            elif col in self._data.index.names:
                col_values = pd.Series(get_index_levels(self._data, col))
                where = pattern_match(col_values, values, regexp=regexp)
                keep_col = get_keep_col(self._data, col_values[where], col)

            else:
                _raise_filter_error(col)

            keep = np.logical_and(keep, keep_col)

        return keep

    def col_apply(self, col, func, *args, **kwargs):
        """Apply a function to a column of data or meta

        Parameters
        ----------
        col: str
            column in either data or meta dataframes
        func: function
            function to apply
        """
        if col in self.data:
            self.data[col] = self.data[col].apply(func, *args, **kwargs)
        else:
            self.meta[col] = self.meta[col].apply(func, *args, **kwargs)

    def _to_file_format(self, iamc_index):
        """Return a dataframe suitable for writing to a file"""
        df = self.timeseries(iamc_index=iamc_index).reset_index()
        df = df.rename(columns={c: str(c).title() for c in df.columns})
        return df

    def to_csv(self, path, iamc_index=False, **kwargs):
        """Write timeseries data of this object to a csv file

        Parameters
        ----------
        path : str or path object
            file path or :class:`pathlib.Path`
        iamc_index : bool, default False
            if True, use `['model', 'scenario', 'region', 'variable', 'unit']`;
            else, use all 'data' columns
        """
        self._to_file_format(iamc_index).to_csv(path, index=False, **kwargs)

    def to_excel(self, excel_writer, sheet_name='data', iamc_index=False,
                 include_meta=True, **kwargs):
        """Write object to an Excel spreadsheet

        Parameters
        ----------
        excel_writer : str, path object or ExcelWriter object
            any valid string path, :class:`pathlib.Path`
            or :class:`pandas.ExcelWriter`
        sheet_name : string
            name of sheet which will contain :meth:`timeseries()` data
        iamc_index : bool, default False
            if True, use `['model', 'scenario', 'region', 'variable', 'unit']`;
            else, use all 'data' columns
        include_meta : boolean or string
            if True, write 'meta' to an Excel sheet name 'meta' (default);
            if this is a string, use it as sheet name
        """
        # open a new ExcelWriter instance (if necessary)
        close = False
        if not isinstance(excel_writer, pd.ExcelWriter):
            close = True
            excel_writer = pd.ExcelWriter(excel_writer, engine='openpyxl')

        # write data table
        write_sheet(excel_writer, sheet_name, self._to_file_format(iamc_index))

        # write meta table unless `include_meta=False`
        if include_meta:
            meta_rename = dict([(i, i.capitalize()) for i in META_IDX])
            write_sheet(excel_writer,
                        'meta' if include_meta is True else include_meta,
                        self.meta.reset_index().rename(columns=meta_rename))

        # close the file if `excel_writer` arg was a file name
        if close:
            excel_writer.close()

    def export_meta(self, excel_writer, sheet_name='meta'):
        """Write the 'meta' indicators of this object to an Excel sheet

        Parameters
        ----------
        excel_writer : str, path object or ExcelWriter object
            any valid string path, :class:`pathlib.Path`
            or :class:`pandas.ExcelWriter`
        sheet_name : str
            name of sheet which will contain dataframe of 'meta' indicators
        """
        if not isinstance(excel_writer, pd.ExcelWriter):
            close = True
            excel_writer = pd.ExcelWriter(excel_writer)
        write_sheet(excel_writer, sheet_name, self.meta, index=True)
        if close:
            excel_writer.close()

    def to_datapackage(self, path):
        """Write object to a frictionless Data Package

        More information: https://frictionlessdata.io

        Returns the saved :class:`datapackage.Package`
        (|datapackage.Package.docs|).
        When adding metadata (descriptors), please follow the `template`
        defined by https://github.com/OpenEnergyPlatform/metadata

        Parameters
        ----------
        path : string or path object
            any valid string path or :class:`pathlib.Path`
        """
        if not HAS_DATAPACKAGE:
            raise ImportError('Required package `datapackage` not found!')

        with TemporaryDirectory(dir='.') as tmp:
            # save data and meta tables to a temporary folder
            self.data.to_csv(Path(tmp) / 'data.csv', index=False)
            self.meta.to_csv(Path(tmp) / 'meta.csv')

            # cast tables to datapackage
            package = Package()
            package.infer('{}/*.csv'.format(tmp))
            if not package.valid:
                logger.warning('The exported datapackage is not valid')
            package.save(path)

        # return the package (needs to reloaded because `tmp` was deleted)
        return Package(path)

    def load_meta(self, path, *args, **kwargs):
        """Load 'meta' indicators from file

        Parameters
        ----------
        path : str or path object
            any valid string path or :class:`pathlib.Path`
        """
        # load from file
        df = read_pandas(Path(path), default_sheet='meta', *args, **kwargs)

        # cast model-scenario column headers to lower-case (if necessary)
        df = df.rename(columns=dict([(i.capitalize(), i) for i in META_IDX]))

        # check that required columns exist
        req_cols = ['model', 'scenario', 'exclude']
        if not set(req_cols).issubset(set(df.columns)):
            e = 'File `{}` does not have required columns {}!'
            raise ValueError(e.format(path, req_cols))

        # set index, filter to relevant scenarios from imported file
        df.set_index(META_IDX, inplace=True)
        idx = self.meta.index.intersection(df.index)

        n_invalid = len(df) - len(idx)
        if n_invalid > 0:
            msg = 'Ignoring {} scenario{} from imported meta file'
            logger.warning(msg.format(n_invalid, 's' if n_invalid > 1 else ''))

        if idx.empty:
            raise ValueError('No valid scenarios in imported meta file!')

        df = df.loc[idx]

        # merge in imported meta indicators
        msg = 'Importing meta indicators for {} scenario{} (for total of {})'
        logger.info(msg.format(len(df), 's' if len(df) > 1 else '',
                                 len(self.meta)))

        for col in df.columns:
            self._new_meta_column(col)
            self.meta[col] = df[col].combine_first(self.meta[col])
        # set column `exclude` to bool
        self.meta.exclude = self.meta.exclude.astype('bool')

    def line_plot(self, x='year', y='value', **kwargs):
        """Plot timeseries lines of existing data

        see pyam.plotting.line_plot() for all available options
        """
        df = self.as_pandas(meta_cols=mpl_args_to_meta_cols(self, **kwargs))

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
                df = df.drop('year', axis=1)  # years causes nan's

        ax, handles, labels = plotting.line_plot(
            df.dropna(), x=x, y=y, **kwargs)
        return ax

    def stack_plot(self, *args, **kwargs):
        """Plot timeseries stacks of existing data

        see pyam.plotting.stack_plot() for all available options
        """
        # TODO: select only relevant meta columns
        df = self.as_pandas()
        ax = plotting.stack_plot(df, *args, **kwargs)
        return ax

    def bar_plot(self, *args, **kwargs):
        """Plot timeseries bars of existing data

        see pyam.plotting.bar_plot() for all available options
        """
        # TODO: select only relevant meta columns
        df = self.as_pandas()
        ax = plotting.bar_plot(df, *args, **kwargs)
        return ax

    def boxplot(self, *args, **kwargs):
        """Plot boxplot of existing data

        see pyam.plotting.boxplot() for all available options
        """
        df = self.as_pandas()
        ax = plotting.boxplot(df, *args, **kwargs)
        return ax

    def pie_plot(self, *args, **kwargs):
        """Plot a pie chart

        see pyam.plotting.pie_plot() for all available options
        """
        # TODO: select only relevant meta columns
        df = self.as_pandas()
        ax = plotting.pie_plot(df, *args, **kwargs)
        return ax

    def scatter(self, x, y, **kwargs):
        """Plot a scatter chart using meta indicators as columns

        see pyam.plotting.scatter() for all available options
        """
        variables = self.data['variable'].unique()
        xisvar = x in variables
        yisvar = y in variables
        if not xisvar and not yisvar:
            cols = [x, y] + mpl_args_to_meta_cols(self, **kwargs)
            df = self.meta[cols].reset_index()
        elif xisvar and yisvar:
            # filter pivot both and rename
            dfx = (
                self
                .filter(variable=x)
                .as_pandas(meta_cols=mpl_args_to_meta_cols(self, **kwargs))
                .rename(columns={'value': x, 'unit': 'xunit'})
                .set_index(YEAR_IDX)
                .drop('variable', axis=1)
            )
            dfy = (
                self
                .filter(variable=y)
                .as_pandas(meta_cols=mpl_args_to_meta_cols(self, **kwargs))
                .rename(columns={'value': y, 'unit': 'yunit'})
                .set_index(YEAR_IDX)
                .drop('variable', axis=1)
            )
            df = dfx.join(dfy, lsuffix='_left', rsuffix='').reset_index()
        else:
            # filter, merge with meta, and rename value column to match var
            var = x if xisvar else y
            df = (
                self
                .filter(variable=var)
                .as_pandas(meta_cols=mpl_args_to_meta_cols(self, **kwargs))
                .rename(columns={'value': var})
            )
        ax = plotting.scatter(df.dropna(), x, y, **kwargs)
        return ax

    def map_regions(self, map_col, agg=None, copy_col=None, fname=None,
                    region_col=None, remove_duplicates=False, inplace=False):
        """Plot regional data for a single model, scenario, variable, and year

        see pyam.plotting.region_plot() for all available options

        Parameters
        ----------
        map_col : str
            The column used to map new regions to. Common examples include
            iso and 5_region.
        agg : str, optional
            Perform a data aggregation. Options include: sum.
        copy_col : str, optional
            Copy the existing region data into a new column for later use.
        fname : str, optional
            Use a non-default region mapping file
        region_col : string, optional
            Use a non-default column name for regions to map from.
        remove_duplicates : bool, optional
            If there are duplicates in the mapping from one regional level to
            another, then remove these duplicates by counting the most common
            mapped value.
            This option is most useful when mapping from high resolution
            (e.g., model regions) to low resolution (e.g., 5_region).
        inplace : bool, optional
            if True, do operation inplace and return None
        """
        fname = fname or run_control()['region_mapping']['default']
        mapping = read_pandas(Path(fname)).rename(str.lower, axis='columns')
        map_col = map_col.lower()

        ret = self.copy() if not inplace else self
        _df = ret.data
        columns_orderd = _df.columns

        # merge data
        dfs = []
        for model in self.model:
            df = _df[_df['model'] == model]
            _col = region_col or '{}.REGION'.format(model)
            _map = mapping.rename(columns={_col.lower(): 'region'})
            _map = _map[['region', map_col]].dropna().drop_duplicates()
            _map = _map[_map['region'].isin(_df['region'])]
            if remove_duplicates and _map['region'].duplicated().any():
                # find duplicates
                where_dup = _map['region'].duplicated(keep=False)
                dups = _map[where_dup]
                logger.warning("""
                Duplicate entries found for the following regions.
                Mapping will occur only for the most common instance.
                {}""".format(dups['region'].unique()))
                # get non duplicates
                _map = _map[~where_dup]
                # order duplicates by the count frequency
                dups = (dups
                        .groupby(['region', map_col])
                        .size()
                        .reset_index(name='count')
                        .sort_values(by='count', ascending=False)
                        .drop('count', axis=1))
                # take top occurance
                dups = dups[~dups['region'].duplicated(keep='first')]
                # combine them back
                _map = pd.concat([_map, dups])
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
            df = df.groupby(self._LONG_IDX).sum().reset_index()

        ret.data = (df
                    .reindex(columns=columns_orderd)
                    .sort_values(SORT_IDX)
                    .reset_index(drop=True)
                    )
        if not inplace:
            return ret


def _meta_idx(data):
    """Return the 'META_IDX' from data by index"""
    return data[META_IDX].drop_duplicates().set_index(META_IDX).index


def _raise_filter_error(col):
    raise ValueError('filter by `{}` not supported'.format(col))


def _check_rows(rows, check, in_range=True, return_test='any'):
    """Check all rows to be in/out of a certain range and provide testing on
    return values based on provided conditions

    Parameters
    ----------
    rows : pd.DataFrame
        data rows
    check : dict
        dictionary with possible values of 'up', 'lo', and 'year'
    in_range : bool, optional
        check if values are inside or outside of provided range
    return_test : str, optional
        possible values:
            - 'any': default, return scenarios where check passes for any entry
            - 'all': test if all values match checks, if not, return empty set
    """
    valid_checks = set(['up', 'lo', 'year'])
    if not set(check.keys()).issubset(valid_checks):
        msg = 'Unknown checking type: {}'
        raise ValueError(msg.format(check.keys() - valid_checks))
    if 'year' not in check:
        where_idx = set(rows.index)
    else:
        if 'time' in rows.index.names:
            _years = rows.index.get_level_values('time').year
        else:
            _years = rows.index.get_level_values('year')
        where_idx = set(rows.index[_years == check['year']])
        rows = rows.loc[list(where_idx)]

    up_op = rows.values.__le__ if in_range else rows.values.__gt__
    lo_op = rows.values.__ge__ if in_range else rows.values.__lt__

    check_idx = []
    for (bd, op) in [('up', up_op), ('lo', lo_op)]:
        if bd in check:
            check_idx.append(set(rows.index[op(check[bd])]))

    if return_test == 'any':
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
        _df = df[df.index.get_level_values('variable') == var]
        for group in _df.groupby(META_IDX):
            grp_idxs = _check_rows(group[-1], check, **kwargs)
            idxs.append(grp_idxs)
    df = df.loc[itertools.chain(*idxs)]
    return df


def _make_index(df, cols=META_IDX, unique=True):
    """Create an index from the columns/index of a dataframe or series"""
    def _get_col(c):
        try:
            return df.index.get_level_values(c)
        except KeyError:
            return df[c]

    index = list(zip(*[_get_col(col) for col in cols]))
    if unique:
        index = pd.unique(index)
    return pd.MultiIndex.from_tuples(index, names=tuple(cols))


def _empty_iamframe(index):
    """Return an empty IamDataFrame with the correct index columns"""
    return IamDataFrame(pd.DataFrame([], columns=index))


def validate(df, criteria={}, exclude_on_fail=False, **kwargs):
    """Validate scenarios using criteria on timeseries values

    Returns all scenarios which do not match the criteria and prints a log
    message or returns None if all scenarios match the criteria.

    When called with `exclude_on_fail=True`, scenarios in `df` not satisfying
    the criteria will be marked as `exclude=True` (object modified in place).

    Parameters
    ----------
    df : IamDataFrame
    args : passed to :meth:`IamDataFrame.validate`
    kwargs : used for downselecting IamDataFrame
        passed to :meth:`IamDataFrame.filter`
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
    df : IamDataFrame
    args : passed to :meth:`IamDataFrame.require_variable`
    kwargs : used for downselecting IamDataFrame
        passed to :meth:`IamDataFrame.filter`
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
    df : IamDataFrame
    args : passed to :meth:`IamDataFrame.categorize`
    kwargs : used for downselecting IamDataFrame
        passed to :meth:`IamDataFrame.filter`
    """
    fdf = df.filter(**kwargs)
    fdf.categorize(name=name, value=value, criteria=criteria, color=color,
                   marker=marker, linestyle=linestyle)

    # update meta indicators
    if name in df.meta:
        df.meta[name].update(fdf.meta[name])
    else:
        df.meta[name] = fdf.meta[name]


def check_aggregate(df, variable, components=None, exclude_on_fail=False,
                    multiplier=1, **kwargs):
    """Check whether the timeseries values match the aggregation
    of sub-categories

    Parameters
    ----------
    df : IamDataFrame
    args : passed to :meth:`IamDataFrame.check_aggregate`
    kwargs : used for downselecting IamDataFrame
        passed to :meth:`IamDataFrame.filter`
    """
    fdf = df.filter(**kwargs)
    if len(fdf.data) > 0:
        vdf = fdf.check_aggregate(variable=variable, components=components,
                                  exclude_on_fail=exclude_on_fail,
                                  multiplier=multiplier)
        df.meta['exclude'] |= fdf.meta['exclude']  # update if any excluded
        return vdf


def filter_by_meta(data, df, join_meta=False, **kwargs):
    """Filter by and join meta columns from an IamDataFrame to a pd.DataFrame

    Parameters
    ----------
    data : pandas.DataFrame
        :class:`pandas.DataFrame` to which meta columns are to be joined,
        index or columns must include `['model', 'scenario']`
    df : IamDataFrame
        IamDataFrame from which meta columns are filtered and joined (optional)
    join_meta : bool, default False
        join selected columns from `df.meta` on `data`
    kwargs
        meta columns to be filtered/joined, where `col=...` applies filters
        with the given arguments (using :meth:`utils.pattern_match`).
        Using `col=None` joins the column without filtering (setting col
        to nan if `(model, scenario)` not in `df.meta.index`)
    """
    if not set(META_IDX).issubset(data.index.names + list(data.columns)):
        raise ValueError('missing required index dimensions or columns!')

    meta = pd.DataFrame(df.meta[list(set(kwargs) - set(META_IDX))].copy())

    # filter meta by columns
    keep = np.array([True] * len(meta))
    apply_filter = False
    for col, values in kwargs.items():
        if col in META_IDX and values is not None:
            _col = meta.index.get_level_values(0 if col == 'model' else 1)
            keep &= pattern_match(_col, values, has_nan=False)
            apply_filter = True
        elif values is not None:
            keep &= pattern_match(meta[col], values)
        apply_filter |= values is not None
    meta = meta[keep]

    # set the data index to META_IDX and apply filtered meta index
    idx = list(data.index.names) if not data.index.names == [None] else None
    data = data.reset_index().set_index(META_IDX)
    meta = meta.loc[meta.index.intersection(data.index.drop_duplicates())]
    meta.index.names = META_IDX
    if apply_filter:
        data = data.loc[meta.index]
    data.index.names = META_IDX

    # join meta (optional), reset index to format as input arg
    data = data.join(meta) if join_meta else data
    data = data.reset_index().set_index(idx or 'index')
    if idx is None:
        data.index.name = None

    return data


def compare(left, right, left_label='left', right_label='right',
            drop_close=True, **kwargs):
    """Compare the data in two IamDataFrames and return a pandas.DataFrame

    Parameters
    ----------
    left, right : IamDataFrames
        two :class:`IamDataFrame` instances to be compared
    left_label, right_label : str, default `left`, `right`
        column names of the returned :class:`pandas.DataFrame`
    drop_close : bool, optional
        remove all data where `left` and `right` are close
    kwargs : arguments for comparison of values
        passed to :func:`numpy.isclose`
    """
    ret = pd.concat({right_label: right.data.set_index(right._LONG_IDX),
                     left_label: left.data.set_index(left._LONG_IDX)}, axis=1)
    ret.columns = ret.columns.droplevel(1)
    if drop_close:
        ret = ret[~np.isclose(ret[left_label], ret[right_label], **kwargs)]
    return ret[[right_label, left_label]]


def concat(dfs):
    """Concatenate a series of IamDataFrame-like objects

    Parameters
    ----------
    dfs : list of IamDataFrames
        a list of :class:`IamDataFrame` instances
    """
    if isstr(dfs) or not hasattr(dfs, '__iter__'):
        msg = 'Argument must be a non-string iterable (e.g., list or tuple)'
        raise TypeError(msg)

    _df = None
    for df in dfs:
        df = df if isinstance(df, IamDataFrame) else IamDataFrame(df)
        if _df is None:
            _df = df.copy()
        else:
            _df.append(df, inplace=True)
    return _df


def read_datapackage(path, data='data', meta='meta'):
    """Read timeseries data and meta-indicators from frictionless Data Package

    Parameters
    ----------
    path : string or path object
        any valid string path or :class:`pathlib.Path`, |br|
        passed to :class:`datapackage.Package` (|datapackage.Package.docs|)
    data : str, optional
        resource containing timeseries data in IAMC-compatible format
    meta : str, optional
        (optional) resource containing a table of categorization and
        quantitative indicators
    """
    if not HAS_DATAPACKAGE:  # pragma: no cover
        raise ImportError('required package `datapackage` not found!')

    package = Package(path)

    def _get_column_names(x):
        return [i['name'] for i in x.descriptor['schema']['fields']]

    # read `data` table
    resource_data = package.get_resource(data)
    _data = pd.DataFrame(resource_data.read())
    _data.columns = _get_column_names(resource_data)
    df = IamDataFrame(_data)

    # read `meta` table
    if meta in package.resource_names:
        resource_meta = package.get_resource(meta)
        _meta = pd.DataFrame(resource_meta.read())
        _meta.columns = _get_column_names(resource_meta)
        df.meta = _meta.set_index(META_IDX)

    return df
