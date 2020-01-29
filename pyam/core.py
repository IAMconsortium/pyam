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
    HAS_DATAPACKAGE = False

try:
    import ixmp
    has_ix = True
except ImportError:
    has_ix = False

from pyam import plotting
from pyam.logging import deprecation_warning
from pyam.run_control import run_control
from pyam.utils import (
    write_sheet,
    read_file,
    read_pandas,
    format_data,
    sort_data,
    to_int,
    find_depth,
    pattern_match,
    years_match,
    month_match,
    hour_match,
    day_match,
    datetime_match,
    isstr,
    islistable,
    META_IDX,
    YEAR_IDX,
    IAMC_IDX,
    SORT_IDX
)
from pyam.read_ixmp import read_ix
from pyam.timeseries import fill_series
from pyam._aggregate import _aggregate, _aggregate_region, _group_and_agg

logger = logging.getLogger(__name__)


class IamDataFrame(object):
    """This class is a wrapper for timeseries data following the IAMC format.
    It provides a number of diagnostic features (including validation of data,
    completeness of variables provided) as well as visualization
    and plotting tools.

    Parameters
    ----------
    data: ixmp.TimeSeries, ixmp.Scenario, pd.DataFrame or data file
        an instance of an TimeSeries or Scenario (requires `ixmp`),
        or pd.DataFrame or data file with IAMC-format data columns.
        A pd.DataFrame can have the required data as columns or index.
        Support is provided additionally for R-style data columns for years,
        like "X2015", etc.
    kwargs:
        if `value=col`, melt `col` to `value` and use `col` name as `variable`;
        else, mapping of columns required for an `IamDataFrame` to:
        - one column in `df`
        - multiple columns, which will be concatenated by pipe
        - a string to be used as value for this column

    Notes
    -----
    When initializing an :class:`IamDataFrame` from an :code:`xlsx` file,
    :class:`pyam` will per default look for the sheets 'data' and 'meta' to
    populate the respective tables. Custom sheet names can be specified with
    the kwargs :code:`sheet_name` (:code:`data`) and :code:`meta_sheet_name`
    (:code:`meta`). Calling the class with :code:`meta_sheet_name=False` will
    skip the import of the :code:`meta` table.
    """

    def __init__(self, data, **kwargs):
        """Initialize an instance of an IamDataFrame"""
        # import data from pd.DataFrame or read from source
        if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
            _data = format_data(data.copy(), **kwargs)
        elif has_ix and isinstance(data, ixmp.TimeSeries):
            _data = read_ix(data, **kwargs)
        else:
            _data = read_file(data, **kwargs)

        self.data, self.time_col, self.extra_cols = _data
        # cast time_col to desired format
        if self.time_col == 'year':
            self._format_year_col()
        elif self.time_col == 'time':
            self._format_datetime_col()

        self._LONG_IDX = IAMC_IDX + [self.time_col] + self.extra_cols

        # define `meta` dataframe for categorization & quantitative indicators
        self.meta = self.data[META_IDX].drop_duplicates().set_index(META_IDX)
        self.reset_exclude()

        # if initializing from xlsx, try to load `meta` table from file
        meta_sheet = kwargs.get('meta_sheet_name', 'meta')
        if isstr(data) and data.endswith('.xlsx') and meta_sheet is not False\
                and meta_sheet in pd.ExcelFile(data).sheet_names:
            self.load_meta(data, sheet_name=meta_sheet)

        # execute user-defined code
        if 'exec' in run_control():
            self._execute_run_control()

    def _format_year_col(self):
        self.data['year'] = to_int(pd.to_numeric(self.data['year']))

    def _format_datetime_col(self):
        self.data['time'] = pd.to_datetime(self.data['time'])

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

    def copy(self):
        """Make a deepcopy of this object

        Documentation about deepcopy is available
        `here <https://docs.python.org/2/library/copy.html#copy.deepcopy>`_
        """
        return copy.deepcopy(self)

    def head(self, *args, **kwargs):
        """Identical to pd.DataFrame.head() operating on data"""
        return self.data.head(*args, **kwargs)

    def tail(self, *args, **kwargs):
        """Identical to pd.DataFrame.tail() operating on data"""
        return self.data.tail(*args, **kwargs)

    @property
    def empty(self):
        """Indicator whether this object is empty"""
        return self.data.empty

    def equals(self, other):
        """Test if two objects contain the same data and meta indicators

        This function allows two IamdDataFrame instances to be compared against
        each other to see if they have the same timeseries data and meta
        indicators. NaNs in the same location of the meta table are considered
        equal.

        Parameters
        ----------
        other: IamDataFrame
            The other IamDataFrame to be compared with the first.
        """
        if not isinstance(other, IamDataFrame):
            raise ValueError('`other` is not an `IamDataFrame` instance')

        if compare(self, other).empty and self.meta.equals(other.meta):
            return True
        else:
            return False

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

    def append(self, other, ignore_meta_conflict=False, inplace=False,
               **kwargs):
        """Append any castable object to this IamDataFrame.

        Columns in `other.meta` that are not in `self.meta` are always merged,
        duplicate region-variable-unit-year rows raise a ValueError.

        Parameters
        ----------
        other: pyam.IamDataFrame, ixmp.TimeSeries, ixmp.Scenario,
        pd.DataFrame or data file
            An IamDataFrame, TimeSeries or Scenario (requires `ixmp`),
            pandas.DataFrame or data file with IAMC-format data columns
        ignore_meta_conflict : bool, default False
            If False and `other` is an IamDataFrame, raise an error if
            any meta columns present in `self` and `other` are not identical.
        inplace : bool, default False
            If True, do operation inplace and return None
        kwargs are passed through to `IamDataFrame(other, **kwargs)`
        """
        if not isinstance(other, IamDataFrame):
            other = IamDataFrame(other, **kwargs)
            ignore_meta_conflict = True

        if self.time_col is not other.time_col:
            raise ValueError('incompatible time format (years vs. datetime)!')

        ret = copy.deepcopy(self) if not inplace else self

        diff = other.meta.index.difference(ret.meta.index)
        intersect = other.meta.index.intersection(ret.meta.index)

        # merge other.meta columns not in self.meta for existing scenarios
        if not intersect.empty:
            # if not ignored, check that overlapping meta dataframes are equal
            if not ignore_meta_conflict:
                cols = [i for i in other.meta.columns if i in ret.meta.columns]
                if not ret.meta.loc[intersect, cols].equals(
                        other.meta.loc[intersect, cols]):
                    conflict_idx = (
                        pd.concat([ret.meta.loc[intersect, cols],
                                   other.meta.loc[intersect, cols]]
                                  ).drop_duplicates()
                        .index.drop_duplicates()
                    )
                    msg = 'conflict in `meta` for scenarios {}'.format(
                        [i for i in pd.DataFrame(index=conflict_idx).index])
                    raise ValueError(msg)

            cols = [i for i in other.meta.columns if i not in ret.meta.columns]
            _meta = other.meta.loc[intersect, cols]
            ret.meta = ret.meta.merge(_meta, how='outer',
                                      left_index=True, right_index=True)

        # join other.meta for new scenarios
        if not diff.empty:
            # sorting not supported by ` pd.append()`  prior to version 23
            sort_kwarg = {} if int(pd.__version__.split('.')[1]) < 23 \
                else dict(sort=False)
            ret.meta = ret.meta.append(other.meta.loc[diff, :], **sort_kwarg)

        # append other.data (verify integrity for no duplicates)
        _data = ret.data.set_index(sorted(ret._LONG_IDX)).append(
            other.data.set_index(sorted(other._LONG_IDX)),
            verify_integrity=True)

        # merge extra columns in `data` and set `LONG_IDX`
        ret.extra_cols += [i for i in other.extra_cols
                           if i not in ret.extra_cols]
        ret._LONG_IDX = IAMC_IDX + [ret.time_col] + ret.extra_cols
        ret.data = sort_data(_data.reset_index(), ret._LONG_IDX)

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

    def swap_time_for_year(self, inplace=False):
        """Convert the `time` column to `year`.

        Parameters
        ----------
        inplace: bool, default False
            if True, do operation inplace and return None

        Raises
        ------
        ValueError
            "time" is not a column of `self.data`
        """
        if "time" not in self.data:
            raise ValueError("time column must be datetime to use this method")

        ret = self.copy() if not inplace else self

        ret.data["year"] = ret.data["time"].apply(lambda x: x.year)
        ret.data = ret.data.drop("time", axis="columns")
        ret._LONG_IDX = [v if v != "time" else "year" for v in ret._LONG_IDX]

        if any(ret.data[ret._LONG_IDX].duplicated()):
            error_msg = ('swapping time for year will result in duplicate '
                         'rows in `data`!')
            raise ValueError(error_msg)

        if not inplace:
            return ret

    def as_pandas(self, with_metadata=False):
        """Return this as a pd.DataFrame

        Parameters
        ----------
        with_metadata : bool, default False or dict
           if True, join data with all meta columns; if a dict, discover
           meaningful meta columns from values (in key-value)
        """
        if with_metadata:
            cols = self._discover_meta_cols(**with_metadata) \
                if isinstance(with_metadata, dict) else self.meta.columns
            return (
                self.data
                .set_index(META_IDX)
                .join(self.meta[cols])
                .reset_index()
            )
        else:
            return self.data.copy()

    def _discover_meta_cols(self, **kwargs):
        """Return the subset of `kwargs` values (not keys!) matching
        a `meta` column name"""
        cols = set(['exclude'])
        for arg, value in kwargs.items():
            if isstr(value) and value in self.meta.columns:
                cols.add(value)
        return list(cols)

    def timeseries(self, iamc_index=False):
        """Returns a pd.DataFrame in wide format (years or datetime as columns)

        Parameters
        ----------
        iamc_index: bool, default False
            if True, use `['model', 'scenario', 'region', 'variable', 'unit']`;
            else, use all `data` columns

        Raises
        ------
        ValueError
            ``IamDataFrame`` is empty
        ValueError
            reducing to IAMC-index yields an index with duplicates
        """
        if self.empty:
            raise ValueError('this `IamDataFrame` is empty')

        index = IAMC_IDX if iamc_index else IAMC_IDX + self.extra_cols
        df = (
            self.data
            .pivot_table(index=index, columns=self.time_col)
            .value  # column name
            .rename_axis(None, axis=1)
        )
        if df.index.has_duplicates:
            raise ValueError('timeseries object has duplicates in index ',
                             'use `iamc_index=False`')
        return df

    def reset_exclude(self):
        """Reset exclusion assignment for all scenarios to `exclude: False`"""
        self.meta['exclude'] = False

    def set_meta(self, meta, name=None, index=None):
        """Add metadata indicators as pd.Series, list or value (int/float/str)

        Parameters
        ----------
        meta: pd.Series, list, int, float or str
            column to be added to metadata
            (by `['model', 'scenario']` index if possible)
        name: str, optional
            meta column name (defaults to meta pd.Series.name);
            either a meta.name or the name kwarg must be defined
        index: pyam.IamDataFrame, pd.DataFrame or pd.MultiIndex, optional
            index to be used for setting meta column (`['model', 'scenario']`)
        """
        # check that name is valid and doesn't conflict with data columns
        if (name or (hasattr(meta, 'name') and meta.name)) in [None, False]:
            raise ValueError('Must pass a name or use a named pd.Series')
        name = name or meta.name
        if name in self.data.columns:
            raise ValueError('`{}` already exists in `data`!'.format(name))

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
            error = "adding metadata for non-existing scenarios '{}'!"
            raise ValueError(error.format(diff))

        self._new_meta_column(name)
        self.meta[name] = meta[name].combine_first(self.meta[name])

    def set_meta_from_data(self, name, method=None, column='value', **kwargs):
        """Add metadata indicators from downselected timeseries `data` of self

        Parameters
        ----------
        name: str
            meta column name
        method: function, optional
            method for aggregation, required if downselected data do not yield
            unique values (e.g., `numpy.max()`)
        column: str, optional
            the column from `data` to be used to derive the indicator
        kwargs: passed to :meth:`IamDataFrame.filter()` for downselected `data`
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
            logger.info("No scenarios satisfy the criteria")
            return  # EXIT FUNCTION

        # update metadata dataframe
        self._new_meta_column(name)
        self.meta.loc[idx, name] = value
        msg = '{} scenario{} categorized as `{}: {}`'
        logger.info(msg.format(len(idx), '' if len(idx) == 1 else 's',
                                 name, value))

    def _new_meta_column(self, name):
        """Add a column to meta if it doesn't exist, set to value `np.nan`"""
        if name is None:
            raise ValueError('cannot add a meta column `{}`'.format(name))
        if name not in self.meta:
            self.meta[name] = np.nan

    def require_variable(self, variable, unit=None, year=None,
                         exclude_on_fail=False):
        """Check whether all scenarios have a required variable

        Parameters
        ----------
        variable: str
            required variable
        unit: str, default None
            name of unit (optional)
        year: int or list, default None
            years (optional)
        exclude_on_fail: bool, default False
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
            msg += ', marked as `exclude: True` in metadata'

        logger.info(msg.format(n, variable))
        return pd.DataFrame(index=idx).reset_index()

    def validate(self, criteria={}, exclude_on_fail=False):
        """Validate scenarios using criteria on timeseries values

        Returns all scenarios which do not match the criteria and prints a log
        message or returns None if all scenarios match the criteria.

        When called with `exclude_on_fail=True`, scenarios in the object not
        satisfying the criteria will be marked as `exclude=True`.

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
            msg = '{} of {} data points do not satisfy the criteria'
            logger.info(msg.format(len(df), len(self.data)))

            if exclude_on_fail and len(df) > 0:
                self._exclude_on_fail(df)

            return df

    def rename(self, mapping=None, inplace=False, append=False,
               check_duplicates=True, **kwargs):
        """Rename and aggregate column entries using `groupby.sum()` on values.
        When renaming models or scenarios, the uniqueness of the index must be
        maintained, and the function will raise an error otherwise.

        Renaming is only applied to any data where a filter matches for all
        columns given in `mapping`. Renaming can only be applied to the `model`
        and `scenario` columns or to other data columns simultaneously.

        Parameters
        ----------
        mapping: dict or kwargs
            mapping of column name to rename-dictionary of that column
            >> {<column_name>: {<current_name_1>: <target_name_1>,
            >>                  <current_name_2>: <target_name_2>}}
            or kwargs as `column_name={<current_name_1>: <target_name_1>, ...}`
        inplace: bool, default False
            if True, do operation inplace and return None
        append: bool, default False
            append renamed timeseries to self; else, return new `IamDataFrame`
        check_duplicates: bool, default True
            check whether conflict between existing and renamed data exists.
            If True, raise ValueError; if False, rename and merge
            with `groupby().sum()`.
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
        ret = copy.deepcopy(self) if not inplace else self

        # renaming is only applied where a filter matches for all given columns
        rows = ret._apply_filters(**filters)
        idx = ret.meta.index.isin(_make_index(ret.data[rows]))

        # if `check_duplicates`, do the rename on a copy until after the check
        _data = ret.data.copy() if check_duplicates else ret.data

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

    def normalize(self, inplace=False, **kwargs):
        """Normalize data to a specific data point

        Currently only supports normalizing to a specific time.

        Parameters
        ----------
        inplace: bool, default False
            if True, do operation inplace and return None
        kwargs: the values on which to normalize (e.g., `year=2005`)
        """
        if len(kwargs) > 1 or self.time_col not in kwargs:
            raise ValueError('Only time(year)-based normalization supported')
        ret = copy.deepcopy(self) if not inplace else self
        df = ret.data
        # change all below if supporting more in the future
        cols = self.time_col
        value = kwargs[self.time_col]
        x = df.set_index(IAMC_IDX)
        x['value'] /= x[x[cols] == value]['value']
        ret.data = x.reset_index()
        if not inplace:
            return ret

    def aggregate(self, variable, components=None, method='sum', append=False):
        """Aggregate timeseries components or sub-categories within each region

        Parameters
        ----------
        variable: str or list of str
            variable(s) for which the aggregate will be computed
        components: list of str, default None
            list of variables, defaults to all sub-categories of `variable`
        method: func or str, default 'sum'
            method to use for aggregation, e.g. np.mean, np.sum, 'min', 'max'
        append: bool, default False
            append the aggregate timeseries to `self` and return None,
            else return aggregate timeseries
        """
        _df = _aggregate(self, variable, components=components, method=method)

        # return None if there is nothing to aggregate
        if _df is None:
            return None

        # else, append to `self` or return as `IamDataFrame`
        if append is True:
            self.append(_df, inplace=True)
        else:
            return IamDataFrame(_df)

    def check_aggregate(self, variable, components=None, method='sum',
                        exclude_on_fail=False, multiplier=1, **kwargs):
        """Check whether a timeseries matches the aggregation of its components

        Parameters
        ----------
        variable: str or list of str
            variable(s) checked for matching aggregation of sub-categories
        components: list of str, default None
            list of variables, defaults to all sub-categories of `variable`
        method: func or str, default 'sum'
            method to use for aggregation, e.g. np.mean, np.sum, 'min', 'max'
        exclude_on_fail: boolean, default False
            flag scenarios failing validation as `exclude: True`
        multiplier: number, default 1
            factor when comparing variable and sum of components
        kwargs: passed to `np.isclose()`
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
        defined at the  `region` level by setting `components=True`

        Parameters
        ----------
        variable: str or list of str
            variable(s) to be aggregated
        region: str, default 'World'
            region to which data will be aggregated
        subregions: list of str
            list of subregions, defaults to all regions other than `region`
        components: bool or list of str, default False
            variables at the `region` level to be included in the aggregation
            (ignored if False); if `True`, use all sub-categories of `variable`
            included in `region` but not in any of the `subregions`;
            or explicit list of variables
        method: func or str, default 'sum'
            method to use for aggregation, e.g. np.mean, np.sum, 'min', 'max'
        weight: str, default None
            variable to use as weight for the aggregation
            (currently only supported with `method='sum'`)
        append: bool, default False
            append the aggregate timeseries to `self` and return None,
            else return aggregate timeseries
        """
        _df = _aggregate_region(
            self, variable, region=region, subregions=subregions,
            components=components, method=method, weight=weight
        )

        # return None if there is nothing to aggregate
        if _df is None:
            return None

        # else, append to `self` or return as `IamDataFrame`
        if append is True:
            self.append(_df, region=region, inplace=True)
        else:
            return IamDataFrame(_df, region=region)

    def check_aggregate_region(self, variable, region='World', subregions=None,
                               components=False, method='sum', weight=None,
                               exclude_on_fail=False, **kwargs):
        """Check whether a timeseries matches the aggregation across subregions

        Parameters
        ----------
        variable: str or list of str
            variable(s) to be checked for matching aggregation of subregions
        region: str, default 'World'
            region to be checked for matching aggregation of subregions
        subregions: list of str
            list of subregions, defaults to all regions other than `region`
        components: bool or list of str, default False
            variables at the `region` level to be included in the aggregation
            (ignored if False); if `True`, use all sub-categories of `variable`
            included in `region` but not in any of the `subregions`;
            or explicit list of variables
        method: func or str, default 'sum'
            method to use for aggregation, e.g. np.mean, np.sum, 'min', 'max'
        weight: str, default None
            variable to use as weight for the aggregation
            (currently only supported with `method='sum'`)
        exclude_on_fail: boolean, default False
            flag scenarios failing validation as `exclude: True`
        kwargs: passed to `np.isclose()`
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

    def downscale_region(self, variable, proxy, region='World',
                         subregions=None, append=False):
        """Downscale a timeseries to a number of subregions

        Parameters
        ----------
        variable: str or list of str
            variable(s) to be downscaled
        proxy: str
            variable to be used as proxy (i.e, weight) for the downscaling
        region: str, default 'World'
            region from which data will be downscaled
        subregions: list of str
            list of subregions, defaults to all regions other than `region`
        append: bool, default False
            append the downscaled timeseries to `self` and return None,
            else return downscaled data as new `IamDataFrame`
        """
        # get default subregions if not specified
        subregions = subregions or self._all_other_regions(region)

        # filter relevant data, transform to `pd.Series` with appropriate index
        _df = self.data[self._apply_filters(variable=proxy, region=subregions)]
        _proxy = _df.set_index(self._get_cols(['region', self.time_col])).value
        _total = _df.groupby(self._get_cols([self.time_col])).value.sum()

        _value = (
            self.data[self._apply_filters(variable=variable, region=region)]
            .set_index(self._get_cols(['variable', 'unit', self.time_col]))
            .value
        )

        # compute downscaled data
        _data = _value * _proxy / _total

        if append is True:
            self.append(_data, inplace=True)
        else:
            df = IamDataFrame(_data)
            df.meta = self.meta.loc[_make_index(df.data)]
            return df

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
        see [this issue](https://github.com/IAMconsortium/pyam/issues/106).

        Parameters
        ----------
        kwargs: passed to `np.isclose()`
        components: bool, default False
            passed to `check_aggregate_region)`: if `True`, use all
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
        idx = df if isinstance(df, pd.MultiIndex) else _meta_idx(df)
        self.meta.loc[idx, 'exclude'] = True
        logger.info('{} non-valid scenario{} will be excluded'
                      .format(len(idx), '' if len(idx) == 1 else 's'))

    def filter(self, keep=True, inplace=False, **kwargs):
        """Return a filtered IamDataFrame (i.e., a subset of current data)

        Parameters
        ----------
        keep: bool, default True
            keep all scenarios satisfying the filters (if True) or the inverse
        inplace: bool, default False
            if True, do operation inplace and return None
        filters by kwargs:
            The following columns are available for filtering:
             - metadata columns: filter by category assignment
             - 'model', 'scenario', 'region', 'variable', 'unit':
               string or list of strings, where `*` can be used as a wildcard
             - 'level': the maximum "depth" of IAM variables (number of '|')
               (excluding the strings given in the 'variable' argument)
             - 'year': takes an integer, a list of integers or a range
               note that the last year of a range is not included,
               so `range(2010, 2015)` is interpreted as `[2010, ..., 2014]`
             - arguments for filtering by `datetime.datetime`
               ('month', 'hour', 'time')
             - 'regexp=True' disables pseudo-regexp syntax in `pattern_match()`
        """
        if not isinstance(keep, bool):
            msg = '`filter(keep={}, ...)` is not valid, must be boolean'
            raise ValueError(msg.format(keep))

        _keep = self._apply_filters(**kwargs)
        _keep = _keep if keep else ~_keep
        ret = copy.deepcopy(self) if not inplace else self
        ret.data = ret.data[_keep]

        idx = _make_index(ret.data)
        if len(idx) == 0:
            logger.warning('Filtered IamDataFrame is empty!')
        ret.meta = ret.meta.loc[idx]
        if not inplace:
            return ret

    def _apply_filters(self, **filters):
        """Determine rows to keep in data for given set of filters

        Parameters
        ----------
        filters: dict
            dictionary of filters ({col: values}}); uses a pseudo-regexp syntax
            by default, but accepts `regexp: True` to use regexp directly
        """
        regexp = filters.pop('regexp', False)
        keep = np.array([True] * len(self.data))

        # filter by columns and list of values
        for col, values in filters.items():
            # treat `_apply_filters(col=None)` as no filter applied
            if values is None:
                continue

            if col in self.meta.columns:
                matches = pattern_match(self.meta[col], values, regexp=regexp)
                cat_idx = self.meta[matches].index
                keep_col = (self.data[META_IDX].set_index(META_IDX)
                                .index.isin(cat_idx))

            elif col == 'variable':
                level = filters['level'] if 'level' in filters else None
                keep_col = pattern_match(self.data[col], values, level, regexp)

            elif col == 'year':
                _data = self.data[col] if self.time_col is not 'time' \
                    else self.data['time'].apply(lambda x: x.year)
                keep_col = years_match(_data, values)

            elif col == 'month' and self.time_col is 'time':
                keep_col = month_match(self.data['time']
                                           .apply(lambda x: x.month),
                                       values)

            elif col == 'day' and self.time_col is 'time':
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

            elif col == 'hour' and self.time_col is 'time':
                keep_col = hour_match(self.data['time']
                                          .apply(lambda x: x.hour),
                                      values)

            elif col == 'time' and self.time_col is 'time':
                keep_col = datetime_match(self.data[col], values)

            elif col == 'level':
                if 'variable' not in filters.keys():
                    keep_col = find_depth(self.data['variable'], level=values)
                else:
                    continue

            elif col in self.data.columns:
                keep_col = pattern_match(self.data[col], values, regexp=regexp)

            else:
                _raise_filter_error(col)

            keep = np.logical_and(keep, keep_col)

        return keep

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

    def _to_file_format(self, iamc_index):
        """Return a dataframe suitable for writing to a file"""
        df = self.timeseries(iamc_index=iamc_index).reset_index()
        df = df.rename(columns={c: str(c).title() for c in df.columns})
        return df

    def to_csv(self, path, iamc_index=False, **kwargs):
        """Write timeseries data of this object to a csv file

        Parameters
        ----------
        path: string
            file path
        iamc_index: bool, default False
            if True, use `['model', 'scenario', 'region', 'variable', 'unit']`;
            else, use all `data` columns
        """
        self._to_file_format(iamc_index).to_csv(path, index=False, **kwargs)

    def to_excel(self, excel_writer, sheet_name='data', iamc_index=False,
                 include_meta=True, **kwargs):
        """Write object to an Excel sheet

        Parameters
        ----------
        excel_writer: string or ExcelWriter object
            file path or existing ExcelWriter
        sheet_name: string
            name of sheet which will contain ``IamDataFrame.timeseries()`` data
        iamc_index: bool, default False
            if True, use :code:`['model', 'scenario', 'region', 'variable',
            'unit']`; else, use all :code:`data` columns
        include_meta: boolean or string
            if True, write :code:`meta` to an Excel sheet 'meta' (default);
            if this is a string, use it as sheet name
        """
        close = False
        if not isinstance(excel_writer, pd.ExcelWriter):
            close = True
            excel_writer = pd.ExcelWriter(excel_writer)
        self._to_file_format(iamc_index)\
            .to_excel(excel_writer, sheet_name=sheet_name, index=False,
                      **kwargs)
        # replace `True` by `'meta'` as default sheet name
        include_meta = 'meta' if include_meta is True else include_meta
        if isstr(include_meta):
            write_sheet(excel_writer, include_meta, self.meta, index=True)

        # close the file if `excel_writer` arg was a file name
        if close:
            excel_writer.close()

    def export_metadata(self, excel_writer, sheet_name='meta'):
        """Deprecated, see :method:`export_meta()`"""
        # TODO: deprecate in next release (>=0.5.0)
        deprecation_warning('Use `export_meta()` instead!')
        self.export_meta(excel_writer, sheet_name='meta')

    def export_meta(self, excel_writer, sheet_name='meta'):
        """Write the ``meta`` table of this object to an Excel sheet

        Parameters
        ----------
        excel_writer: string or ExcelWriter object
            file path or existing ExcelWriter
        sheet_name: string
            name of sheet which will contain ``IamDataFrame.meta`` table
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

        Returns the saved :class:`datapackage.Package`. When adding metadata
        (descriptors), please follow the `template` defined by
        https://github.com/OpenEnergyPlatform/metadata

        Parameters
        ----------
        path: string or pathlib.Path
            file path
        """

        if not HAS_DATAPACKAGE:
            raise ImportError('required package `datapackage` not found!')

        with TemporaryDirectory(dir='.') as tmp:
            # save data and meta tables to a temporary folder
            self.data.to_csv(Path(tmp) / 'data.csv', index=False)
            self.meta.to_csv(Path(tmp) / 'meta.csv')

            # cast tables to datapackage
            package = Package()
            package.infer('{}/*.csv'.format(tmp))
            if not package.valid:
                logger.warning('the exported package is not valid')
            package.save(path)

        # return the package (needs to reloaded because `tmp` was deleted)
        return Package(path)

    def load_metadata(self, path, *args, **kwargs):
        """Deprecated, see :method:`load_meta()`"""
        # TODO: deprecate in next release (>=0.5.0)
        deprecation_warning('Use `load_meta()` instead!')
        self.load_meta(path, *args, **kwargs)

    def load_meta(self, path, *args, **kwargs):
        """Load ``meta`` table  exported from a :class:`IamDataFrame` instance

        Parameters
        ----------
        path: string
            xlsx or csv file path to ``meta`` table
        """
        if path.endswith('csv'):
            df = pd.read_csv(path, *args, **kwargs)
        else:
            xl = pd.ExcelFile(path)
            if len(xl.sheet_names) > 1 and 'sheet_name' not in kwargs:
                kwargs['sheet_name'] = 'meta'
            df = pd.read_excel(path, *args, **kwargs)

        req_cols = ['model', 'scenario', 'exclude']
        if not set(req_cols).issubset(set(df.columns)):
            e = 'File `{}` does not have required columns ({})!'
            raise ValueError(e.format(path, req_cols))

        # set index, filter to relevant scenarios from imported metadata file
        df.set_index(META_IDX, inplace=True)
        idx = self.meta.index.intersection(df.index)

        n_invalid = len(df) - len(idx)
        if n_invalid > 0:
            msg = 'Ignoring {} scenario{} from imported metadata'
            logger.info(msg.format(n_invalid, 's' if n_invalid > 1 else ''))

        if idx.empty:
            raise ValueError('No valid scenarios in imported metadata file!')

        df = df.loc[idx]

        # merge in imported metadata
        msg = 'Importing metadata for {} scenario{} (for total of {})'
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
        df = self.as_pandas(with_metadata=kwargs)

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

        ax, handles, labels = plotting.line_plot(
            df.dropna(), x=x, y=y, **kwargs)
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

    def scatter(self, x, y, **kwargs):
        """Plot a scatter chart using metadata columns

        see pyam.plotting.scatter() for all available options
        """
        variables = self.data['variable'].unique()
        xisvar = x in variables
        yisvar = y in variables
        if not xisvar and not yisvar:
            cols = [x, y] + self._discover_meta_cols(**kwargs)
            df = self.meta[cols].reset_index()
        elif xisvar and yisvar:
            # filter pivot both and rename
            dfx = (
                self
                .filter(variable=x)
                .as_pandas(with_metadata=kwargs)
                .rename(columns={'value': x, 'unit': 'xunit'})
                .set_index(YEAR_IDX)
                .drop('variable', axis=1)
            )
            dfy = (
                self
                .filter(variable=y)
                .as_pandas(with_metadata=kwargs)
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
                .as_pandas(with_metadata=kwargs)
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
        remove_duplicates: bool, optional, default: False
            If there are duplicates in the mapping from one regional level to
            another, then remove these duplicates by counting the most common
            mapped value.
            This option is most useful when mapping from high resolution
            (e.g., model regions) to low resolution (e.g., 5_region).
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
    """Return the `META_IDX` from `data` by index or columns"""
    return data[META_IDX].drop_duplicates().set_index(META_IDX).index


def _raise_filter_error(col):
    raise ValueError('filter by `{}` not supported'.format(col))


def _check_rows(rows, check, in_range=True, return_test='any'):
    """Check all rows to be in/out of a certain range and provide testing on
    return values based on provided conditions

    Parameters
    ----------
    rows: pd.DataFrame
        data rows
    check: dict
        dictionary with possible values of 'up', 'lo', and 'year'
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

    if 'year' not in rows:
        rows = rows.copy()
        rows['year'] = rows['time'].apply(lambda x: x.year)

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


def _make_index(df, cols=META_IDX):
    """Create an index from the columns of a dataframe"""
    return pd.MultiIndex.from_tuples(
        pd.unique(list(zip(*[df[col] for col in cols]))), names=tuple(cols))


def validate(df, criteria={}, exclude_on_fail=False, **kwargs):
    """Validate scenarios using criteria on timeseries values

    Returns all scenarios which do not match the criteria and prints a log
    message or returns None if all scenarios match the criteria.

    When called with `exclude_on_fail=True`, scenarios in `df` not satisfying
    the criteria will be marked as `exclude=True` (object modified in place).

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


def check_aggregate(df, variable, components=None, exclude_on_fail=False,
                    multiplier=1, **kwargs):
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
                                  exclude_on_fail=exclude_on_fail,
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
        meta columns to be filtered/joined, where `col=...` applies filters
        by the given arguments (using `utils.pattern_match()`) and `col=None`
        joins the column without filtering (setting col to `np.nan`
        if `(model, scenario) not in df.meta.index`)
    """
    if not set(META_IDX).issubset(data.index.names + list(data.columns)):
        raise ValueError('missing required index dimensions or columns!')

    meta = pd.DataFrame(df.meta[list(set(kwargs) - set(META_IDX))].copy())

    # filter meta by columns
    keep = np.array([True] * len(meta))
    apply_filter = False
    for col, values in kwargs.items():
        if col in META_IDX and values is not None:
            _col = meta.index.get_level_values(0 if col is 'model' else 1)
            keep &= pattern_match(_col, values, has_nan=False)
            apply_filter = True
        elif values is not None:
            keep &= pattern_match(meta[col], values)
        apply_filter |= values is not None
    meta = meta[keep]

    # set the data index to META_IDX and apply filtered meta index
    data = data.copy()
    idx = list(data.index.names) if not data.index.names == [None] else None
    data = data.reset_index().set_index(META_IDX)
    meta = meta.loc[meta.index.intersection(data.index)]
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
    """Compare the data in two IamDataFrames and return a pd.DataFrame

    Parameters
    ----------
    left, right: IamDataFrames
        the IamDataFrames to be compared
    left_label, right_label: str, default `left`, `right`
        column names of the returned dataframe
    drop_close: bool, default True
        remove all data where `left` and `right` are close
    kwargs: passed to `np.isclose()`
    """
    ret = pd.concat({right_label: right.data.set_index(right._LONG_IDX),
                     left_label: left.data.set_index(left._LONG_IDX)}, axis=1)
    ret.columns = ret.columns.droplevel(1)
    if drop_close:
        ret = ret[~np.isclose(ret[left_label], ret[right_label], **kwargs)]
    return ret[[right_label, left_label]]


def concat(dfs):
    """Concatenate a series of `pyam.IamDataFrame`-like objects together"""
    if isstr(dfs) or not hasattr(dfs, '__iter__'):
        msg = 'Argument must be a non-string iterable (e.g., list or tuple)'
        raise TypeError(msg)

    _df = None
    for df in dfs:
        df = df if isinstance(df, IamDataFrame) else IamDataFrame(df)
        if _df is None:
            _df = copy.deepcopy(df)
        else:
            _df.append(df, inplace=True)
    return _df


def read_datapackage(path, data='data', meta='meta'):
    """Read timeseries data and meta-indicators from frictionless Data Package

    Parameters
    ----------
    path: path to Data Package
        passed to ``datapackage.Package()``
    data: str, default `data`
        resource containing timeseries data in IAMC-compatible format
    meta: str, default `meta`
        (optional) resource containing a table of categorization and
        quantitative indicators
    """

    if not HAS_DATAPACKAGE:
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
