import copy
import importlib
import itertools
import logging
import os
import sys

import numpy as np
import pandas as pd
from pandas.api.types import is_integer

from pathlib import Path
from tempfile import TemporaryDirectory

from pyam.slice import IamSlice
from pyam.filter import filter_by_time_domain, filter_by_year, filter_by_dt_arg

try:
    from datapackage import Package

    HAS_DATAPACKAGE = True
except ImportError:
    Package = None
    HAS_DATAPACKAGE = False

from pyam.run_control import run_control
from pyam.utils import (
    write_sheet,
    read_file,
    read_pandas,
    format_data,
    merge_meta,
    find_depth,
    pattern_match,
    to_list,
    isstr,
    islistable,
    print_list,
    s,
    DEFAULT_META_INDEX,
    META_IDX,
    IAMC_IDX,
    SORT_IDX,
    ILLEGAL_COLS,
)
from pyam.filter import (
    datetime_match,
)
from pyam.plotting import PlotAccessor
from pyam.compute import IamComputeAccessor
from pyam._compare import _compare
from pyam.aggregation import (
    _aggregate,
    _aggregate_region,
    _aggregate_time,
    _aggregate_recursive,
    _group_and_agg,
)
from pyam._ops import _op_data
from pyam.units import convert_unit
from pyam.index import (
    get_index_levels,
    get_index_levels_codes,
    get_keep_col,
    verify_index_integrity,
    replace_index_values,
)
from pyam.time import swap_time_for_year, swap_year_for_time
from pyam.logging import raise_data_error, deprecation_warning

logger = logging.getLogger(__name__)


class IamDataFrame(object):
    """Scenario timeseries data and meta indicators

    The class provides a number of diagnostic features (including validation of
    data, completeness of variables provided), processing tools (e.g.,
    unit conversion), as well as visualization and plotting tools.

    Parameters
    ----------
    data : :class:`pandas.DataFrame` or file-like object as str or :class:`pathlib.Path`
        Scenario timeseries data following the IAMC data format or
        a supported variation as pandas object or a path to a file.
    meta : :class:`pandas.DataFrame`, optional
        A dataframe with suitable 'meta' indicators for the new instance.
        The index will be downselected to scenarios present in `data`.
    index : list, optional
        Columns to use for resulting IamDataFrame index.
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
    |pyam| will per default parse all sheets starting with 'data' or 'Data'
    for timeseries and a sheet 'meta' to populate the respective table.
    Sheet names can be specified with kwargs :code:`sheet_name` ('data')
    and :code:`meta_sheet_name` ('meta'), where
    values can be a string or a list and '*' is interpreted as a wildcard.
    Calling the class with :code:`meta_sheet_name=False` will
    skip the import of the 'meta' table.

    When initializing an :class:`IamDataFrame` from an object that is already
    an :class:`IamDataFrame` instance, the new object will be hard-linked to
    all attributes of the original object - so any changes on one object
    (e.g., with :code:`inplace=True`) may also modify the other object!
    This is intended behaviour and consistent with pandas but may be confusing
    for those who are not used to the pandas/Python universe.
    """

    def __init__(self, data, meta=None, index=DEFAULT_META_INDEX, **kwargs):
        """Initialize an instance of an IamDataFrame"""
        if isinstance(data, IamDataFrame):
            if kwargs:
                raise ValueError(
                    f"Invalid arguments for initializing from IamDataFrame: {kwargs}"
                )
            if index != data.index.names:
                msg = f"Incompatible `index={index}` with {type(data)} "
                raise ValueError(msg + f"(index={data.index.names})")
            for attr, value in data.__dict__.items():
                setattr(self, attr, value)
        else:
            self._init(data, meta, index=index, **kwargs)

    def _init(self, data, meta=None, index=DEFAULT_META_INDEX, **kwargs):
        """Process data and set attributes for new instance"""

        # pop kwarg for meta_sheet_name (prior to reading data from file)
        meta_sheet = kwargs.pop("meta_sheet_name", "meta")

        # if meta is given explicitly, verify that index matches
        if meta is not None and not meta.index.names == index:
            raise ValueError(
                f"Incompatible `index={index}` with `meta` (index={meta.index.names})!"
            )

        # try casting to Path if file-like is string or LocalPath or pytest.LocalPath
        try:
            data = Path(data)
        except TypeError:
            pass

        # read from file
        if isinstance(data, Path):
            data = Path(data)  # casting str or LocalPath to Path
            if not data.is_file():
                raise FileNotFoundError(f"No such file: '{data}'")
            logger.info(f"Reading file {data}")
            _data = read_file(data, index=index, **kwargs)

        # cast data from pandas
        elif isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
            _data = format_data(data.copy(), index=index, **kwargs)

        # unsupported `data` args
        elif islistable(data):
            raise ValueError(
                "Initializing from list is not supported, "
                "use `IamDataFrame.append()` or `pyam.concat()`"
            )
        else:
            raise ValueError("IamDataFrame constructor not properly called!")

        self._data, index, self.time_col, self.extra_cols = _data

        # define `meta` dataframe for categorization & quantitative indicators
        self.meta = pd.DataFrame(index=_make_index(self._data, cols=index))
        self.reset_exclude()

        # if given explicitly, merge meta dataframe after downselecting
        if meta is not None:
            self.set_meta(meta)

        # if initializing from xlsx, try to load `meta` table from file
        if meta_sheet and isinstance(data, Path) and data.suffix in [".xlsx", ".xls"]:
            excel_file = pd.ExcelFile(data)
            if meta_sheet in excel_file.sheet_names:
                self.load_meta(excel_file, sheet_name=meta_sheet, ignore_conflict=True)

        self._set_attributes()

        # execute user-defined code
        if "exec" in run_control():
            self._execute_run_control()

        # add the `plot` and `compute` handlers
        self.plot = PlotAccessor(self)
        self._compute = None

    def _set_attributes(self):
        """Utility function to set attributes, called on __init__/filter/append/..."""

        # add/reset internal time-index attribute (set when first using `time`)
        setattr(self, "_time", None)

        # add/reset year attribute (only if time domain is year, i.e., all integer)
        if self.time_col == "year":
            setattr(self, "year", get_index_levels(self._data, "year"))

        #  add/reset internal time domain attribute (set when first using `time_domain`)
        setattr(self, "_time_domain", None)

        # set non-standard index columns as attributes
        for c in self.meta.index.names:
            if c not in META_IDX:
                setattr(self, c, get_index_levels(self.meta, c))

        # set extra data columns as attributes
        for c in self.extra_cols:
            setattr(self, c, get_index_levels(self._data, c))

    def _finalize(self, data, append, **args):
        """Append `data` to `self` or return as new IamDataFrame with copy of `meta`"""
        if append:
            self.append(data, **args, inplace=True)
        else:
            if data is None or data.empty:
                return _empty_iamframe(self.dimensions + ["value"])
            return IamDataFrame(data, meta=self.meta, **args)

    def __getitem__(self, key):
        _key_check = [key] if isstr(key) else key
        if isinstance(key, IamSlice):
            return IamDataFrame(self._data.loc[key])
        elif key == "value":
            return pd.Series(self._data.values, name="value")
        elif set(_key_check).issubset(self.meta.columns):
            return self.meta.__getitem__(key)
        else:
            return self.get_data_column(key)

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        return self.info()

    @property
    def compute(self):
        """Access to advanced computation methods, see :class:`IamComputeAccessor`"""
        if self._compute is None:
            self._compute = IamComputeAccessor(self)
        return self._compute

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
        info = f"{type(self)}\nIndex:\n"
        c1 = max([len(i) for i in self.dimensions]) + 1
        c2 = n - c1 - 5
        info += "\n".join(
            [
                f" * {i:{c1}}: {print_list(getattr(self, i), c2)}"
                for i in self.index.names
            ]
        )

        # concatenate list of index of _data (not in index.names)
        info += "\nTimeseries data coordinates:\n"
        info += "\n".join(
            [
                f"   {i:{c1}}: {print_list(getattr(self, i), c2)}"
                for i in self.dimensions
                if i not in self.index.names
            ]
        )

        # concatenate list of (head of) meta indicators and levels/values
        def print_meta_row(m, t, lst):
            _lst = print_list(lst, n - len(m) - len(t) - 7)
            return f"   {m} ({t}) {_lst}"

        info += "\nMeta indicators:\n"
        info += "\n".join(
            [
                print_meta_row(m, t, self.meta[m].unique())
                for m, t in zip(
                    self.meta.columns[0:meta_rows], self.meta.dtypes[0:meta_rows]
                )
            ]
        )
        # print `...` if more than `meta_rows` columns
        if len(self.meta.columns) > meta_rows:
            info += "\n   ..."

        # add info on size (optional)
        if memory_usage:
            size = self._data.memory_usage() + sum(self.meta.memory_usage())
            info += f"\nMemory usage: {size} bytes"

        return info

    def _execute_run_control(self):
        for module_block in run_control()["exec"]:
            fname = module_block["file"]
            functions = module_block["functions"]

            dirname = os.path.dirname(fname)
            if dirname:
                sys.path.append(dirname)

            module = os.path.basename(fname).split(".")[0]
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
        return self._get_meta_index_levels("model")

    @property
    def scenario(self):
        """Return the list of (unique) scenario names"""
        return self._get_meta_index_levels("scenario")

    def _get_meta_index_levels(self, name):
        """Return the list of a level from meta"""
        if name in self.meta.index.names:
            return get_index_levels(self.meta, name)
        # in case of non-standard meta.index.names
        raise KeyError(f"Index `{name}` does not exist!")

    @property
    def region(self):
        """Return the list of (unique) regions"""
        return get_index_levels(self._data, "region")

    @property
    def variable(self):
        """Return the list of (unique) variables"""
        return get_index_levels(self._data, "variable")

    @property
    def unit(self):
        """Return the list of (unique) units"""
        return get_index_levels(self._data, "unit")

    @property
    def unit_mapping(self):
        """Return a dictionary of variables to (list of) corresponding units"""

        def list_or_str(x):
            x = list(x.drop_duplicates())
            return x if len(x) > 1 else x[0]

        return (
            pd.DataFrame(
                zip(self.get_data_column("variable"), self.get_data_column("unit")),
                columns=["variable", "unit"],
            )
            .groupby("variable")
            .apply(lambda u: list_or_str(u.unit))
            .to_dict()
        )

    @property
    def time(self):
        """The time index, i.e., axis labels related to the time domain.

        Returns
        -------
        - A :class:`pandas.Int64Index` if the :attr:`time_domain` is 'year'
        - A :class:`pandas.DatetimeIndex` if the :attr:`time_domain` is 'datetime'
        - A :class:`pandas.Index` if the :attr:`time_domain` is 'mixed'
        """
        if self._time is None:
            self._time = pd.Index(
                self._data.index.unique(level=self.time_col).values, name="time"
            )

        return self._time

    @property
    def data(self):
        """Return the timeseries data as a long :class:`pandas.DataFrame`"""
        if self.empty:  # reset_index fails on empty with `datetime` column
            return pd.DataFrame([], columns=self.dimensions + ["value"])
        return self._data.reset_index()

    def get_data_column(self, column):
        """Return a `column` from the timeseries data in long format

        Equivalent to :meth:`IamDataFrame.data[column] <IamDataFrame.data>`.

        Parameters
        ----------
        column : str
            The column name.

        Returns
        -------
        pd.Series
        """
        return pd.Series(self._data.index.get_level_values(column), name=column)

    @property
    def dimensions(self):
        """Return the list of `data` columns (index names & data coordinates)"""
        return list(self._data.index.names)

    @property
    def coordinates(self):
        """Return the list of `data` coordinates (columns not including index names)"""
        return [i for i in self._data.index.names if i not in self.index.names]

    @property
    def time_domain(self):
        """Indicator of the time domain: 'year', 'datetime', or 'mixed'"""
        if self._time_domain is None:
            if self.time_col == "year":
                self._time_domain = "year"
            elif isinstance(self.time, pd.DatetimeIndex):
                self._time_domain = "datetime"
            else:
                self._time_domain = "mixed"

        return self._time_domain

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
            raise ValueError("`other` is not an `IamDataFrame` instance")

        if compare(self, other).empty and self.meta.equals(other.meta):
            return True
        else:
            return False

    def append(
        self,
        other,
        ignore_meta_conflict=False,
        inplace=False,
        verify_integrity=True,
        **kwargs,
    ):
        """Append any IamDataFrame-like object to this object

        Indicators in `other.meta` that are not in `self.meta` are merged.
        Missing values are set to `NaN`.
        Conflicting `data` rows always raise a `ValueError`.

        Parameters
        ----------
        other : IamDataFrame, pandas.DataFrame or data file
            Any object castable as IamDataFrame to be appended
        ignore_meta_conflict : bool, optional
            If False and `other` is an IamDataFrame, raise an error if
            any meta columns present in `self` and `other` are not identical.
        inplace : bool, optional
            If True, do operation inplace and return None
        verify_integrity : bool, optional
            If True, verify integrity of index
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
            If time domain or other timeseries data index dimension don't match.
        """
        if other is None:
            return None if inplace else self.copy()

        if not isinstance(other, IamDataFrame):
            other = IamDataFrame(other, **kwargs)
            ignore_meta_conflict = True

        if self.extra_cols != other.extra_cols:
            raise ValueError("Incompatible timeseries data index dimensions")

        if other.empty:
            return None if inplace else self.copy()

        ret = self.copy() if not inplace else self

        if ret.time_col != other.time_col:
            if ret.time_col == "year":
                ret.swap_year_for_time(inplace=True)
            else:
                other = other.swap_year_for_time(inplace=False)

        # merge `meta` tables
        ret.meta = merge_meta(ret.meta, other.meta, ignore_meta_conflict)

        # append other.data (verify integrity for no duplicates)
        _data = pd.concat([ret._data, other._data])
        if verify_integrity:
            verify_index_integrity(_data)

        # merge extra columns in `data`
        ret.extra_cols += [i for i in other.extra_cols if i not in ret.extra_cols]
        ret._data = _data.sort_index()
        ret._set_attributes()

        if not inplace:
            return ret

    def pivot_table(
        self,
        index,
        columns,
        values="value",
        aggfunc="count",
        fill_value=None,
        style=None,
    ):
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

        if values != "value":
            raise ValueError("This method only supports `values='value'`!")

        df = self._data

        # allow 'aggfunc' to be passed as string for easier user interface
        if isstr(aggfunc):
            if aggfunc == "count":
                df = self._data.groupby(index + columns).count()
                fill_value = 0
            elif aggfunc == "mean":
                df = self._data.groupby(index + columns).mean().round(2)
                fill_value = 0 if style == "heatmap" else ""
            elif aggfunc == "sum":
                df = self._data.groupby(index + columns).sum()
                fill_value = 0 if style == "heatmap" else ""

        df = df.unstack(level=columns, fill_value=fill_value)
        return df

    def interpolate(self, time, inplace=False, **kwargs):
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
        ret = self.copy() if not inplace else self
        interp_kwargs = dict(method="slinear", axis=1)
        interp_kwargs.update(kwargs)
        time = to_list(time)
        # TODO - have to explicitly cast to numpy datetime to sort later,
        # could enforce as we do for year below
        if self.time_col == "time":
            time = list(map(np.datetime64, time))
        elif not all(is_integer(x) for x in time):
            raise ValueError(f"The `time` argument {time} contains non-integers")

        old_cols = list(ret[ret.time_col].unique())
        columns = np.unique(np.concatenate([old_cols, time]))

        # calculate a separate dataframe with full interpolation
        df = ret.timeseries()
        newdf = df.reindex(columns=columns).interpolate(**interp_kwargs)

        # replace only columns asked for
        for col in time:
            df[col] = newdf[col]

        # replace underlying data object
        # TODO naming time_col could be done in timeseries()
        df.columns.name = ret.time_col
        df = df.stack()  # long-data to pd.Series
        df.name = "value"
        ret._data = df.sort_index()
        ret._set_attributes()

        if not inplace:
            return ret

    def swap_time_for_year(self, subannual=False, inplace=False):
        """Convert the `time` dimension to `year` (as integer).

        Parameters
        ----------
        subannual : bool, str or func, optional
            Merge non-year components of the "time" domain as new column "subannual".
            Apply :meth:`strftime() <datetime.date.strftime>` on the values of the
            "time" domain using `subannual` (if a string) or "%m-%d %H:%M%z" (if True).
            If it is a function, apply the function on the values of the "time" domain.
        inplace : bool, optional
            If True, do operation inplace and return None.

        Returns
        -------
        :class:`IamDataFrame` or **None**
            Object with altered time domain or None if `inplace=True`.

        Raises
        ------
        ValueError
            "time" is not a column of `self.data`

        See Also
        --------
        swap_year_for_time

        """
        return swap_time_for_year(self, subannual=subannual, inplace=inplace)

    def swap_year_for_time(self, inplace=False):
        """Convert the `year` and `subannual` dimensions to `time` (as datetime).

        The method applies :meth:`dateutil.parser.parse` on the combined columns
        `year` and `subannual`:

        .. code-block:: python

            dateutil.parser.parse([f"{y}-{s}" for y, s in zip(year, subannual)])

        Parameters
        ----------
        inplace : bool, optional
            If True, do operation inplace and return None.

        Returns
        -------
        :class:`IamDataFrame` or **None**
            Object with altered time domain or None if `inplace=True`.

        Raises
        ------
        ValueError
            "year" or "subannual" are not a column of `self.data`

        See Also
        --------
        swap_time_for_year

        """
        return swap_year_for_time(self, inplace=inplace)

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
                self.data.set_index(META_IDX).join(self.meta[meta_cols]).reset_index()
            )
        else:
            return self.data.copy()

    def timeseries(self, iamc_index=False):
        """Returns `data` as :class:`pandas.DataFrame` in wide format

        Parameters
        ----------
        iamc_index : bool, optional
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
            raise ValueError("This IamDataFrame is empty!")

        s = self._data
        if iamc_index:
            if self.time_col == "time":
                raise ValueError(
                    "Cannot use `iamc_index=True` with 'datetime' time-domain!"
                )
            s = s.droplevel(self.extra_cols)

        return s.unstack(level=self.time_col).rename_axis(None, axis=1)

    def reset_exclude(self):
        """Reset exclusion assignment for all scenarios to `exclude: False`"""
        self.meta["exclude"] = False

    def set_meta(self, meta, name=None, index=None):
        """Add meta indicators as pandas.Series, list or value (int/float/str)

        Parameters
        ----------
        meta : pandas.DataFrame, pandas.Series, list, int, float or str
            column to be added to 'meta'
            (by `['model', 'scenario']` index if possible)
        name : str, optional
            meta column name (defaults to meta `pandas.Series.name`);
            either `meta.name` or the name kwarg must be defined
        index : IamDataFrame, pandas.DataFrame or pandas.MultiIndex, optional
            index to be used for setting meta column (`['model', 'scenario']`)
        """
        if isinstance(meta, pd.DataFrame):
            if meta.index.names != self.meta.index.names:
                # catch Model, Scenario instead of model, scenario
                meta = meta.rename(
                    columns={i.capitalize(): i for i in META_IDX}
                ).set_index(self.meta.index.names)

            meta = meta.loc[self.meta.index.intersection(meta.index)]
            meta.index = meta.index.remove_unused_levels()
            self.meta = merge_meta(meta, self.meta, ignore_conflict=True)
            return

        # check that name is valid and doesn't conflict with data columns
        if (name or (hasattr(meta, "name") and meta.name)) in [None, False]:
            raise ValueError("Must pass a name or use a named pd.Series")
        name = name or meta.name
        if name in self.dimensions:
            raise ValueError(f"Column {name} already exists in `data`!")
        if name in ILLEGAL_COLS:
            raise ValueError(f"Name {name} is illegal for meta indicators!")

        # check if meta has a valid index and use it for further workflow
        if (
            hasattr(meta, "index")
            and hasattr(meta.index, "names")
            and set(META_IDX).issubset(meta.index.names)
        ):
            index = meta.index

        # if no valid index is provided, add meta as new column `name` and exit
        if index is None:
            self.meta[name] = list(meta) if islistable(meta) else meta
            return

        # use meta.index if index arg is an IamDataFrame
        if isinstance(index, IamDataFrame):
            index = index.meta.index
        # turn dataframe to index if index arg is a DataFrame
        if isinstance(index, pd.DataFrame):
            index = index.set_index(META_IDX).index
        if not isinstance(index, pd.MultiIndex):
            raise ValueError("Index cannot be coerced to pd.MultiIndex")

        # raise error if index is not unique
        if index.duplicated().any():
            raise ValueError("Non-unique ['model', 'scenario'] index!")

        # create pd.Series from meta, index and name if provided
        meta = pd.Series(data=meta, index=index, name=name)

        # reduce index dimensions to model-scenario only
        meta = meta.reset_index().reindex(columns=META_IDX + [name]).set_index(META_IDX)

        # check if trying to add model-scenario index not existing in self
        diff = meta.index.difference(self.meta.index)
        if not diff.empty:
            raise ValueError(f"Adding meta for non-existing scenarios:\n{diff}")

        self._new_meta_column(name)
        self.meta[name] = meta[name].combine_first(self.meta[name])

    def set_meta_from_data(self, name, method=None, column="value", **kwargs):
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

    def categorize(
        self, name, value, criteria, color=None, marker=None, linestyle=None
    ):
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
        color : str, optional
            assign a color to this category for plotting
        marker : str, optional
            assign a marker to this category for plotting
        linestyle : str, optional
            assign a linestyle to this category for plotting
        """
        # add plotting run control
        for kind, arg in [
            ("color", color),
            ("marker", marker),
            ("linestyle", linestyle),
        ]:
            if arg:
                run_control().update({kind: {name: {value: arg}}})
        # find all data that matches categorization
        rows = _apply_criteria(self._data, criteria, in_range=True, return_test="all")
        idx = _make_index(rows, cols=self.index.names)

        if len(idx) == 0:
            logger.info("No scenarios satisfy the criteria")
            return

        # update meta dataframe
        self._new_meta_column(name)
        self.meta.loc[idx, name] = value
        msg = "{} scenario{} categorized as `{}: {}`"
        logger.info(msg.format(len(idx), "" if len(idx) == 1 else "s", name, value))

    def _new_meta_column(self, name):
        """Add a column to meta if it doesn't exist, set value to nan"""
        if name is None:
            raise ValueError(f"Cannot add a meta column {name}")
        if name not in self.meta:
            self.meta[name] = np.nan

    def require_data(
        self, region=None, variable=None, unit=None, year=None, exclude_on_fail=False
    ):
        """Check whether scenarios have values for all combinations of given elements

        Parameters
        ----------
        region : str or list of str, optional
            Required region(s).
        variable : str or list of str, optional
            Required variable(s).
        unit : str or list of str, optional
            Required unit(s).
        year : int or list of int, optional
            Required year(s).
        exclude_on_fail : bool, optional
            Set *meta* indicator for scenarios failing validation as `exclude: True`.

        Returns
        -------
        pd.DataFrame
            A dataframe of the *index* of scenarios not satisfying the cr
        """

        # TODO: option to require values in certain ranges, see `_apply_criteria()`

        # create mapping of required dimensions
        required = {}
        n = 1  # expected number of rows per scenario
        for dim, value in [
            ("region", region),
            ("variable", variable),
            ("unit", unit),
            ("year", year),
        ]:
            if value is not None:
                required[dim] = value
                n *= len(to_list(value))

        # fast exit if no arguments values are given
        if not required:
            return

        # downselect to relevant rows
        keep = self._apply_filters(**required)
        rows = self._data.index[keep]

        # identify scenarios that have none of the required values
        index_none = self.index.difference(
            rows.droplevel(level=self.coordinates).drop_duplicates()
        ).to_frame(index=False)

        # identify scenarios that have some but not all required values
        columns = [i for i in self.coordinates if i not in required]
        rows = rows.droplevel(level=columns).drop_duplicates()
        data = (
            pd.DataFrame(index=rows)
            .reset_index(level=list(required))
            .groupby(self.index.names)
        )

        index_incomplete = pd.DataFrame(
            [idx for idx, df in data if len(df) != n], columns=self.index.names
        )

        # merge all scenarios where not all required data is present
        index_missing_required = pd.concat([index_none, index_incomplete])
        if not index_missing_required.empty:
            if exclude_on_fail:
                self._exclude_on_fail(index_missing_required)
            return index_missing_required

    def require_variable(self, variable, unit=None, year=None, exclude_on_fail=False):
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
        # TODO: deprecated, remove for release >= 2.0
        deprecation_warning("Use `df.require_data()` instead.")

        criteria = {"variable": variable}
        if unit:
            criteria.update({"unit": unit})
        if year:
            criteria.update({"year": year})

        keep = self._apply_filters(**criteria)
        idx = self.meta.index.difference(_meta_idx(self.data[keep]))

        n = len(idx)
        if n == 0:
            logger.info(
                "All scenarios have the required variable `{}`".format(variable)
            )
            return

        msg = (
            "{} scenario does not include required variable `{}`"
            if n == 1
            else "{} scenarios do not include required variable `{}`"
        )

        if exclude_on_fail:
            self.meta.loc[idx, "exclude"] = True
            msg += ", marked as `exclude: True` in `meta`"

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
            msg = "{} of {} data points do not satisfy the criteria"
            logger.info(msg.format(len(df), len(self.data)))

            if exclude_on_fail and len(df) > 0:
                self._exclude_on_fail(df)
            return df.reset_index()

    def rename(
        self, mapping=None, inplace=False, append=False, check_duplicates=True, **kwargs
    ):
        """Rename any index dimension or data coordinate.

        When renaming models or scenarios, the uniqueness of the index must be
        maintained, and the function will raise an error otherwise.

        Renaming is only applied to any data row that matches for all
        columns given in `mapping`. Renaming can only be applied to the `model`
        and `scenario` columns, or to other data coordinates simultaneously.

        Parameters
        ----------
        mapping : dict or kwargs
            mapping of column name to rename-dictionary of that column

            .. code-block:: python

               dict(<column_name>: {<current_name_1>: <target_name_1>,
                                    <current_name_2>: <target_name_2>})

            or kwargs as `column_name={<current_name_1>: <target_name_1>, ...}`
        inplace : bool, optional
            Do operation inplace and return None.
        append : bool, optional
            Whether to append aggregated timeseries data to this instance
            (if `inplace=True`) or to a returned new instance (if `inplace=False`).
        check_duplicates : bool, optional
            Check whether conflicts exist after renaming of timeseries data coordinates.
            If True, raise a ValueError; if False, rename and merge
            with :meth:`groupby().sum() <pandas.core.groupby.GroupBy.sum>`.

        Returns
        -------
        :class:`IamDataFrame` or **None**
            Aggregated timeseries data as new object or None if `inplace=True`.
        """
        # combine `mapping` arg and mapping kwargs, ensure no rename conflicts
        mapping = mapping or {}
        duplicate = set(mapping).intersection(kwargs)
        if duplicate:
            raise ValueError(f"Conflicting rename args for columns {duplicate}")
        mapping.update(kwargs)

        # return without any changes if self is empty
        if self.empty:
            return self if inplace else self.copy()

        # determine columns that are not in the meta index
        meta_idx = self.meta.index.names
        data_cols = set(self.dimensions) - set(meta_idx)

        # changing index and data columns can cause model-scenario mismatch
        if any(i in mapping for i in meta_idx) and any(i in mapping for i in data_cols):
            msg = "Renaming index and data columns simultaneously not supported!"
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
        idx = ret.meta.index.isin(_make_index(ret._data[rows], cols=meta_idx))

        # apply renaming changes (for `data` only on the index)
        _data_index = ret._data.index

        for col, _mapping in mapping.items():
            if col in meta_idx:
                _index = pd.DataFrame(index=ret.meta.index).reset_index()
                _index.loc[idx, col] = _index.loc[idx, col].replace(_mapping)
                if _index.duplicated().any():
                    raise ValueError(f"Renaming to non-unique {col} index!")
                ret.meta.index = _index.set_index(meta_idx).index
            elif col not in data_cols:
                raise ValueError(f"Renaming by {col} not supported!")
            _data_index = replace_index_values(_data_index, col, _mapping, rows)

        # check if duplicates exist in the new timeseries data index
        duplicate_rows = _data_index.duplicated()
        has_duplicates = any(duplicate_rows)

        if has_duplicates and check_duplicates:
            raise_data_error(
                "Conflicting data rows after renaming "
                "(use `aggregate()` or `check_duplicates=False` instead)",
                _data_index[duplicate_rows].to_frame(index=False),
            )

        ret._data.index = _data_index
        ret._set_attributes()

        # merge using `groupby().sum()` only if duplicates exist
        if has_duplicates:
            ret._data = ret._data.reset_index().groupby(ret.dimensions).sum().value

        if not inplace:
            return ret

    def convert_unit(
        self, current, to, factor=None, registry=None, context=None, inplace=False
    ):
        """Convert all timeseries data having *current* units to new units.

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
        e.g. 'AR5GWP100' (optionally prefixed by '\gwp_', e.g. 'gwp_AR5GWP100').

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
        # check that (only) either factor or registry/context is provided
        if factor and any([registry, context]):
            raise ValueError("Use either `factor` or `registry`!")

        return convert_unit(self, current, to, factor, registry, context, inplace)

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
            raise ValueError("Only time(year)-based normalization supported")
        ret = self.copy() if not inplace else self
        df = ret.data
        # change all below if supporting more in the future
        cols = self.time_col
        value = kwargs[self.time_col]
        x = df.set_index(IAMC_IDX)
        x["value"] /= x[x[cols] == value]["value"]

        x = x.reset_index()
        ret._data = x.set_index(self.dimensions).value

        if not inplace:
            return ret

    def offset(self, padding=0, fill_value=None, inplace=False, **kwargs):
        """Compute new data which is offset from a specific data point

        For example, offsetting from `year=2005` will provide data
        *relative* to `year=2005` such that the value in 2005 is 0 and
        all other values `value[year] - value[2005]`.

        Conceptually this operation performs as:
        ```
        df - df.filter(**kwargs) + padding
        ```

        Note: Currently only supports normalizing to a specific time.

        Parameters
        ----------
        padding : float, optional
            an additional offset padding
        fill_value : float or None, optional
            Applied on subtraction. Fills exisiting missing (NaN) values. See
            https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.subtract.html
        inplace : bool, optional
            if :obj:`True`, do operation inplace and return None
        kwargs
            the column and value on which to offset (e.g., `year=2005`)
        """
        if len(kwargs) > 1 or self.time_col not in kwargs:
            raise ValueError("Only time(year)-based normalization supported")
        ret = self.copy() if not inplace else self
        data = ret._data
        value = kwargs[self.time_col]
        base_value = data.loc[data.index.isin([value], level=self.time_col)].droplevel(
            self.time_col
        )
        ret._data = data.subtract(base_value, fill_value=fill_value) + padding

        if not inplace:
            return ret

    def aggregate(
        self,
        variable,
        components=None,
        method="sum",
        recursive=False,
        append=False,
    ):
        """Aggregate timeseries data by components or subcategories within each region.

        Parameters
        ----------
        variable : str or list of str
            Variable(s) for which the aggregate will be computed.
        components : list of str, optional
            Components to be aggregate, defaults to all subcategories of `variable`.
        method : func or str, optional
            Aggregation method, e.g. :any:`numpy.mean`, :any:`numpy.sum`, 'min', 'max'.
        recursive : bool or str, optional
            Iterate recursively (bottom-up) over all subcategories of `variable`.
            If there are existing intermediate variables, it validates the aggregated
            value. If recursive='skip-validate', it skips the validation.
        append : bool, optional
            Whether to append aggregated timeseries data to this instance.

        Returns
        -------
        :class:`IamDataFrame` or **None**
            Aggregated timeseries data or None if `append=True`.

        See Also
        --------
        add : Add timeseries data items along an `axis`.
        aggregate_region : Aggregate timeseries data along the `region` dimension.

        Notes
        -----
        The aggregation function interprets any missing values (:any:`numpy.nan`)
        for individual components as 0.

        """

        if recursive:
            if components is not None:
                raise ValueError("Recursive aggregation cannot take `components`!")
            if method != "sum":
                raise ValueError(
                    "Recursive aggregation only supported with `method='sum'`!"
                )

            _df = IamDataFrame(
                _aggregate_recursive(self, variable, recursive), meta=self.meta
            )
        else:
            _df = _aggregate(self, variable, components=components, method=method)

        # append to `self` or return as `IamDataFrame`
        return self._finalize(_df, append=append)

    def check_aggregate(
        self,
        variable,
        components=None,
        method="sum",
        exclude_on_fail=False,
        multiplier=1,
        **kwargs,
    ):
        """Check whether timeseries data matches the aggregation by its components.

        Parameters
        ----------
        variable : str or list of str
            Variable(s) checked for matching aggregation of sub-categories.
        components : list of str, default None
            List of variables to aggregate, defaults to sub-categories of `variable`.
        method : func or str, optional
            Method to use for aggregation,
            e.g. :any:`numpy.mean`, :any:`numpy.sum`, 'min', 'max'.
        exclude_on_fail : bool, optional
            Flag scenarios failing validation as `exclude: True`.
        multiplier : number, optional
            Multiplicative factor when comparing variable and sum of components.
        kwargs : Tolerance arguments for comparison of values
            Passed to :func:`numpy.isclose`.

        Returns
        -------
        :class:`pandas.DataFrame` or None
            Data where variables and aggregate does not match.

        """
        # compute aggregate from components, return None if no components
        df_components = _aggregate(self, variable, components, method)
        if df_components is None:
            return

        # filter and groupby data, use `pd.Series.align` for matching index
        rows = self._apply_filters(variable=variable)
        df_var, df_components = _group_and_agg(self._data[rows], [], method).align(
            df_components
        )

        # use `np.isclose` for checking match
        rows = ~np.isclose(df_var, multiplier * df_components, **kwargs)

        # if aggregate and components don't match, return inconsistent data
        if sum(rows):
            msg = "`{}` - {} of {} rows are not aggregates of components"
            logger.info(msg.format(variable, sum(rows), len(df_var)))

            if exclude_on_fail:
                self._exclude_on_fail(_meta_idx(df_var[rows].reset_index()))

            return pd.concat(
                [df_var[rows], df_components[rows]],
                axis=1,
                keys=(["variable", "components"]),
            )

    def aggregate_region(
        self,
        variable,
        region="World",
        subregions=None,
        components=False,
        method="sum",
        weight=None,
        append=False,
        drop_negative_weights=True,
    ):
        """Aggregate timeseries data by subregions.

        This function allows to add variable sub-categories that are only
        defined at the `region` level by setting `components=True`

        Parameters
        ----------
        variable : str or list of str
            Variable(s) to be aggregated.
        region : str, optional
            Region to which data will be aggregated
        subregions : list of str, optional
            List of subregions, defaults to all regions other than `region`.
        components : bool or list of str, optional
            Variables at the `region` level to be included in the aggregation
            (ignored if False); if `True`, use all sub-categories of `variable`
            included in `region` but not in any of the `subregions`;
            or explicit list of variables.
        method : func or str, optional
            Method to use for aggregation,
            e.g. :any:`numpy.mean`, :any:`numpy.sum`, 'min', 'max'.
        weight : str, optional
            Variable to use as weight for the aggregation
            (currently only supported with `method='sum'`).
        append : bool, optional
            Append the aggregate timeseries to `self` and return None,
            else return aggregate timeseries as new :class:`IamDataFrame`.
        drop_negative_weights : bool, optional
            Removes any aggregated values that are computed using negative weights.

        Returns
        -------
        :class:`IamDataFrame` or **None**
            Aggregated timeseries data or None if `append=True`.

        See Also
        --------
        add : Add timeseries data items `a` and `b` along an `axis`
        aggregate : Aggregate timeseries data along the `variable` hierarchy.

        """
        _df = _aggregate_region(
            self,
            variable,
            region=region,
            subregions=subregions,
            components=components,
            method=method,
            weight=weight,
            drop_negative_weights=drop_negative_weights,
        )

        # append to `self` or return as `IamDataFrame`
        return self._finalize(_df, append=append, region=region)

    def check_aggregate_region(
        self,
        variable,
        region="World",
        subregions=None,
        components=False,
        method="sum",
        weight=None,
        exclude_on_fail=False,
        drop_negative_weights=True,
        **kwargs,
    ):
        """Check whether timeseries data matches the aggregation across subregions.

        Parameters
        ----------
        variable : str or list of str
            Variable(s) to be checked for matching aggregation of subregions.
        region : str, optional
            Region to be checked for matching aggregation of subregions.
        subregions : list of str, optional
            List of subregions, defaults to all regions other than `region`.
        components : bool or list of str, optional
            Variables at the `region` level to be included in the aggregation
            (ignored if False); if `True`, use all sub-categories of `variable`
            included in `region` but not in any of the `subregions`;
            or explicit list of variables.
        method : func or str, optional
            Method to use for aggregation,
            e.g. :any:`numpy.mean`, :any:`numpy.sum`, 'min', 'max'.
        weight : str, optional
            Variable to use as weight for the aggregation
            (currently only supported with `method='sum'`).
        exclude_on_fail : boolean, optional
            Flag scenarios failing validation as `exclude: True`.
        drop_negative_weights : bool, optional
            Removes any aggregated values that are computed using negative weights
        kwargs : Tolerance arguments for comparison of values
            Passed to :func:`numpy.isclose`.

        Returns
        -------
        :class:`pandas.DataFrame` or None
            Data where variables and region-aggregate does not match.

        """
        # compute aggregate from subregions, return None if no subregions
        df_subregions = _aggregate_region(
            self,
            variable,
            region,
            subregions,
            components,
            method,
            weight,
            drop_negative_weights,
        )

        if df_subregions is None:
            return

        # filter and groupby data, use `pd.Series.align` for matching index
        rows = self._apply_filters(region=region, variable=variable)
        if not rows.any():
            logger.info(f"Variable '{variable}' does not exist in region '{region}'!")
            return

        df_region, df_subregions = _group_and_agg(self._data[rows], "region").align(
            df_subregions
        )

        # use `np.isclose` for checking match
        rows = ~np.isclose(df_region, df_subregions, **kwargs)

        # if region and subregions don't match, return inconsistent data
        if sum(rows):
            msg = "`{}` - {} of {} rows are not aggregates of subregions"
            logger.info(msg.format(variable, sum(rows), len(df_region)))

            if exclude_on_fail:
                self._exclude_on_fail(_meta_idx(df_region[rows].reset_index()))

            _df = pd.concat(
                [
                    pd.concat(
                        [df_region[rows], df_subregions[rows]],
                        axis=1,
                        keys=(["region", "subregions"]),
                    )
                ],
                keys=[region],
                names=["region"],
            )
            _df.index = _df.index.reorder_levels(self.dimensions)
            return _df

    def aggregate_time(
        self,
        variable,
        column="subannual",
        value="year",
        components=None,
        method="sum",
        append=False,
    ):
        """Aggregate timeseries data by subannual time resolution

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
        _df = _aggregate_time(
            self,
            variable,
            column=column,
            value=value,
            components=components,
            method=method,
        )

        # append to `self` or return as `IamDataFrame`
        return self._finalize(_df, append=append)

    def downscale_region(
        self,
        variable,
        region="World",
        subregions=None,
        proxy=None,
        weight=None,
        append=False,
    ):
        """Downscale timeseries data to a number of subregions

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
            raise ValueError("Using both 'proxy' and 'weight' arguments is not valid!")
        elif proxy is not None:
            # get default subregions if not specified and select data from self
            subregions = subregions or self._all_other_regions(region)
            rows = self._apply_filters(variable=proxy, region=subregions)
            cols = self._get_cols(["region", self.time_col])
            _proxy = self.data[rows].set_index(cols).value
        elif weight is not None:
            # downselect weight to subregions or remove `region` from index
            if subregions is not None:
                rows = weight.index.isin(subregions, level="region")
            else:
                rows = ~weight.index.isin([region], level="region")
            _proxy = weight[rows].stack()
        else:
            raise ValueError("Either a 'proxy' or 'weight' argument is required!")

        _value = (
            self.data[self._apply_filters(variable=variable, region=region)]
            .set_index(self._get_cols(["variable", "unit", self.time_col]))
            .value
        )

        # compute downscaled data
        _total = _proxy.groupby(self.time_col).sum()
        _data = _value * _proxy / _total

        # append to `self` or return as `IamDataFrame`
        return self._finalize(_data, append=append)

    def _all_other_regions(self, region, variable=None):
        """Return list of regions other than `region` containing `variable`"""
        rows = self._apply_filters(variable=variable)
        return self._data[rows].index.get_level_values("region").difference([region])

    def _variable_components(self, variable, level=0):
        """Get all components (sub-categories) of a variable for a given level

        If `level=0`, for `variable='foo'`, return `['foo|bar']`, but don't
        include `'foo|bar|baz'`, which is a sub-sub-category. If `level=None`,
        all variables below `variable` in the hierarchy are returned."""
        var_list = pd.Series(self.variable)
        return var_list[pattern_match(var_list, "{}|*".format(variable), level=level)]

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
        for variable in self.variable:
            diff_agg = self.check_aggregate(variable, **kwargs)
            if diff_agg is not None:
                lst.append(diff_agg)

            diff_regional = self.check_aggregate_region(
                variable, components=components, **kwargs
            )
            if diff_regional is not None:
                lst.append(diff_regional)

        if len(lst):
            _df = pd.concat(lst, sort=True).sort_index()
            return _df[
                [
                    c
                    for c in ["variable", "components", "region", "subregions"]
                    if c in _df.columns
                ]
            ]

    def _exclude_on_fail(self, df):
        """Assign a selection of scenarios as `exclude: True` in meta"""
        idx = (
            df
            if isinstance(df, pd.MultiIndex)
            else _make_index(df, cols=self.index.names)
        )
        self.meta.loc[idx, "exclude"] = True
        logger.info(
            "{} non-valid scenario{} will be excluded".format(
                len(idx), "" if len(idx) == 1 else "s"
            )
        )

    def slice(self, keep=True, **kwargs):
        """Return a (filtered) slice object of the IamDataFrame timeseries data index

        Parameters
        ----------
        keep : bool, optional
            Keep all scenarios satisfying the filters (if *True*) or the inverse.
        **kwargs
            Arguments for filtering. See the "Notes".

        Returns
        -------
        :class:`IamSlice`

        Notes
        -----
        The following arguments are available for filtering:

         - 'meta' columns: filter by string value of that column
         - 'model', 'scenario', 'region', 'variable', 'unit':
           string or list of strings, where `*` can be used as a wildcard
         - 'index': list of model, scenario 2-tuples or :class:`pandas.MultiIndex`
         - 'level': the "depth" of entries in the variable column (number of '|')
           (excluding the strings given in the 'variable' argument)
         - 'year': takes an integer (int/np.int64), a list of integers or
           a range. Note that the last year of a range is not included,
           so `range(2010, 2015)` is interpreted as `[2010, ..., 2014]`
         - 'time_domain': can be "year" or "datetime"
         - arguments for filtering by `datetime.datetime` or np.datetime64
           ('month', 'hour', 'time')
         - 'regexp=True' disables pseudo-regexp syntax in `pattern_match()`

        """

        if not isinstance(keep, bool):
            raise ValueError(f"Value of `keep` must be a boolean, found: {keep}")

        _keep = self._apply_filters(**kwargs)
        _keep = _keep if keep else ~_keep

        return (
            IamSlice(_keep)
            if isinstance(_keep, pd.Series)
            else IamSlice(_keep, self._data.index)
        )

    def filter(self, keep=True, inplace=False, **kwargs):
        """Return a (copy of a) filtered (downselected) IamDataFrame

        Parameters
        ----------
        keep : bool, optional
            Keep all scenarios satisfying the filters (if *True*) or the inverse.
        inplace : bool, optional
            If *True*, do operation inplace and return *None*.
        **kwargs
            Passed to :meth:`slice`.
        """

        # downselect `data` rows and clean up index
        ret = self.copy() if not inplace else self
        ret._data = ret._data[self.slice(keep=keep, **kwargs)]
        ret._data.index = ret._data.index.remove_unused_levels()

        # swap time for year if downselected to years-only
        if ret.time_col == "time":
            time_values = get_index_levels(ret._data, "time")
            if time_values and all([pd.api.types.is_integer(y) for y in time_values]):
                ret.swap_time_for_year(inplace=True)
                msg = "Only yearly data after filtering, time-domain changed to 'year'."
                logger.info(msg)

        # downselect `meta` dataframe
        idx = _make_index(ret._data, cols=self.index.names)
        if len(idx) == 0:
            logger.warning("Filtered IamDataFrame is empty!")
        ret.meta = ret.meta.loc[idx]
        ret.meta.index = ret.meta.index.remove_unused_levels()
        ret._set_attributes()
        if not inplace:
            return ret

    def _apply_filters(self, level=None, **filters):
        """Determine rows to keep in data for given set of filters

        Parameters
        ----------
        filters : dict
            dictionary of filters of the format (`{col: values}`);
            uses a pseudo-regexp syntax by default,
            but accepts `regexp: True` in the dictionary to use regexp directly
        """
        regexp = filters.pop("regexp", False)
        keep = np.ones(len(self), dtype=bool)

        # filter by columns and list of values
        for col, values in filters.items():
            # treat `_apply_filters(col=None)` as no filter applied
            if values is None:
                continue

            if col in self.meta.columns:
                matches = pattern_match(
                    self.meta[col], values, regexp=regexp, has_nan=True
                )
                cat_idx = self.meta[matches].index
                keep_col = _make_index(
                    self._data, cols=self.index.names, unique=False
                ).isin(cat_idx)

            elif col == "index":
                if not isinstance(values, pd.MultiIndex):
                    values = pd.MultiIndex.from_tuples(values, names=self.index.names)
                elif all(n is None for n in values.names):
                    values = values.rename(names=self.index.names)
                elif not set(values.names).issubset(self.index.names):
                    index_levels = ", ".join(map(str, self.index.names))
                    values_levels = ", ".join(map(str, values.names))
                    raise ValueError(
                        f"Filtering by `index` with a MultiIndex object needs to have "
                        f"the IamDataFrame index levels {index_levels}, "
                        f"but has {values_levels}"
                    )
                index = self._data.index
                keep_col = index.droplevel(index.names.difference(values.names)).isin(
                    values
                )

            elif col == "time_domain":
                # fast-pass if `self` already has selected time-domain
                if self.time_domain == values:
                    keep_col = np.ones(len(self), dtype=bool)
                else:
                    levels, codes = get_index_levels_codes(self._data, self.time_col)
                    keep_col = filter_by_time_domain(values, levels, codes)

            elif col == "year":
                levels, codes = get_index_levels_codes(self._data, self.time_col)
                keep_col = filter_by_year(self.time_col, values, levels, codes)

            elif col in ["month", "hour", "day"]:
                if self.time_col != "time":
                    logger.error(f"Filter by `{col}` not supported with yearly data.")
                    return np.zeros(len(self), dtype=bool)

                keep_col = filter_by_dt_arg(col, values, self.get_data_column("time"))

            elif col == "time":
                if self.time_col != "time":
                    logger.error(f"Filter by `{col}` not supported with yearly data.")
                    return np.zeros(len(self), dtype=bool)

                keep_col = datetime_match(self.get_data_column("time"), values)

            elif col in self.dimensions:
                levels, codes = get_index_levels_codes(self._data, col)

                matches = pattern_match(
                    levels,
                    values,
                    regexp=regexp,
                    level=level if col == "variable" else None,
                    has_nan=True,
                    return_codes=True,
                )
                keep_col = get_keep_col(codes, matches)

            else:
                raise ValueError(f"Filter by `{col}` not supported!")

            keep = np.logical_and(keep, keep_col)

        if level is not None and "variable" not in filters:
            col = "variable"
            lvl_index, lvl_codes = get_index_levels_codes(self._data, col)
            matches = find_depth(lvl_index, level=level)
            keep_col = get_keep_col(lvl_codes, matches)

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

    def add(
        self, a, b, name, axis="variable", fillna=None, ignore_units=False, append=False
    ):
        """Add timeseries data items `a` and `b` along an `axis`

        This function computes `a + b`. If `a` or `b` are lists, the method applies
        :meth:`pandas.groupby().sum() <pandas.core.groupby.GroupBy.sum>` on each group.
        If either `a` or `b` are not defined for a row and `fillna` is not specified,
        no value is computed for that row.

        Parameters
        ----------
        a, b : str, list of str or a number
            Items to be used for the addition.
        name : str
            Name of the computed timeseries data on the `axis`.
        axis : str, optional
            Axis along which to compute.
        fillna : dict or scalar, optional
            Value to fill holes when rows are not defined for either `a` or `b`.
            Can be a scalar or a dictionary of the form :code:`{arg: default}`.
        ignore_units : bool or str, optional
            Perform operation on values without considering units. Set units of returned
            data to `unknown` (if True) or the value of `ignore_units` (if str).
        append : bool, optional
            Whether to append aggregated timeseries data to this instance.

        Returns
        -------
        :class:`IamDataFrame` or **None**
            Computed timeseries data or None if `append=True`.

        See Also
        --------
        subtract, multiply, divide
        apply : Apply a custom function on the timeseries data along any axis.
        aggregate : Aggregate timeseries data along the `variable` hierarchy.
        aggregate_region : Aggregate timeseries data along the `region` dimension.

        Notes
        -----
        This function uses the :mod:`pint` package and the :mod:`iam-units` registry
        (`read the docs <https://github.com/IAMconsortium/units>`_) to handle units.
        :mod:`pyam` will keep notation consistent with the input format (if possible)
        and otherwise uses abbreviated units :code:`'{:~}'.format(u)` (see
        `here <https://pint.readthedocs.io/en/stable/tutorial.html#string-formatting>`_
        for more information).

        As a result, the notation of returned units may differ from the input format.
        For example, the unit :code:`EJ/yr` may be reformatted to :code:`EJ / a`.
        """
        kwds = dict(axis=axis, fillna=fillna, ignore_units=ignore_units)
        _value = _op_data(self, name, "add", **kwds, a=a, b=b)

        # append to `self` or return as `IamDataFrame`
        return self._finalize(_value, append=append)

    def subtract(
        self, a, b, name, axis="variable", fillna=None, ignore_units=False, append=False
    ):
        """Compute the difference of timeseries data items `a` and `b` along an `axis`

        This function computes `a - b`. If `a` or `b` are lists, the method applies
        :meth:`pandas.groupby().sum() <pandas.core.groupby.GroupBy.sum>` on each group.
        If either `a` or `b` are not defined for a row and `fillna` is not specified,
        no value is computed for that row.

        Parameters
        ----------
        a, b : str, list of str or a number
            Items to be used for the subtraction.
        name : str
            Name of the computed timeseries data on the `axis`.
        axis : str, optional
            Axis along which to compute.
        fillna : dict or scalar, optional
            Value to fill holes when rows are not defined for either `a` or `b`.
            Can be a scalar or a dictionary of the form :code:`{arg: default}`.
        ignore_units : bool or str, optional
            Perform operation on values without considering units. Set units of returned
            data to `unknown` (if True) or the value of `ignore_units` (if str).
        append : bool, optional
            Whether to append aggregated timeseries data to this instance.

        Returns
        -------
        :class:`IamDataFrame` or **None**
            Computed timeseries data or None if `append=True`.

        See Also
        --------
        add, multiply, divide
        diff : Compute the difference of timeseries data along the time dimension.
        apply : Apply a custom function on the timeseries data along any axis.

        Notes
        -----
        This function uses the :mod:`pint` package and the :mod:`iam-units` registry
        (`read the docs <https://github.com/IAMconsortium/units>`_) to handle units.
        :mod:`pyam` will keep notation consistent with the input format (if possible)
        and otherwise uses abbreviated units :code:`'{:~}'.format(u)` (see
        `here <https://pint.readthedocs.io/en/stable/tutorial.html#string-formatting>`_
        for more information).

        As a result, the notation of returned units may differ from the input format.
        For example, the unit :code:`EJ/yr` may be reformatted to :code:`EJ / a`.
        """
        kwds = dict(axis=axis, fillna=fillna, ignore_units=ignore_units)
        _value = _op_data(self, name, "subtract", **kwds, a=a, b=b)

        # append to `self` or return as `IamDataFrame`
        return self._finalize(_value, append=append)

    def multiply(
        self, a, b, name, axis="variable", fillna=None, ignore_units=False, append=False
    ):
        """Multiply timeseries data items `a` and `b` along an `axis`

        This function computes `a * b`. If `a` or `b` are lists, the method applies
        :meth:`pandas.groupby().sum() <pandas.core.groupby.GroupBy.sum>` on each group.
        If either `a` or `b` are not defined for a row and `fillna` is not specified,
        no value is computed for that row.

        Parameters
        ----------
        a, b : str, list of str or a number
            Items to be used for the division.
        name : str
            Name of the computed timeseries data on the `axis`.
        axis : str, optional
            Axis along which to compute.
        fillna : dict or scalar, optional
            Value to fill holes when rows are not defined for either `a` or `b`.
            Can be a scalar or a dictionary of the form :code:`{arg: default}`.
        ignore_units : bool or str, optional
            Perform operation on values without considering units. Set units of returned
            data to `unknown` (if True) or the value of `ignore_units` (if str).
        append : bool, optional
            Whether to append aggregated timeseries data to this instance.

        Returns
        -------
        :class:`IamDataFrame` or **None**
            Computed timeseries data or None if `append=True`.

        See Also
        --------
        add, subtract, divide
        apply : Apply a custom function on the timeseries data along any axis.

        Notes
        -----
        This function uses the :mod:`pint` package and the :mod:`iam-units` registry
        (`read the docs <https://github.com/IAMconsortium/units>`_) to handle units.
        :mod:`pyam` will keep notation consistent with the input format (if possible)
        and otherwise uses abbreviated units :code:`'{:~}'.format(u)` (see
        `here <https://pint.readthedocs.io/en/stable/tutorial.html#string-formatting>`_
        for more information).

        As a result, the notation of returned units may differ from the input format.
        For example, the unit :code:`EJ/yr` may be reformatted to :code:`EJ / a`.
        """
        kwds = dict(axis=axis, fillna=fillna, ignore_units=ignore_units)
        _value = _op_data(self, name, "multiply", **kwds, a=a, b=b)

        # append to `self` or return as `IamDataFrame`
        return self._finalize(_value, append=append)

    def divide(
        self, a, b, name, axis="variable", fillna=None, ignore_units=False, append=False
    ):
        """Divide the timeseries data items `a` and `b` along an `axis`

        This function computes `a / b`. If `a` or `b` are lists, the method applies
        :meth:`pandas.groupby().sum() <pandas.core.groupby.GroupBy.sum>` on each group.
        If either `a` or `b` are not defined for a row and `fillna` is not specified,
        no value is computed for that row.

        Parameters
        ----------
        a, b : str, list of str or a number
            Items to be used for the division.
        name : str
            Name of the computed timeseries data on the `axis`.
        axis : str, optional
            Axis along which to compute.
        fillna : dict or scalar, optional
            Value to fill holes when rows are not defined for either `a` or `b`.
            Can be a scalar or a dictionary of the form :code:`{arg: default}`.
        ignore_units : bool or str, optional
            Perform operation on values without considering units. Set units of returned
            data to `unknown` (if True) or the value of `ignore_units` (if str).
        append : bool, optional
            Whether to append aggregated timeseries data to this instance.

        Returns
        -------
        :class:`IamDataFrame` or **None**
            Computed timeseries data or None if `append=True`.

        See Also
        --------
        add, subtract, multiply
        apply : Apply a custom function on the timeseries data along any axis.

        Notes
        -----
        This function uses the :mod:`pint` package and the :mod:`iam-units` registry
        (`read the docs <https://github.com/IAMconsortium/units>`_) to handle units.
        :mod:`pyam` will keep notation consistent with the input format (if possible)
        and otherwise uses abbreviated units :code:`'{:~}'.format(u)` (see
        `here <https://pint.readthedocs.io/en/stable/tutorial.html#string-formatting>`_
        for more information).

        As a result, the notation of returned units may differ from the input format.
        For example, the unit :code:`EJ/yr` may be reformatted to :code:`EJ / a`.
        """
        kwds = dict(axis=axis, fillna=fillna, ignore_units=ignore_units)
        _value = _op_data(self, name, "divide", **kwds, a=a, b=b)

        # append to `self` or return as `IamDataFrame`
        return self._finalize(_value, append=append)

    def apply(
        self, func, name, axis="variable", fillna=None, append=False, args=(), **kwds
    ):
        """Apply a function to components of timeseries data along an `axis`

        This function computes a function `func` using timeseries data selected
        along an `axis` downselected by keyword arguments.
        The length of components needs to match the number of required arguments
        of `func`.

        Parameters
        ----------
        func : function
            Function to apply to `components` along `axis`.
        name : str
            Name of the computed timeseries data on the `axis`.
        axis : str, optional
            Axis along which to compute.
        fillna : dict or scalar, optional
            Value to fill holes when rows are not defined for items in `args` or `kwds`.
            Can be a scalar or a dictionary of the form :code:`{kwd: default}`.
        append : bool, optional
            Whether to append aggregated timeseries data to this instance.
        args : tuple or list of str
            List of variables to pass as positional arguments to `func`.
        **kwds
            Additional keyword arguments to pass as keyword arguments to `func`. If the
            name of a variable is given, the associated timeseries is passed. Otherwise
            the value itself is passed.

        Returns
        -------
        :class:`IamDataFrame` or **None**
            Computed timeseries data or None if `append=True`.

        See Also
        --------
        add, subtract, multiply, divide, diff

        Notes
        -----
        This function uses the :mod:`pint` package and the :mod:`iam-units` registry
        (`read the docs <https://github.com/IAMconsortium/units>`_) to handle units.
        :mod:`pyam` uses abbreviated units :code:`'{:~}'.format(u)` (see
        `here <https://pint.readthedocs.io/en/stable/tutorial.html#string-formatting>`_
        for more information).

        As a result, the notation of returned units may differ from the input format.
        For example, the unit :code:`EJ/yr` may be reformatted to :code:`EJ / a`.
        """
        _value = _op_data(self, name, func, axis=axis, fillna=fillna, args=args, **kwds)

        # append to `self` or return as `IamDataFrame`
        return self._finalize(_value, append=append)

    def diff(self, mapping, periods=1, append=False):
        """Compute the difference of timeseries data along the time dimension

        This methods behaves as if applying :meth:`pandas.DataFrame.diff` on the
        timeseries data in wide format.
        By default, the diff-value in period *t* is computed as *x[t] - x[t-1]*.

        Parameters
        ----------
        mapping : dict
            Mapping of *variable* item(s) to the name(s) of the diff-ed timeseries data,
            e.g.,

            .. code-block:: python

               {"current variable": "name of diff-ed variable", ...}

        periods : int, optional
            Periods to shift for calculating difference, accepts negative values;
            passed to :meth:`pandas.DataFrame.diff`.
        append : bool, optional
            Whether to append computed timeseries data to this instance.

        Returns
        -------
        :class:`IamDataFrame` or **None**
            Computed timeseries data or None if `append=True`.

        See Also
        --------
        subtract, apply, interpolate

        Notes
        -----
        This method behaves as if applying :meth:`pandas.DataFrame.diff` by row in a
        wide data format, so the difference is computed on the previous existing value.
        This can lead to unexpected results if the data has inconsistent period lengths.

        Use the following to ensure that no missing values exist prior to computing
        the difference:

        .. code-block:: python

            df.interpolate(time=df.year)

        """
        cols = [d for d in self.dimensions if d != self.time_col]
        _value = self.filter(variable=mapping)._data.groupby(cols).diff(periods=periods)
        _value.index = replace_index_values(_value.index, "variable", mapping)

        # append to `self` or return as `IamDataFrame`
        return self._finalize(_value, append=append)

    def _to_file_format(self, iamc_index):
        """Return a dataframe suitable for writing to a file"""
        df = self.timeseries(iamc_index=iamc_index).reset_index()
        df = df.rename(columns={c: str(c).title() for c in df.columns})
        return df

    def to_csv(self, path=None, iamc_index=False, **kwargs):
        """Write :meth:`IamDataFrame.timeseries` to a comma-separated values (csv) file

        Parameters
        ----------
        path : str, path or file-like, optional
            File path as string or :class:`pathlib.Path`, or file-like object.
            If *None*, the result is returned as a csv-formatted string.
            See :meth:`pandas.DataFrame.to_csv` for details.
        iamc_index : bool, optional
            If True, use `['model', 'scenario', 'region', 'variable', 'unit']`;
            else, use all :attr:`dimensions`.
            See :meth:`IamDataFrame.timeseries` for details.
        **kwargs
            Passed to :meth:`pandas.DataFrame.to_csv`.
        """
        return self._to_file_format(iamc_index).to_csv(path, index=False, **kwargs)

    def to_excel(
        self,
        excel_writer,
        sheet_name="data",
        iamc_index=False,
        include_meta=True,
        **kwargs,
    ):
        """Write object to an Excel spreadsheet

        Parameters
        ----------
        excel_writer : path-like, file-like, or ExcelWriter object
            File path as string or :class:`pathlib.Path`,
            or existing :class:`pandas.ExcelWriter`.
        sheet_name : string
            Name of sheet which will contain :meth:`IamDataFrame.timeseries` data.
        iamc_index : bool, optional
            If True, use `['model', 'scenario', 'region', 'variable', 'unit']`;
            else, use all :attr:`dimensions`.
            See :meth:`IamDataFrame.timeseries` for details.
        include_meta : boolean or string, optional
            If True, write :any:`IamDataFrame.meta` to a sheet 'meta' (default);
            if this is a string, use it as sheet name.
        **kwargs
            Passed to :class:`pandas.ExcelWriter` (if *excel_writer* is path-like)
        """
        # open a new ExcelWriter instance (if necessary)
        close = False
        if not isinstance(excel_writer, pd.ExcelWriter):
            close = True
            excel_writer = pd.ExcelWriter(excel_writer, **kwargs)

        # write data table
        write_sheet(excel_writer, sheet_name, self._to_file_format(iamc_index))

        # write meta table unless `include_meta=False`
        if include_meta:
            meta_rename = dict([(i, i.capitalize()) for i in self.index.names])
            write_sheet(
                excel_writer,
                "meta" if include_meta is True else include_meta,
                self.meta.reset_index().rename(columns=meta_rename),
            )

        # close the file if `excel_writer` arg was a file name
        if close:
            excel_writer.close()

    def export_meta(self, excel_writer, sheet_name="meta", **kwargs):
        """Write the 'meta' indicators of this object to an Excel spreadsheet

        Parameters
        ----------
        excel_writer : str, path object or ExcelWriter object
            File path, :class:`pathlib.Path`, or existing :class:`pandas.ExcelWriter`.
        sheet_name : str
            Name of sheet which will contain 'meta'.
        **kwargs
            Passed to :class:`pandas.ExcelWriter` (if *excel_writer* is path-like)
        """
        close = False
        if not isinstance(excel_writer, pd.ExcelWriter):
            excel_writer = pd.ExcelWriter(excel_writer, **kwargs)
            close = True
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
            raise ImportError("Required package `datapackage` not found!")

        with TemporaryDirectory(dir=".") as tmp:
            # save data and meta tables to a temporary folder
            self.data.to_csv(Path(tmp) / "data.csv", index=False)
            self.meta.to_csv(Path(tmp) / "meta.csv")

            # cast tables to datapackage
            package = Package()
            package.infer("{}/*.csv".format(tmp))
            if not package.valid:
                logger.warning("The exported datapackage is not valid")
            package.save(path)

        # return the package (needs to reloaded because `tmp` was deleted)
        return Package(path)

    def load_meta(
        self, path, sheet_name="meta", ignore_conflict=False, *args, **kwargs
    ):
        """Load 'meta' indicators from file

        Parameters
        ----------
        path : str, :class:`pathlib.Path` or :class:`pandas.ExcelFile`
            A valid path or instance of an xlsx or csv file
        sheet_name : str, optional
            Name of the sheet to be parsed (if xlsx)
        ignore_conflict : bool, optional
            If `True`, values in `path` take precedence over existing `meta`.
            If `False`, raise an error in case of conflicts.
        kwargs
            Passed to :func:`pandas.read_excel` or :func:`pandas.read_csv`
        """
        # load from file
        path = path if isinstance(path, pd.ExcelFile) else Path(path)
        df = read_pandas(path, sheet_name=sheet_name, **kwargs)

        # cast model-scenario column headers to lower-case (if necessary)
        df = df.rename(columns=dict([(i.capitalize(), i) for i in META_IDX]))

        # check that required index columns exist
        missing_cols = [c for c in self.index.names if c not in df.columns]
        if missing_cols:
            raise ValueError(
                f"File {Path(path)} (sheet {sheet_name}) "
                f"missing required index columns {missing_cols}!"
            )

        # set index, filter to relevant scenarios from imported file
        n = len(df)
        df.set_index(self.index.names, inplace=True)
        df = df.loc[self.meta.index.intersection(df.index)]

        # skip import of meta indicators if np
        if not n:
            logger.info(f"No scenarios found in sheet {sheet_name}")
            return

        msg = "Reading meta indicators"
        # indicate if not all scenarios are included in the meta file
        if len(df) < len(self.meta):
            i = len(self.meta)
            msg += f" for {len(df)} out of {i} scenario{s(i)}"

        # indicate if more scenarios exist in meta file than in self
        invalid = n - len(df)
        if invalid:
            msg += f", ignoring {invalid} scenario{s(invalid)} from file"
            logger.warning(msg)
        else:
            logger.info(msg)

        # merge imported meta indicators
        self.meta = merge_meta(df, self.meta, ignore_conflict=ignore_conflict)

    def map_regions(
        self,
        map_col,
        agg=None,
        copy_col=None,
        fname=None,
        region_col=None,
        remove_duplicates=False,
        inplace=False,
    ):
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
        fname = fname or run_control()["region_mapping"]["default"]
        mapping = read_pandas(Path(fname)).rename(str.lower, axis="columns")
        map_col = map_col.lower()

        ret = self.copy() if not inplace else self
        _df = ret.data
        columns_orderd = _df.columns

        # merge data
        dfs = []
        for model in self.model:
            df = _df[_df["model"] == model]
            _col = region_col or "{}.REGION".format(model)
            _map = mapping.rename(columns={_col.lower(): "region"})
            _map = _map[["region", map_col]].dropna().drop_duplicates()
            _map = _map[_map["region"].isin(_df["region"])]
            if remove_duplicates and _map["region"].duplicated().any():
                # find duplicates
                where_dup = _map["region"].duplicated(keep=False)
                dups = _map[where_dup]
                logger.warning(
                    """
                Duplicate entries found for the following regions.
                Mapping will occur only for the most common instance.
                {}""".format(
                        dups["region"].unique()
                    )
                )
                # get non duplicates
                _map = _map[~where_dup]
                # order duplicates by the count frequency
                dups = (
                    dups.groupby(["region", map_col])
                    .size()
                    .reset_index(name="count")
                    .sort_values(by="count", ascending=False)
                    .drop("count", axis=1)
                )
                # take top occurance
                dups = dups[~dups["region"].duplicated(keep="first")]
                # combine them back
                _map = pd.concat([_map, dups])
            if copy_col is not None:
                df[copy_col] = df["region"]

            df = (
                df.merge(_map, on="region")
                .drop("region", axis=1)
                .rename(columns={map_col: "region"})
            )
            dfs.append(df)
        df = pd.concat(dfs)

        # perform aggregations
        if agg == "sum":
            df = df.groupby(self.dimensions).sum().reset_index()

        df = (
            df.reindex(columns=columns_orderd)
            .sort_values(SORT_IDX)
            .reset_index(drop=True)
        )
        ret._data = df.set_index(self.dimensions).value

        if not inplace:
            return ret


def _meta_idx(data):
    """Return the 'META_IDX' from data by index"""
    return data[META_IDX].drop_duplicates().set_index(META_IDX).index


def _check_rows(rows, check, in_range=True, return_test="any"):
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
    valid_checks = set(["up", "lo", "year"])
    if not set(check.keys()).issubset(valid_checks):
        msg = "Unknown checking type: {}"
        raise ValueError(msg.format(check.keys() - valid_checks))
    if "year" not in check:
        where_idx = set(rows.index)
    else:
        if "time" in rows.index.names:
            _years = rows.index.get_level_values("time").year
        else:
            _years = rows.index.get_level_values("year")
        where_idx = set(rows.index[_years == check["year"]])
        rows = rows.loc[list(where_idx)]

    up_op = rows.values.__le__ if in_range else rows.values.__gt__
    lo_op = rows.values.__ge__ if in_range else rows.values.__lt__

    check_idx = []
    for (bd, op) in [("up", up_op), ("lo", lo_op)]:
        if bd in check:
            check_idx.append(set(rows.index[op(check[bd])]))

    if return_test == "any":
        ret = where_idx & set.union(*check_idx)
    elif return_test == "all":
        ret = where_idx if where_idx == set.intersection(*check_idx) else set()
    else:
        raise ValueError("Unknown return test: {}".format(return_test))
    return ret


def _apply_criteria(df, criteria, **kwargs):
    """Apply criteria individually to every model/scenario instance"""
    idxs = []
    for var, check in criteria.items():
        _df = df[df.index.get_level_values("variable") == var]
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
        df.meta["exclude"] |= fdf.meta["exclude"]  # update if any excluded
        return vdf


def require_variable(
    df, variable, unit=None, year=None, exclude_on_fail=False, **kwargs
):
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
        vdf = fdf.require_variable(
            variable=variable, unit=unit, year=year, exclude_on_fail=exclude_on_fail
        )
        df.meta["exclude"] |= fdf.meta["exclude"]  # update if any excluded
        return vdf


def categorize(
    df, name, value, criteria, color=None, marker=None, linestyle=None, **kwargs
):
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
    fdf.categorize(
        name=name,
        value=value,
        criteria=criteria,
        color=color,
        marker=marker,
        linestyle=linestyle,
    )

    # update meta indicators
    if name in df.meta:
        df.meta[name].update(fdf.meta[name])
    else:
        df.meta[name] = fdf.meta[name]


def check_aggregate(
    df, variable, components=None, exclude_on_fail=False, multiplier=1, **kwargs
):
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
        vdf = fdf.check_aggregate(
            variable=variable,
            components=components,
            exclude_on_fail=exclude_on_fail,
            multiplier=multiplier,
        )
        df.meta["exclude"] |= fdf.meta["exclude"]  # update if any excluded
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
        raise ValueError("Missing required index dimensions or columns!")

    meta = pd.DataFrame(df.meta[list(set(kwargs) - set(META_IDX))].copy())

    # filter meta by columns
    keep = np.array([True] * len(meta))
    apply_filter = False
    for col, values in kwargs.items():
        if col in META_IDX and values is not None:
            _col = meta.index.get_level_values(0 if col == "model" else 1)
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
    data = data.reset_index().set_index(idx or "index")
    if idx is None:
        data.index.name = None

    return data


def compare(
    left, right, left_label="left", right_label="right", drop_close=True, **kwargs
):
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
    return _compare(left, right, left_label, right_label, drop_close=True, **kwargs)


def concat(objs, ignore_meta_conflict=False, **kwargs):
    """Concatenate a series of IamDataFrame-like objects

    Parameters
    ----------
    objs : iterable of IamDataFrames
        A list of objects castable to :class:`IamDataFrame`
    ignore_meta_conflict : bool, optional
        If False, raise an error if any meta columns present in `dfs` are not identical.
        If True, values in earlier elements of `dfs` take precedence.
    kwargs
        Passed to :class:`IamDataFrame(other, **kwargs) <IamDataFrame>`
        for any item of `dfs` which isn't already an IamDataFrame.

    Returns
    -------
    IamDataFrame

    Raises
    ------
    TypeError
        If `dfs` is not a list.
    ValueError
        If time domain or other timeseries data index dimension don't match.

    Notes
    -----
    The *meta* attributes are merged only for those objects of *objs* that are passed
    as :class:`IamDataFrame` instances.

    The :attr:`dimensions` and :attr:`index` names of all elements of *dfs* must be
    identical. The returned IamDataFrame inherits the dimensions and index names.
    """
    if not islistable(objs) or isinstance(objs, pd.DataFrame):
        raise TypeError(f"'{objs.__class__.__name__}' object is not iterable")

    objs = list(objs)
    if len(objs) < 1:
        raise ValueError("No objects to concatenate")

    def as_iamdataframe(df):
        if isinstance(df, IamDataFrame):
            return df, True
        else:
            return IamDataFrame(df, **kwargs), False

    # cast first item to IamDataFrame (if necessary)
    df, _merge_meta = as_iamdataframe(objs[0])
    extra_cols, time_col = df.extra_cols, df.time_col

    consistent_time_domain = True
    iam_dfs = [(df, _merge_meta)]

    # cast all items to IamDataFrame (if necessary) and check consistency of items
    for df in objs[1:]:
        df, _merge_meta = as_iamdataframe(df)
        if df.extra_cols != extra_cols:
            raise ValueError("Items have incompatible timeseries data dimensions")
        if df.time_col != time_col:
            consistent_time_domain = False
        iam_dfs.append((df, _merge_meta))

    # cast all instances to "time"
    if not consistent_time_domain:
        _iam_dfs = []
        for (df, _merge_meta) in iam_dfs:
            if df.time_col == "year":
                df = df.swap_year_for_time()
            _iam_dfs.append((df, _merge_meta))
        iam_dfs = _iam_dfs  # replace list of IamDataFrames with consistent list

    # extract timeseries data and meta attributes
    ret_data, ret_meta = [], None
    for (df, _merge_meta) in iam_dfs:
        ret_data.append(df._data)
        if _merge_meta:
            ret_meta = (
                df.meta
                if ret_meta is None
                else merge_meta(ret_meta, df.meta, ignore_meta_conflict)
            )

    # return as new IamDataFrame, this will verify integrity as part of `__init__()`
    return IamDataFrame(
        pd.concat(ret_data, verify_integrity=False),
        meta=ret_meta,
        index=ret_meta.index.names,
    )


def read_datapackage(path, data="data", meta="meta"):
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
        raise ImportError("Required package `datapackage` not found!")

    package = Package(path)

    def _get_column_names(x):
        return [i["name"] for i in x.descriptor["schema"]["fields"]]

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
