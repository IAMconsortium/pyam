from pathlib import Path
import itertools
import logging
import string
import six
import re
import dateutil

from pyam.index import get_index_levels, replace_index_labels
from pyam.logging import raise_data_error
import numpy as np
import pandas as pd
from collections.abc import Iterable

logger = logging.getLogger(__name__)

# common indices
DEFAULT_META_INDEX = ["model", "scenario"]
META_IDX = ["model", "scenario"]
YEAR_IDX = ["model", "scenario", "region", "year"]
IAMC_IDX = ["model", "scenario", "region", "variable", "unit"]
SORT_IDX = ["model", "scenario", "variable", "year", "region"]
LONG_IDX = IAMC_IDX + ["year"]

# required columns
REQUIRED_COLS = ["region", "variable", "unit"]

# illegal terms for data/meta column names to prevent attribute conflicts
ILLEGAL_COLS = ["data", "meta", "level", ""]

# dictionary to translate column count to Excel column names
NUMERIC_TO_STR = dict(
    zip(
        range(0, 702),
        [i for i in string.ascii_uppercase]
        + [
            "{}{}".format(i, j)
            for i, j in itertools.product(
                string.ascii_uppercase, string.ascii_uppercase
            )
        ],
    )
)

KNOWN_FUNCS = {
    "min": np.min,
    "max": np.max,
    "avg": np.mean,
    "mean": np.mean,
    "sum": np.sum,
}


def requires_package(pkg, msg, error_type=ImportError):
    """Decorator when a function requires an optional dependency

    Parameters
    ----------
    pkg : imported package object
    msg : string
        Message to show to user with error_type
    error_type : python error class
    """

    def _requires_package(func):
        def wrapper(*args, **kwargs):
            if pkg is None:
                raise error_type(msg)
            return func(*args, **kwargs)

        return wrapper

    return _requires_package


def isstr(x):
    """Returns True if x is a string"""
    return isinstance(x, six.string_types)


def isscalar(x):
    """Returns True if x is a scalar"""
    return not isinstance(x, Iterable) or isstr(x)


def islistable(x):
    """Returns True if x is a list but not a string"""
    return isinstance(x, Iterable) and not isstr(x)


def to_list(x):
    """Return x as a list"""
    return x if islistable(x) else [x]


def remove_from_list(x, items):
    """Remove `items` from list `x`"""
    items = to_list(items)
    return [i for i in x if i not in items]


def write_sheet(writer, name, df, index=False):
    """Write a pandas.DataFrame to an ExcelWriter

    The function applies formatting of the column width depending on maxwidth
    of values and column header

    Parameters
    ----------
    writer: pandas.ExcelWriter
        an instance of a :class:`pandas.ExcelWriter`
    name: string
        name of the sheet to be written
    df: pandas.DataFrame
        a :class:`pandas.DataFrame` to be written to the sheet
    index: boolean, default False
        should the index be written to the sheet
    """
    if index:
        df = df.reset_index()
    df.to_excel(writer, name, index=False)
    for i, col in enumerate(df.columns):
        if df.dtypes[col].name.startswith(("float", "int")):
            width = len(str(col)) + 2
        else:
            width = (
                max([df[col].map(lambda x: len(str(x or "None"))).max(), len(col)]) + 2
            )
        writer.sheets[name].set_column(i, i, width)  # assumes xlsxwriter as engine


def read_pandas(path, sheet_name=["data*", "Data*"], *args, **kwargs):
    """Read a file and return a pandas.DataFrame"""

    if isinstance(path, Path) and path.suffix == ".csv":
        return pd.read_csv(path, *args, **kwargs)

    else:
        xl = path if isinstance(path, pd.ExcelFile) else pd.ExcelFile(path)

        # reading multiple sheets
        sheet_names = pd.Series(xl.sheet_names)
        if len(sheet_names) > 1:
            sheets = kwargs.pop("sheet_name", sheet_name)
            # apply pattern-matching for sheet names (use * as wildcard)
            sheets = sheet_names[pattern_match(sheet_names, values=sheets)]
            if sheets.empty:
                raise ValueError(f"No sheets {sheet_name} in file {path}!")

            df = pd.concat([xl.parse(s, *args, **kwargs) for s in sheets])

        # read single sheet (if only one exists in file) ignoring sheet name
        else:
            df = pd.read_excel(xl, *args, **kwargs)

        # remove unnamed and empty columns, and rows were all values are nan
        def is_empty(name, s):
            if str(name).startswith("Unnamed: "):
                try:
                    if len(s) == 0 or all(np.isnan(s)):
                        return True
                except TypeError:
                    pass
            return False

        empty_cols = [c for c in df.columns if is_empty(c, df[c])]
        return df.drop(columns=empty_cols).dropna(axis=0, how="all")


def read_file(path, *args, **kwargs):
    """Read data from a file"""
    # extract kwargs that are intended for `format_data`
    format_kwargs = dict(index=kwargs.pop("index"))
    for c in [i for i in IAMC_IDX + ["year", "time", "value"] if i in kwargs]:
        format_kwargs[c] = kwargs.pop(c)
    return format_data(read_pandas(path, *args, **kwargs), **format_kwargs)


def _convert_r_columns(df):
    """Check and convert R-style year columns"""

    def strip_R_integer_prefix(c):
        try:
            first = c[0]
            second = c[1:]
            if first == "X":
                try:
                    #  bingo! was X2015 R-style, return the integer
                    return int(second)
                except:
                    # nope, not an int, fall down to final return statement
                    pass
        except:
            # not a string/iterable/etc, fall down to final return statement
            pass
        return c

    return df.set_axis(df.columns.map(strip_R_integer_prefix), axis="columns")


def _knead_data(df, **kwargs):
    """Replace, rename and concat according to user arguments"""

    # if `value` is given but not `variable`,
    # melt value columns and use column name as `variable`
    if "value" in kwargs and "variable" not in kwargs:
        value = kwargs.pop("value")
        value = value if islistable(value) else [value]
        _df = df.set_index(list(set(df.columns) - set(value)))
        dfs = []
        for v in value:
            if v not in df.columns:
                raise ValueError("column `{}` does not exist!".format(v))
            vdf = _df[v].to_frame().rename(columns={v: "value"})
            vdf["variable"] = v
            dfs.append(vdf.reset_index())
        df = pd.concat(dfs).reset_index(drop=True)

    # otherwise, rename columns or concat to IAMC-style or do a fill-by-value
    for col, value in kwargs.items():
        if col in df:
            raise ValueError(f"Conflict of kwarg with column `{col}` in dataframe!")

        if isstr(value) and value in df:
            df.rename(columns={value: col}, inplace=True)
        elif islistable(value) and all([c in df.columns for c in value]):
            df[col] = df.apply(lambda x: concat_with_pipe(x, value), axis=1)
            df.drop(value, axis=1, inplace=True)
        elif isstr(value):
            df[col] = value
        else:
            raise ValueError(f"Invalid argument for casting `{col}: {value}`")

    return df


def _format_from_legacy_database(df):
    """Process data from legacy databases (SSP and earlier)"""

    logger.info("Ignoring notes column in `data`")
    df.drop(columns="notes", inplace=True)
    col = df.columns[0]  # first column has database copyright notice
    df = df[~df[col].str.contains("database", case=False)]
    if "scenario" in df.columns and "model" not in df.columns:
        # model and scenario are jammed together in RCP data
        parts = df["scenario"].str.split("-", n=1, expand=True)
        df = df.assign(model=parts[0].str.strip(), scenario=parts[1].str.strip())

    return df


def _intuit_column_groups(df, index, include_index=False):
    """Check and categorise columns in dataframe"""

    if include_index:
        existing_cols = pd.Index(df.index.names)
    else:
        existing_cols = pd.Index([])
    if isinstance(df, pd.Series):
        existing_cols = existing_cols.union(["value"])
    elif isinstance(df, pd.DataFrame):
        existing_cols = existing_cols.union(df.columns)

    # check that there is no column in the timeseries data with reserved names
    if None in existing_cols:
        raise ValueError("Unnamed column in `data`: None")

    # check that there is no column in the timeseries data with reserved/illegal names
    conflict_cols = [i for i in existing_cols if i in ILLEGAL_COLS]
    if conflict_cols:
        sep = "', '"
        _cols = f"'{sep.join(conflict_cols)}'"
        _args = ", ".join([f"<alternative_column_name>='{i}'" for i in conflict_cols])
        raise ValueError(
            f"Illegal column{s(len(conflict_cols))} for timeseries data: {_cols}\n"
            f"Use `IamDataFrame(..., {_args})` to rename at initialization."
        )

    # check that index and required columns exist
    missing_index = [c for c in index if c not in existing_cols]
    if missing_index:
        raise ValueError(f"Missing index columns: {missing_index}")

    missing_required_col = [c for c in REQUIRED_COLS if c not in existing_cols]
    if missing_required_col:
        raise ValueError(f"Missing required columns: {missing_required_col}")

    # check whether data in wide format (standard IAMC) or long format (`value` column)
    if "value" in existing_cols:
        # check if time column is given as `year` (int) or `time` (datetime)
        if "year" in existing_cols and "time" not in existing_cols:
            time_col = "year"
        elif "time" in existing_cols and "year" not in existing_cols:
            time_col = "time"
        else:
            raise ValueError("Invalid time domain, must have either `year` or `time`!")
        extra_cols = [
            c
            for c in existing_cols
            if c not in index + REQUIRED_COLS + [time_col, "value"]
        ]
        data_cols = []
    else:
        # if in wide format, check if columns are years (int) or datetime
        cols = [c for c in existing_cols if c not in index + REQUIRED_COLS]
        year_cols, time_cols, extra_cols = [], [], []
        for i in cols:
            # if the column name can be cast to integer, assume it's a year column
            try:
                int(i)
                year_cols.append(i)

            # otherwise, try casting to datetime
            except (ValueError, TypeError):
                try:
                    dateutil.parser.parse(str(i))
                    time_cols.append(i)

                # neither year nor datetime, so it is an extra-column
                except ValueError:
                    extra_cols.append(i)

        if year_cols and not time_cols:
            time_col = "year"
            data_cols = sorted(year_cols)
        else:
            time_col = "time"
            data_cols = sorted(year_cols) + sorted(time_cols)
        if not data_cols:
            raise ValueError("Missing time domain")

    return time_col, extra_cols, data_cols


def _format_data_to_series(df, index):
    """Convert a long or wide pandas dataframe to a series with the required columns"""

    time_col, extra_cols, data_cols = _intuit_column_groups(df, index)

    _validate_complete_index(df[index + REQUIRED_COLS + extra_cols])

    idx_order = index + REQUIRED_COLS + [time_col] + extra_cols

    if data_cols:
        # wide format
        df = (
            df.set_index(index + REQUIRED_COLS + extra_cols)
            .rename_axis(columns=time_col)
            .stack(dropna=True)
            .rename("value")
            .reorder_levels(idx_order)
        )
    else:
        # long format
        df = df.set_index(idx_order)["value"].dropna()

    return df, time_col, extra_cols


def format_data(df, index, **kwargs):
    """Convert a pandas.Dataframe or pandas.Series to the required format"""

    # Fast-pass if `df` has the index and required columns as a pd.MultiIndex
    if set(df.index.names) >= set(index) | set(REQUIRED_COLS) and not kwargs:
        time_col, extra_cols, data_cols = _intuit_column_groups(
            df, index=index, include_index=True
        )

        if isinstance(df, pd.DataFrame):
            extra_cols_not_in_index = [c for c in extra_cols if c in df.columns]
            if extra_cols_not_in_index:
                df = df.set_index(extra_cols_not_in_index, append=True)

            if data_cols:
                df = df[data_cols].rename_axis(columns=time_col).stack().rename("value")
            else:
                df = df["value"]

        df = df.reorder_levels(index + REQUIRED_COLS + [time_col] + extra_cols).dropna()

    else:
        if isinstance(df, pd.Series):
            if not df.name:
                df = df.rename("value")
            df = df.reset_index()
        elif not list(df.index.names) == [None]:
            # reset the index if meaningful entries are included there
            df = df.reset_index()

        df = _convert_r_columns(df)

        if kwargs:
            df = _knead_data(df, **kwargs)

        # all lower case
        df.rename(
            columns={c: str(c).lower() for c in df.columns if isstr(c)}, inplace=True
        )

        if "notes" in df.columns:  # this came from a legacy database (SSP or earlier)
            df = _format_from_legacy_database(df)

        # replace missing units by an empty string for user-friendly filtering
        if "unit" in df.columns:
            df = df.assign(unit=df["unit"].fillna(""))

        df, time_col, extra_cols = _format_data_to_series(df, index)

    # cast value column to numeric
    try:
        df = pd.to_numeric(df)
    except ValueError as e:
        # get the row number where the error happened
        row_nr_regex = re.compile(r"(?<=at position )\d+")
        row_nr = int(row_nr_regex.search(str(e)).group())
        short_error_regex = re.compile(r".*(?= at position \d*)")
        short_error = short_error_regex.search(str(e)).group()
        raise_data_error(f"{short_error} in `data`", df.iloc[[row_nr]])

    # format the time-column
    _time = [to_time(i) for i in get_index_levels(df.index, time_col)]
    df.index = replace_index_labels(df.index, time_col, _time)

    rows = df.index.duplicated()
    if any(rows):
        raise_data_error(
            "Duplicate rows in `data`", df[rows].index.to_frame(index=False)
        )
    del rows
    if df.empty:
        logger.warning("Formatted data is empty!")

    return df.sort_index(), index, time_col, extra_cols


def _validate_complete_index(df):
    """Validate that there are no nan's in the (index) columns"""
    null_cells = df.isnull()
    null_rows = null_cells.any(axis=1)
    if null_rows.any():
        null_cols = null_cells.any()
        cols = ", ".join(null_cols.index[null_cols])
        raise_data_error(
            f"Empty cells in `data` (columns: '{cols}')", df.loc[null_rows]
        )


def sort_data(data, cols):
    """Sort data rows and order columns by cols"""
    return data.sort_values(cols)[cols + ["value"]].reset_index(drop=True)


def merge_meta(left, right, ignore_conflict=False):
    """Merge two `meta` tables; raise if values are in conflict (optional)

    If conflicts are ignored, values in `left` take precedence over `right`.
    """
    left = left.copy()  # make a copy to not change the original object
    diff = right.index.difference(left.index)
    sect = right.index.intersection(left.index)

    # merge `right` into `left` for overlapping scenarios ( `sect`)
    if not sect.empty:
        # if not ignored, check that overlapping `meta` columns are equal
        if not ignore_conflict:
            cols = [i for i in right.columns if i in left.columns]
            if not left.loc[sect, cols].equals(right.loc[sect, cols]):
                conflict_idx = (
                    pd.concat([right.loc[sect, cols], left.loc[sect, cols]])
                    .drop_duplicates()
                    .index.drop_duplicates()
                )
                msg = "conflict in `meta` for scenarios {}".format(
                    [i for i in pd.DataFrame(index=conflict_idx).index]
                )
                raise ValueError(msg)
        # merge new columns
        cols = [i for i in right.columns if i not in left.columns]
        left = left.merge(
            right.loc[sect, cols], how="outer", left_index=True, right_index=True
        )

    # join `other.meta` for new scenarios (`diff`)
    if not diff.empty:
        left = pd.concat([left, right.loc[diff, :]], sort=False)

    # remove any columns that are all-nan
    return left.dropna(axis=1, how="all")


def find_depth(data, s="", level=None):
    """Return or assert the depth (number of ``|``) of variables

    Parameters
    ----------
    data : str or list of strings
        IAMC-style variables
    s : str, default ''
        remove leading `s` from any variable in `data`
    level : int or str, default None
        if None, return depth (number of ``|``); else, return list of booleans
        whether depth satisfies the condition (equality if level is int,
        >= if ``.+``,  <= if ``.-``)
    """
    if islistable(level):
        raise ValueError(
            "Level is only run with ints or strings, not lists. Use strings with "
            "integers and + or - to filter by ranges."
        )
    if isstr(data):
        return _find_depth([data], s, level)[0]

    return _find_depth(data, s, level)


def _find_depth(data, s="", level=None):
    """Internal implementation of `find_depth()´"""
    # remove wildcard as last character from string, escape regex characters
    _s = re.compile("^" + _escape_regexp(s.rstrip("*")))
    _p = re.compile("\\|")

    # find depth
    def _count_pipes(val):
        return len(_p.findall(re.sub(_s, "", val))) if _s.match(val) else None

    n_pipes = map(_count_pipes, to_list(data))

    # if no level test is specified, return the depth as (list of) int
    if level is None:
        return list(n_pipes)

    # if `level` is given, set function for finding depth level =, >=, <= |s
    if not isstr(level):
        test = lambda x: level == x if x is not None else False
    elif level[-1] == "-":
        level = int(level[:-1])
        test = lambda x: level >= x if x is not None else False
    elif level[-1] == "+":
        level = int(level[:-1])
        test = lambda x: level <= x if x is not None else False
    else:
        raise ValueError("Unknown level type: `{}`".format(level))

    return list(map(test, n_pipes))


def pattern_match(
    data, values, level=None, regexp=False, has_nan=False, return_codes=False
):
    """Return list where data matches values

    The function matches model/scenario names, variables, regions
    and meta columns to pseudo-regex (if `regexp == False`)
    for filtering (str, int, bool)
    """
    codes = []
    matches = np.zeros(len(data), dtype=bool)
    values = values if islistable(values) else [values]

    # issue (#40) with string-to-nan comparison, replace nan by empty string
    _data = data.fillna("") if has_nan else data

    for s in values:
        if return_codes and isinstance(data, pd.Index):
            try:
                codes.append(data.get_loc(s))
                continue
            except KeyError:
                pass

        if isstr(s):
            pattern = re.compile(_escape_regexp(s) + "$" if not regexp else s)
            depth = True if level is None else find_depth(_data, s, level)
            matches |= data.str.match(pattern) & depth
        else:
            matches = np.logical_or(matches, data == s)

    if return_codes:
        codes.extend(np.where(matches)[0])
        return codes

    return matches


def _escape_regexp(s):
    """Escape characters with specific regexp use"""
    return (
        str(s)
        .replace("|", "\\|")
        .replace(".", "\.")  # `.` has to be replaced before `*`
        .replace("*", ".*")
        .replace("+", "\+")
        .replace("(", "\(")
        .replace(")", "\)")
        .replace("$", "\\$")
    )


def print_list(x, n):
    """Return a printable string of a list shortened to n characters"""
    # if list is empty, only write count
    if len(x) == 0:
        return "(0)"

    # write number of elements, subtract count added at end from line width
    x = [i if i != "" else "''" for i in map(str, x)]
    count = f" ({len(x)})"
    n -= len(count)

    # if not enough space to write first item, write shortest sensible line
    if len(x[0]) > n - 5:
        return "..." + count

    # if only one item in list
    if len(x) == 1:
        return f"{x[0]} (1)"

    # add first item
    lst = f"{x[0]}, "
    n -= len(lst)

    # if possible, add last item before number of elements
    if len(x[-1]) + 4 > n:
        return lst + "..." + count
    else:
        count = f"{x[-1]}{count}"
        n -= len({x[-1]}) + 3

    # iterate over remaining entries until line is full
    for i in x[1:-1]:
        if len(i) + 6 <= n:
            lst += f"{i}, "
            n -= len(i) + 2
        else:
            lst += "... "
            break

    return lst + count


def to_time(x):
    """Cast a value to either year (int) or datetime"""

    # if the column name can be cast to integer, assume it's a year column
    try:
        j = int(x)
        is_year = True

    # otherwise, try casting to Timestamp (pandas-equivalent of datetime)
    except (ValueError, TypeError):
        try:
            j = pd.Timestamp(x)
            is_year = False
        except ValueError:
            raise ValueError(f"Invalid time domain: {x}")

    # This is to guard against "years" with decimals (e.g., '2010.5')
    if is_year and float(x) != j:
        raise ValueError(f"Invalid time domain: {x}")

    return j


def to_int(x, index=False):
    """Formatting series or timeseries columns to int and checking validity

    If `index=False`, the function works on the :class:`pandas.Series` x;
    else, the function casts the index of x to int and returns x with new index
    """
    _x = x.index if index else x
    cols = list(map(int, _x))
    error = _x[cols != _x]
    if not error.empty:
        raise ValueError(f"Invalid values: {error}")
    if index:
        x.index = cols
        return x
    else:
        return _x


def concat_with_pipe(x, cols=None):
    """Concatenate a pandas.Series x using ``|``, drop None or numpy.nan"""
    cols = cols or x.index
    return "|".join([x[i] for i in cols if x[i] not in [None, np.nan]])


def reduce_hierarchy(x, depth):
    """Reduce the hierarchy (indicated by ``|``) of x to the specified depth"""
    _x = x.split("|")
    depth = len(_x) + depth - 1 if depth < 0 else depth
    return "|".join(_x[0 : (depth + 1)])


def get_variable_components(x, level, join=False):
    """Return components for requested level in a list or join these in a str.

    Parameters
    ----------
    x : str
        Uses ``|`` to separate the components of the variable.
    level : int or list of int
        Position of the component.
    join : bool or str, optional
        If True, IAMC-style (``|``) is used as separator for joined components.
    """
    _x = x.split("|")
    if join is False:
        return [_x[i] for i in level] if islistable(level) else _x[level]
    else:
        level = [level] if type(level) == int else level
        join = "|" if join is True else join
        return join.join([_x[i] for i in level])


def s(n):
    """Return an s if n!=1 for nicer formatting of log messages"""
    return "s" if n != 1 else ""
