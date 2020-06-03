import itertools
import logging
import string
import six
import re
import collections
import datetime
import dateutil
import time

import numpy as np
import pandas as pd

try:
    import seaborn as sns
except ImportError:
    pass

logger = logging.getLogger(__name__)

# common indices
META_IDX = ['model', 'scenario']
YEAR_IDX = ['model', 'scenario', 'region', 'year']
IAMC_IDX = ['model', 'scenario', 'region', 'variable', 'unit']
SORT_IDX = ['model', 'scenario', 'variable', 'year', 'region']
LONG_IDX = IAMC_IDX + ['year']

# dictionary to translate column count to Excel column names
NUMERIC_TO_STR = dict(zip(range(0, 702),
                          [i for i in string.ascii_uppercase]
                          + ['{}{}'.format(i, j) for i, j in itertools.product(
                              string.ascii_uppercase, string.ascii_uppercase)]))

KNOWN_FUNCS = {'min': np.min, 'max': np.max, 'avg': np.mean, 'mean': np.mean,
               'sum': np.sum}


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
    return not isinstance(x, collections.Iterable) or isstr(x)


def islistable(x):
    """Returns True if x is a list but not a string"""
    return isinstance(x, collections.Iterable) and not isstr(x)


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
    worksheet = writer.sheets[name]
    for i, col in enumerate(df.columns):
        if df.dtypes[col].name.startswith(('float', 'int')):
            width = len(str(col)) + 2
        else:
            width = max([df[col].map(lambda x: len(str(x or 'None'))).max(),
                         len(col)]) + 2
        # this line fails if using an xlsx-engine other than openpyxl
        try:
            worksheet.column_dimensions[NUMERIC_TO_STR[i]].width = width
        except AttributeError:
            pass


def read_pandas(path, default_sheet='data', *args, **kwargs):
    """Read a file and return a pandas.DataFrame"""
    if path.endswith('csv'):
        df = pd.read_csv(path, *args, **kwargs)
    else:
        xl = pd.ExcelFile(path)
        if len(xl.sheet_names) > 1 and 'sheet_name' not in kwargs:
            kwargs['sheet_name'] = default_sheet
        df = pd.read_excel(path, *args, **kwargs)
    return df


def read_file(path, *args, **kwargs):
    """Read data from a file"""
    if not isstr(path):
        raise ValueError('Reading multiple files not supported, '
                         'use `IamDataFrame.append()` or `pyam.concat()`')
    format_kwargs = {}
    # extract kwargs that are intended for `format_data`
    for c in [i for i in IAMC_IDX + ['year', 'time', 'value'] if i in kwargs]:
        format_kwargs[c] = kwargs.pop(c)
    return format_data(read_pandas(path, *args, **kwargs), **format_kwargs)


def format_data(df, **kwargs):
    """Convert a pandas.Dataframe or pandas.Series to the required format"""
    if isinstance(df, pd.Series):
        df.name = df.name or 'value'
        df = df.to_frame()

    # Check for R-style year columns, converting where necessary
    def convert_r_columns(c):
        try:
            first = c[0]
            second = c[1:]
            if first == 'X':
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
    df.columns = df.columns.map(convert_r_columns)

    # if `value` is given but not `variable`,
    # melt value columns and use column name as `variable`
    if 'value' in kwargs and 'variable' not in kwargs:
        value = kwargs.pop('value')
        value = value if islistable(value) else [value]
        _df = df.set_index(list(set(df.columns) - set(value)))
        dfs = []
        for v in value:
            if v not in df.columns:
                raise ValueError('column `{}` does not exist!'.format(v))
            vdf = _df[v].to_frame().rename(columns={v: 'value'})
            vdf['variable'] = v
            dfs.append(vdf.reset_index())
        df = pd.concat(dfs).reset_index(drop=True)

    # otherwise, rename columns or concat to IAMC-style or do a fill-by-value
    for col, value in kwargs.items():
        if col in df:
            raise ValueError('conflict of kwarg with column `{}` in dataframe!'
                             .format(col))

        if isstr(value) and value in df:
            df.rename(columns={value: col}, inplace=True)
        elif islistable(value) and all([c in df.columns for c in value]):
            df[col] = df.apply(lambda x: concat_with_pipe(x, value), axis=1)
            df.drop(value, axis=1, inplace=True)
        elif isstr(value):
            df[col] = value
        else:
            raise ValueError('invalid argument for casting `{}: {}`'
                             .format(col, value))

    # all lower case
    str_cols = [c for c in df.columns if isstr(c)]
    df.rename(columns={c: str(c).lower() for c in str_cols}, inplace=True)

    if 'notes' in df.columns:  # this came from the database
        logger.info('Ignoring notes column in dataframe')
        df.drop(columns='notes', inplace=True)
        col = df.columns[0]  # first column has database copyright notice
        df = df[~df[col].str.contains('database', case=False)]
        if 'scenario' in df.columns and 'model' not in df.columns:
            # model and scenario are jammed together in RCP data
            scen = df['scenario']
            df.loc[:, 'model'] = scen.apply(lambda s: s.split('-')[0].strip())
            df.loc[:, 'scenario'] = scen.apply(
                lambda s: '-'.join(s.split('-')[1:]).strip())

    # reset the index if meaningful entries are included there
    if not list(df.index.names) == [None]:
        df.reset_index(inplace=True)

    # format columns to lower-case and check that all required columns exist
    if not set(IAMC_IDX).issubset(set(df.columns)):
        missing = list(set(IAMC_IDX) - set(df.columns))
        raise ValueError("missing required columns `{}`!".format(missing))

    # check whether data in wide format (IAMC) or long format (`value` column)
    if 'value' in df.columns:
        # check if time column is given as `year` (int) or `time` (datetime)
        cols = df.columns
        if 'year' in cols:
            time_col = 'year'
        elif 'time' in cols:
            time_col = 'time'
        else:
            msg = 'invalid time format, must have either `year` or `time`!'
            raise ValueError(msg)
        extra_cols = list(set(cols) - set(IAMC_IDX + [time_col, 'value']))
    else:
        # if in wide format, check if columns are years (int) or datetime
        cols = set(df.columns) - set(IAMC_IDX)
        year_cols, time_cols, extra_cols = [], [], []
        for i in cols:
            try:
                int(i)  # this is a year
                year_cols.append(i)
            except (ValueError, TypeError):
                try:
                    dateutil.parser.parse(str(i))  # this is datetime
                    time_cols.append(i)
                except ValueError:
                    extra_cols.append(i)  # some other string
        if year_cols and not time_cols:
            time_col = 'year'
            melt_cols = year_cols
        elif not year_cols and time_cols:
            time_col = 'time'
            melt_cols = time_cols
        else:
            msg = 'invalid column format, must be either years or `datetime`!'
            raise ValueError(msg)
        df = pd.melt(df, id_vars=IAMC_IDX + extra_cols, var_name=time_col,
                     value_vars=sorted(melt_cols), value_name='value')

    # cast value column to numeric and drop nan
    df['value'] = df['value'].astype('float64')
    df.dropna(inplace=True, subset=['value'])

    # verify that there are no nan's left (in columns)
    null_rows = df.isnull().values
    if null_rows.any():
        _raise_data_error('empty cells in `data`', df.loc[null_rows])

    # check for duplicates and empty data
    idx_cols = IAMC_IDX + [time_col] + extra_cols
    rows = df[idx_cols].duplicated()
    if any(rows):
        _raise_data_error('duplicate rows in `data`', df.loc[rows, idx_cols])

    if df.empty:
        logger.warning('Formatted data is empty!')

    return sort_data(df, idx_cols), time_col, extra_cols


def _raise_data_error(msg, data):
    """Utils function to format error message from data formatting"""
    data = data.drop_duplicates()
    msg = f'{msg}:\n{data.head()}' + ('\n...' if len(data) > 5 else '')
    logger.error(msg)
    raise ValueError(msg)


def sort_data(data, cols):
    """Sort data rows and order columns by cols"""
    return data.sort_values(cols)[cols + ['value']].reset_index(drop=True)


def find_depth(data, s='', level=None):
    """Return or assert the depth (number of ``|``) of variables

    Parameters
    ----------
    data : pandas.Series of strings
        IAMC-style variables
    s : str, default ''
        remove leading `s` from any variable in `data`
    level : int or str, default None
        if None, return depth (number of ``|``); else, return list of booleans
        whether depth satisfies the condition (equality if level is int,
        >= if ``.+``,  <= if ``.-``)
    """
    # remove wildcard as last character from string, escape regex characters
    _s = re.compile('^' + _escape_regexp(s.rstrip('*')))
    _p = re.compile('\\|')

    # find depth
    def _count_pipes(val):
        return len(_p.findall(re.sub(_s, '', val))) if _s.match(val) else None

    n_pipes = map(_count_pipes, data)

    # if no level test is specified, return the depth as int
    if level is None:
        return list(n_pipes)

    # if `level` is given, set function for finding depth level =, >=, <= |s
    if not isstr(level):
        test = lambda x: level == x if x is not None else False
    elif level[-1] == '-':
        level = int(level[:-1])
        test = lambda x: level >= x if x is not None else False
    elif level[-1] == '+':
        level = int(level[:-1])
        test = lambda x: level <= x if x is not None else False
    else:
        raise ValueError('Unknown level type: `{}`'.format(level))

    return list(map(test, n_pipes))


def pattern_match(data, values, level=None, regexp=False, has_nan=True):
    """Return list where data matches values

    The function matches model/scenario names, variables, regions
    and meta columns to pseudo-regex (if `regexp == False`)
    for filtering (str, int, bool)
    """
    matches = np.array([False] * len(data))
    if not isinstance(values, collections.Iterable) or isstr(values):
        values = [values]

    # issue (#40) with string-to-nan comparison, replace nan by empty string
    _data = data.copy()
    if has_nan:
        _data.loc[[np.isnan(i) if not isstr(i) else False for i in _data]] = ''

    for s in values:
        if isstr(s):
            pattern = re.compile(_escape_regexp(s) + '$' if not regexp else s)
            subset = filter(pattern.match, _data)
            depth = True if level is None else find_depth(_data, s, level)
            matches = np.logical_or(matches, _data.isin(subset) & depth)
        else:
            matches = np.logical_or(matches, data == s)
    return matches


def _escape_regexp(s):
    """Escape characters with specific regexp use"""
    return (
        str(s)
        .replace('|', '\\|')
        .replace('.', '\.')  # `.` has to be replaced before `*`
        .replace('*', '.*')
        .replace('+', '\+')
        .replace('(', '\(')
        .replace(')', '\)')
        .replace('$', '\\$')
    )


def years_match(data, years):
    """Return rows where data matches year"""
    years = [years] if (
        isinstance(years, (int, np.int64))
    ) else years
    dt = (datetime.datetime, np.datetime64)
    if isinstance(years, dt) or isinstance(years[0], dt):
        error_msg = "`year` can only be filtered with ints or lists of ints"
        raise TypeError(error_msg)
    return data.isin(years)


def month_match(data, months):
    """Return rows where data matches months"""
    return time_match(data, months, ['%b', '%B'], "tm_mon", "months")


def day_match(data, days):
    """Return rows where data matches days"""
    return time_match(data, days, ['%a', '%A'], "tm_wday", "days")


def hour_match(data, hours):
    """Return rows where data matches hours"""
    hours = [hours] if isinstance(hours, int) else hours
    return data.isin(hours)


def time_match(data, times, conv_codes, strptime_attr, name):
    """Return rows where data matches a timestamp"""
    def conv_strs(strs_to_convert, conv_codes, name):
        for conv_code in conv_codes:
            try:
                res = [getattr(time.strptime(t, conv_code), strptime_attr)
                       for t in strs_to_convert]
                break
            except ValueError:
                continue

        try:
            return res
        except NameError:
            raise ValueError("Could not convert {} to integer".format(name))

    times = [times] if isinstance(times, (int, str)) else times
    if isinstance(times[0], str):
        to_delete = []
        to_append = []
        for i, timeset in enumerate(times):
            if "-" in timeset:
                ints = conv_strs(timeset.split("-"), conv_codes, name)
                if ints[0] > ints[1]:
                    error_msg = (
                        "string ranges must lead to increasing integer ranges,"
                        " {} becomes {}".format(timeset, ints)
                    )
                    raise ValueError(error_msg)

                # + 1 to include last month
                to_append += [j for j in range(ints[0], ints[1] + 1)]
                to_delete.append(i)

        for i in to_delete:
            del times[i]

        times = conv_strs(times, conv_codes, name)
        times += to_append

    return data.isin(times)


def datetime_match(data, dts):
    """Matching of datetimes in time columns for data filtering"""
    dts = dts if islistable(dts) else [dts]
    if any([not (
            isinstance(i, (datetime.datetime, np.datetime64))
    ) for i in dts]):
        error_msg = (
            "`time` can only be filtered by datetimes and datetime64s"
        )
        raise TypeError(error_msg)
    return data.isin(dts)


def to_int(x, index=False):
    """Formatting series or timeseries columns to int and checking validity

    If `index=False`, the function works on the :class:`pandas.Series` x;
    else, the function casts the index of x to int and returns x with new index
    """
    _x = x.index if index else x
    cols = list(map(int, _x))
    error = _x[cols != _x]
    if not error.empty:
        raise ValueError('invalid values `{}`'.format(list(error)))
    if index:
        x.index = cols
        return x
    else:
        return _x


def concat_with_pipe(x, cols=None):
    """Concatenate a pandas.Series x using ``|``, drop None or numpy.nan"""
    cols = cols or x.index
    return '|'.join([x[i] for i in cols if x[i] not in [None, np.nan]])


def reduce_hierarchy(x, depth):
    """Reduce the hierarchy (indicated by ``|``) of x to the specified depth"""
    _x = x.split('|')
    depth = len(_x) + depth - 1 if depth < 0 else depth
    return '|'.join(_x[0:(depth + 1)])
