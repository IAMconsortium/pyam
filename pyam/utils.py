import os
import itertools
import string
import six
import re
import glob
import collections
import datetime
import time

import numpy as np
import pandas as pd

try:
    import seaborn as sns
except ImportError:
    pass

from pyam.logger import logger

# common indicies
META_IDX = ['model', 'scenario']
YEAR_IDX = ['model', 'scenario', 'region', 'year']
REGION_IDX = ['model', 'scenario', 'variable', 'year']
IAMC_IDX = ['model', 'scenario', 'region', 'variable', 'unit']
SORT_IDX = ['model', 'scenario', 'variable', 'year', 'region']
LONG_IDX = IAMC_IDX + ['year']
GROUP_IDX = ['model', 'scenario', 'region', 'unit', 'year']

# dictionary to translate column count to Excel column names
NUMERIC_TO_STR = dict(zip(range(0, 702),
                          [i for i in string.ascii_uppercase]
                          + ['{}{}'.format(i, j) for i, j in itertools.product(
                              string.ascii_uppercase, string.ascii_uppercase)]))


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
    """Write a pandas DataFrame to an ExcelWriter,
    auto-formatting column width depending on maxwidth of data and colum header

    Parameters
    ----------
    writer: pandas.ExcelWriter
        an instance of a pandas ExcelWriter
    name: string
        name of the sheet to be written
    df: pandas.DataFrame
        a pandas DataFrame to be written to the sheet
    index: boolean, default False
        flag whether index should be written to the sheet
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
        xls_col = '{c}:{c}'.format(c=NUMERIC_TO_STR[i])
        worksheet.set_column(xls_col, width)


def read_pandas(fname, *args, **kwargs):
    """Read a file and return a pd.DataFrame"""
    if not os.path.exists(fname):
        raise ValueError('no data file `{}` found!'.format(fname))
    if fname.endswith('csv'):
        df = pd.read_csv(fname, *args, **kwargs)
    else:
        xl = pd.ExcelFile(fname)
        if len(xl.sheet_names) > 1 and 'sheet_name' not in kwargs:
            kwargs['sheet_name'] = 'data'
        df = pd.read_excel(fname, *args, **kwargs)
    return df


def read_files(fnames, *args, **kwargs):
    """Read data from a snapshot file saved in the standard IAMC format
    or a table with year/value columns
    """
    if not isstr(fnames):
        raise ValueError('reading multiple files not supported, '
                         'please use `pyam.IamDataFrame.append()`')
    logger().info('Reading `{}`'.format(fnames))
    return format_data(read_pandas(fnames, *args, **kwargs))


def format_data(df):
    """Convert an imported dataframe and check all required columns"""
    if isinstance(df, pd.Series):
        df = df.to_frame()

    # all lower case
    str_cols = [c for c in df.columns if isstr(c)]
    df.rename(columns={c: str(c).lower() for c in str_cols}, inplace=True)

    if 'notes' in df.columns:  # this came from the database
        logger().info('Ignoring notes column in dataframe')
        df.drop(columns='notes', inplace=True)
        col = df.columns[0]  # first column has database copyright notice
        df = df[~df[col].str.contains('database', case=False)]
        if 'scenario' in df.columns and 'model' not in df.columns:
            # model and scenario are jammed together in RCP data
            scen = df['scenario']
            df['model'] = scen.apply(lambda s: s.split('-')[0].strip())
            df['scenario'] = scen.apply(
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
        if 'year' in cols and 'time' not in cols:
            time_col = 'year'
        elif 'time' in cols and 'year' not in cols:
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
                year_cols.append(i) if int(i) else None
            except (ValueError, TypeError):
                try:
                    pd.to_datetime([i])
                    time_cols.append(i)
                except ValueError:
                    extra_cols.append(i)
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

    # cast time_col to correct format
    if time_col == 'year':
        if not df.year.dtype == 'int64':
            df['year'] = cast_years_to_int(pd.to_numeric(df['year']))
    if time_col == 'time':
        df['time'] = pd.to_datetime(df['time'])

    # cast value columns to numeric, drop NaN's, sort data
    df['value'] = df['value'].astype('float64')
    df.dropna(inplace=True)
    df.sort_values(META_IDX + ['variable', time_col, 'region'] + extra_cols,
                   inplace=True)

    return df, time_col, extra_cols


def style_df(df, style='heatmap'):
    if style == 'highlight_not_max':
        return df.style.apply(lambda s: ['' if v else 'background-color: yellow' for v in s == s.max()])
    if style == 'heatmap':
        cm = sns.light_palette("green", as_cmap=True)
        return df.style.background_gradient(cmap=cm)


def find_depth(data, s, level):
    # determine function for finding depth level =, >=, <= |s
    if not isstr(level):
        test = lambda x: level == x
    elif level[-1] == '-':
        level = int(level[:-1])
        test = lambda x: level >= x
    elif level[-1] == '+':
        level = int(level[:-1])
        test = lambda x: level <= x
    else:
        raise ValueError('Unknown level type: {}'.format(level))

    # determine depth
    pipe = re.compile('\\|')
    regexp = str(s).replace('*', '')
    apply_test = lambda val: test(len(pipe.findall(val.replace(regexp, ''))))
    return list(map(apply_test, data))


def pattern_match(data, values, level=None, regexp=False, has_nan=True):
    """
    matching of model/scenario names, variables, regions, and meta columns to
    pseudo-regex (if `regexp == False`) for filtering (str, int, bool)
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
            _regexp = (str(s)
                       .replace('|', '\\|')
                       .replace('.', '\.')  # `.` has to be replaced before `*`
                       .replace('*', '.*')
                       .replace('+', '\+')
                       .replace('(', '\(')
                       .replace(')', '\)')
                       .replace('$', '\\$')
                       ) + "$"
            pattern = re.compile(_regexp if not regexp else s)

            subset = filter(pattern.match, _data)
            depth = True if level is None else find_depth(_data, s, level)
            matches |= (_data.isin(subset) & depth)
        else:
            matches |= data == s
    return matches


def years_match(data, years):
    """
    matching of year columns for data filtering
    """
    years = [years] if isinstance(years, int) else years
    dt = datetime.datetime
    if isinstance(years, dt) or isinstance(years[0], dt):
        error_msg = "`year` can only be filtered with ints or lists of ints"
        raise TypeError(error_msg)
    return data.isin(years)


def month_match(data, months):
    """
    matching of months in time columns for data filtering
    """
    return time_match(data, months, ['%b', '%B'], "tm_mon", "months")


def day_match(data, days):
    """
    matching of days in time columns for data filtering
    """
    return time_match(data, days, ['%a', '%A'], "tm_wday", "days")


def hour_match(data, hours):
    """
    matching of days in time columns for data filtering
    """
    hours = [hours] if isinstance(hours, int) else hours
    return data.isin(hours)


def time_match(data, times, conv_codes, strptime_attr, name):
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
    """
    matching of datetimes in time columns for data filtering
    """
    dts = [dts] if isinstance(dts, datetime.datetime) else dts
    if isinstance(dts, int) or isinstance(dts[0], int):
        error_msg = (
            "`time` can only be filtered with datetimes or lists of datetimes"
        )
        raise TypeError(error_msg)
    return data.isin(dts)


def cast_years_to_int(x, index=False):
    """Formatting series or timeseries columns to int and checking validity.
    If `index=False`, the function works on the `pd.Series x`; else,
    the function casts the index of `x` to int and returns x with a new index.
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
