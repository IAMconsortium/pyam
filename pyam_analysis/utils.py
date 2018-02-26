import os
import itertools
import string
import logging
import six
import re
import glob

import numpy as np
import pandas as pd

try:
    import ixmp
except ImportError:
    pass

try:
    import seaborn as sns
except ImportError:
    pass

_LOGGER = None

# common indicies
META_IDX = ['model', 'scenario']
IAMC_IDX = ['model', 'scenario', 'region', 'variable', 'unit']

# dictionary to translate column count to Excel column names
NUMERIC_TO_STR = dict(zip(range(0, 702),
                          [i for i in string.ascii_uppercase]
                          + ['{}{}'.format(i, j) for i, j in itertools.product(
                              string.ascii_uppercase, string.ascii_uppercase)]))


def logger():
    """Access global logger"""
    global _LOGGER
    if _LOGGER is None:
        logging.basicConfig()
        _LOGGER = logging.getLogger()
        _LOGGER.setLevel('INFO')
    return _LOGGER


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


def read_ix(ix, **kwargs):
    """Read timeseries data from an ix object

    Parameters
    ----------
    ix: ixmp.TimeSeries or ixmp.Scenario
        this option requires the ixmp package as a dependency
    regions: list
        list of regions to be loaded from the database snapshot
    """
    if isinstance(ix, ixmp.TimeSeries):
        df = ix.timeseries(iamc=False, **kwargs)
        df['model'] = ix.model
        df['scenario'] = ix.scenario
    else:
        error = 'arg ' + ix + ' not recognized as valid ixmp class'
        raise ValueError(error)

    return df


def read_files(fnames, *args, **kwargs):
    """Read data from a snapshot file saved in the standard IAMC format
    or a table with year/value columns
    """
    if isinstance(fnames, six.string_types):
        fnames = [fnames]

    fnames = itertools.chain(*[glob.glob(f) for f in fnames])
    dfs = []
    for fname in fnames:
        logger().info('Reading {}'.format(fname))
        if not os.path.exists(fname):
            raise ValueError("no data file '" + fname + "' found!")
        # read from database snapshot csv or xlsx
        if fname.endswith('csv'):
            df = pd.read_csv(fname, *args, **kwargs)
        else:
            df = pd.read_excel(fname, *args, **kwargs)
        df.rename(columns={c: str(c).lower()
                           for c in df.columns}, inplace=True)
        dfs.append(df)

    return format_data(pd.concat(dfs))


def format_data(df):
    """Convert an imported dataframe and check all required columns"""

    # format columns to lower-case and check that all required columns exist
    df.rename(columns={c: str(c).lower() for c in df.columns}, inplace=True)
    if not set(IAMC_IDX).issubset(set(df.columns)):
        missing = list(set(IAMC_IDX) - set(df.columns))
        raise ValueError("missing required columns {}!".format(missing))

    # check whether data in IAMC style or year/value layout
    if 'value' not in df.columns:
        numcols = sorted(set(df.columns) - set(IAMC_IDX))
        df = pd.melt(df, id_vars=IAMC_IDX, var_name='year',
                     value_vars=numcols, value_name='value')
    df.year = pd.to_numeric(df.year)

    # drop NaN's
    df.dropna(inplace=True)

    return df


def style_df(df, style='heatmap'):
    if style == 'highlight_not_max':
        return df.style.apply(lambda s: ['' if v else 'background-color: yellow' for v in s == s.max()])
    if style == 'heatmap':
        cm = sns.light_palette("green", as_cmap=True)
        return df.style.background_gradient(cmap=cm)


def pattern_match(data, strings, level=None):
    """
    matching of model/scenario names, variables, regions, and categories
    to pseudo-regex for filtering by columns (str)
    """
    matches = np.array([False] * len(data))

    if isinstance(strings, six.string_types):
        strings = [strings]

    for s in strings:
        regexp = (str(s)
                  .replace('|', '\\|')
                  .replace('*', '.*')
                  .replace('+', '\+')
                  ) + "$"
        pattern = re.compile(regexp)
        subset = filter(pattern.match, data)
        # check for depth by counting '|' after excluding the filter string
        if level is not None:
            pipe = re.compile('\\|')
            regexp = str(s).replace('*', '')
            depth = [len(pipe.findall(c.replace(regexp, ''))) <= level
                     for c in data]
            matches = matches | (data.isin(subset) & depth)
        else:
            matches = matches | data.isin(subset)

    return matches


def years_match(data, years):
    """
    matching of year columns for data filtering
    """
    if isinstance(years, int):
        return data == years
    elif isinstance(years, list) or isinstance(years, range):
        return data.isin(years)
    else:
        raise ValueError('filtering for years by ' + years + ' not supported,' +
                         'must be int, list or range')
