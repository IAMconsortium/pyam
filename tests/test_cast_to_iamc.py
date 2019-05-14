import datetime as dt


import pytest
import pandas as pd
from pyam import IamDataFrame, compare


def test_cast_from_value_col(meta_df):
    df_with_value_cols = pd.DataFrame([
        ['model_a', 'scen_a', 'World', 'EJ/y', dt.datetime(2005, 6, 17), 1, 0.5],
        ['model_a', 'scen_a', 'World', 'EJ/y', dt.datetime(2010, 7, 21), 6., 3],
        ['model_a', 'scen_b', 'World', 'EJ/y', dt.datetime(2005, 6, 17), 2, None],
        ['model_a', 'scen_b', 'World', 'EJ/y', dt.datetime(2010, 7, 21), 7, None]
    ],
        columns=['model', 'scenario', 'region', 'unit', 'time',
                 'Primary Energy', 'Primary Energy|Coal'],
    )
    df = IamDataFrame(df_with_value_cols,
                      value=['Primary Energy', 'Primary Energy|Coal'])
    if "year" in meta_df.data.columns:
        df = df.swap_time_for_year()

    assert compare(meta_df, df).empty
    pd.testing.assert_frame_equal(df.data, meta_df.data, check_like=True)


def test_cast_from_value_col_and_args(meta_df):
    # checks for issue [#210](https://github.com/IAMconsortium/pyam/issues/210)
    df_with_value_cols = pd.DataFrame([
        ['scen_a', 'World', 'EJ/y', dt.datetime(2005, 6, 17), 1, 0.5],
        ['scen_a', 'World', 'EJ/y', dt.datetime(2010, 7, 21), 6., 3],
        ['scen_b', 'World', 'EJ/y', dt.datetime(2005, 6, 17), 2, None],
        ['scen_b', 'World', 'EJ/y', dt.datetime(2010, 7, 21), 7, None]
    ],
        columns=['scenario', 'iso', 'unit', 'time',
                 'Primary Energy', 'Primary Energy|Coal'],
    )
    df = IamDataFrame(df_with_value_cols, model='model_a', region='iso',
                      value=['Primary Energy', 'Primary Energy|Coal'])
    if "year" in meta_df.data.columns:
        df = df.swap_time_for_year()

    assert compare(meta_df, df).empty
    pd.testing.assert_frame_equal(df.data, meta_df.data, check_like=True)


def test_cast_with_model_arg_raises():
    df = pd.DataFrame([
        ['model_a', 'scen_a', 'World', 'EJ/y', 2005, 1, 0.5],
    ],
        columns=['model', 'scenario', 'region', 'unit', 'year',
                 'Primary Energy', 'Primary Energy|Coal'],
    )
    pytest.raises(ValueError, IamDataFrame, df, model='foo')


def test_cast_with_model_arg(meta_df):
    df = meta_df.timeseries().reset_index()
    df.rename(columns={'model': 'foo'}, inplace=True)

    df = IamDataFrame(df, model='foo')
    assert compare(meta_df, df).empty
    pd.testing.assert_frame_equal(df.data, meta_df.data)


def test_cast_by_column_concat(meta_df):
    dts = [dt.datetime(2005, 6, 17), dt.datetime(2010, 7, 21)]
    df = pd.DataFrame([
        ['scen_a', 'World', 'Primary Energy', None, 'EJ/y', 1, 6.],
        ['scen_a', 'World', 'Primary Energy', 'Coal', 'EJ/y', 0.5, 3],
        ['scen_b', 'World', 'Primary Energy', None, 'EJ/y', 2, 7],
    ],
        columns=['scenario', 'region', 'var_1', 'var_2', 'unit'] + dts,
    )

    df = IamDataFrame(df, model='model_a', variable=['var_1', 'var_2'])
    if "year" in meta_df.data.columns:
        df = df.swap_time_for_year()

    assert compare(meta_df, df).empty
    pd.testing.assert_frame_equal(df.data, meta_df.data, check_like=True)


def test_cast_with_variable_and_value(meta_df):
    pe_df = meta_df.filter(variable='Primary Energy')
    df = pe_df.data.rename(columns={'value': 'lvl'}).drop('variable', axis=1)

    df = IamDataFrame(df, variable='Primary Energy', value='lvl')

    assert compare(pe_df, df).empty
    pd.testing.assert_frame_equal(df.data, pe_df.data.reset_index(drop=True))


def test_cast_from_r_df(test_pd_df):
    df = test_pd_df.copy()
    # last two columns are years
    df.columns = list(df.columns[:-2]) + ['X{}'.format(c)
                                          for c in df.columns[-2:]]
    obs = IamDataFrame(df)
    exp = IamDataFrame(test_pd_df)
    assert compare(obs, exp).empty
    pd.testing.assert_frame_equal(obs.data, exp.data)


def test_cast_from_r_df_err(test_pd_df):
    df = test_pd_df.copy()
    # last two columns are years
    df.columns = list(df.columns[:-2]) + ['Xfoo', 'Xbar']
    pytest.raises(ValueError, IamDataFrame, df)
