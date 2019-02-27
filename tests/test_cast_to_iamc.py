import pytest
import pandas as pd
from pyam import IamDataFrame, compare


def test_cast_from_value_col(meta_df):
    df_with_value_cols = pd.DataFrame([
        ['model_a', 'scen_a', 'World', 'EJ/y', 2005, 1, 0.5],
        ['model_a', 'scen_a', 'World', 'EJ/y', 2010, 6., 3],
        ['model_a', 'scen_b', 'World', 'EJ/y', 2005, 2, None],
        ['model_a', 'scen_b', 'World', 'EJ/y', 2010, 7, None]
    ],
        columns=['model', 'scenario', 'region', 'unit', 'year',
                 'Primary Energy', 'Primary Energy|Coal'],
    )
    df = IamDataFrame(df_with_value_cols,
                      value=['Primary Energy', 'Primary Energy|Coal'])

    assert compare(meta_df, df).empty
    pd.testing.assert_frame_equal(df.data, meta_df.data)


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
    df = pd.DataFrame([
        ['scen_a', 'World', 'Primary Energy', None, 'EJ/y', 1, 6.],
        ['scen_a', 'World', 'Primary Energy', 'Coal', 'EJ/y', 0.5, 3],
        ['scen_b', 'World', 'Primary Energy', None, 'EJ/y', 2, 7],
    ],
        columns=['scenario', 'region', 'var_1', 'var_2', 'unit', 2005, 2010],
    )

    df = IamDataFrame(df, model='model_a', variable=['var_1', 'var_2'])
    assert compare(meta_df, df).empty
    pd.testing.assert_frame_equal(df.data, meta_df.data)


def test_cast_with_variable_and_value(meta_df):
    pe_df = meta_df.filter(variable='Primary Energy')
    df = pe_df.data.rename(columns={'value': 'lvl'}).drop('variable', axis=1)

    df = IamDataFrame(df, variable='Primary Energy', value='lvl')

    assert compare(pe_df, df).empty
    pd.testing.assert_frame_equal(df.data, pe_df.data.reset_index(drop=True))
