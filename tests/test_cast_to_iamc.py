import pytest
import pandas as pd
from pyam import compare, df_to_pyam


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
    df = df_to_pyam(df_with_value_cols,
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
    pytest.raises(ValueError, df_to_pyam, df=df, model='foo')


def test_cast_with_model_arg(meta_df):
    df = meta_df.timeseries().reset_index()
    df.rename(columns={'model': 'foo'}, inplace=True)

    df = df_to_pyam(df, model='foo')
    assert compare(meta_df, df).empty
    pd.testing.assert_frame_equal(df.data, meta_df.data)
