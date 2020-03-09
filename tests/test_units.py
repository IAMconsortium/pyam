import pytest

import pandas as pd
from pyam import IamDataFrame


PRECISE_ARG = dict(check_less_precise=True)


@pytest.mark.parametrize("current,to", [
    ('EJ', 'TWh'),
    ('EJ/yr', 'TWh/yr')
])
def test_convert_unit_with_pint(test_df, current, to):
    print(current, to)
    # unit conversion with standard pint
    df = test_df.copy()
    df.data.loc[0:1, 'unit'] = 'custom_unit'

    # replace EJ by EJ/yr to test pint with combined units
    if current == 'EJ/yr':
        df.rename(unit={'EJ': 'EJ/yr'}, inplace=True)

    exp = pd.Series([1., 6., 138.88, 833.33, 555.55, 1944.44], name='value')

    # testing for `inplace=False`
    _df = df.convert_unit(current, to, inplace=False)
    pd.testing.assert_series_equal(_df.data.value, exp, **PRECISE_ARG)

    # testing for `inplace=False`
    df.convert_unit(current, to, inplace=True)
    pd.testing.assert_series_equal(df.data.value, exp, **PRECISE_ARG)


def test_convert_unit_from_repo(test_df):
    # unit conversion with definition loaded from common units repo
    df = test_df.copy()
    df.data.loc[0:1, 'unit'] = 'custom_unit'

    exp = pd.Series([1., 6., 17.06, 102.361, 68.241, 238.843], name='value')

    # testing for `inplace=False`
    _df = df.convert_unit('EJ', 'Mtce', inplace=False)
    pd.testing.assert_series_equal(_df.data.value, exp, **PRECISE_ARG)

    # testing for `inplace=False`
    df.convert_unit('EJ', 'Mtce', inplace=True)
    pd.testing.assert_series_equal(df.data.value, exp, **PRECISE_ARG)


def test_convert_unit_with_custom_factor(test_df):
    # unit conversion with custom factor
    df = test_df.copy()
    df.data.loc[0:1, 'unit'] = 'custom_unit'

    exp = pd.Series([1., 6., 1., 6., 4., 14.], name='value')

    # testing for `inplace=False`
    _df = df.convert_unit('EJ', 'foo', factor=2, inplace=False)
    pd.testing.assert_series_equal(_df.data.value, exp)

    # testing for `inplace=False`
    df.convert_unit('EJ', 'foo', factor=2, inplace=True)
    pd.testing.assert_series_equal(df.data.value, exp)


def test_convert_unit_with_mapping():
    # TODO: deprecate in next release (>=0.6.0)
    df = IamDataFrame(pd.DataFrame([
        ['model', 'scen', 'SST', 'test_1', 'A', 1, 5],
        ['model', 'scen', 'SDN', 'test_2', 'unit', 2, 6],
        ['model', 'scen', 'SST', 'test_3', 'C', 3, 7],
    ], columns=['model', 'scenario', 'region',
                'variable', 'unit', 2005, 2010],
    ))

    unit_conv = {'A': ['B', 5], 'C': ['D', 3]}

    obs = df.convert_unit(unit_conv).data.reset_index(drop=True)

    exp = IamDataFrame(pd.DataFrame([
        ['model', 'scen', 'SST', 'test_1', 'B', 5, 25],
        ['model', 'scen', 'SDN', 'test_2', 'unit', 2, 6],
        ['model', 'scen', 'SST', 'test_3', 'D', 9, 21],
    ], columns=['model', 'scenario', 'region',
                'variable', 'unit', 2005, 2010],
    )).data.reset_index(drop=True)

    pd.testing.assert_frame_equal(obs, exp, check_index_type=False)
