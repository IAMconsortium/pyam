import pytest

import pandas as pd
from pyam import IamDataFrame


PRECISE_ARG = dict(check_less_precise=True)


def get_units_test_df(test_df):
    # modify units in standard test dataframe
    df = test_df.copy()
    df.data.loc[0:1, 'unit'] = 'custom_unit'
    return df


def assert_converted_units(df, current, to, exp, **kwargs):
    # testing for `inplace=False`
    _df = df.convert_unit(current, to, **kwargs, inplace=False)
    pd.testing.assert_series_equal(_df.data.value, exp, **PRECISE_ARG)

    # testing for `inplace=True`
    df.convert_unit(current, to, **kwargs, inplace=True)
    pd.testing.assert_series_equal(df.data.value, exp, **PRECISE_ARG)


@pytest.mark.parametrize("current,to", [
    ('EJ/yr', 'TWh/yr'),
    ('EJ', 'TWh')
])
def test_convert_unit_with_pint(test_df, current, to):
    # unit conversion with default UnitRegistry (i.e, application_registry)
    df = get_units_test_df(test_df)

    # replace EJ/yr by EJ to test pint with single unit
    if current == 'EJ':
        df.rename(unit={'EJ/yr': 'EJ'}, inplace=True)

    exp = pd.Series([1., 6., 138.88, 833.33, 555.55, 1944.44], name='value')
    assert_converted_units(df, current, to, exp)


def test_convert_unit_from_repo(test_df):
    # unit conversion with definition loaded from `IAMconsortium/units` repo
    df = get_units_test_df(test_df)
    exp = pd.Series([1., 6., 17.06, 102.361, 68.241, 238.843], name='value')
    assert_converted_units(df, 'EJ/yr', 'Mtce/yr', exp)


def test_convert_unit_with_custom_factor(test_df):
    # unit conversion with custom factor
    df = get_units_test_df(test_df)
    exp = pd.Series([1., 6., 1., 6., 4., 14.], name='value')
    assert_converted_units(df, 'EJ/yr', 'foo', exp, factor=2)


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
