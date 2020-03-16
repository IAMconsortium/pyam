import pytest

import pandas as pd
import pint
from pyam import IamDataFrame


PRECISE_ARG = dict(check_less_precise=True)


def get_units_test_df(test_df):
    # modify units in standard test dataframe
    df = test_df.copy()
    df.data.loc[0:1, 'unit'] = 'custom_unit'
    return df


def assert_converted_units(df, current, to, exp, **kwargs):
    # testing for `inplace=False` - converted values and expected unit
    _df = df.convert_unit(current, to, **kwargs, inplace=False)
    pd.testing.assert_series_equal(_df.data.value, exp, **PRECISE_ARG)
    assert _df.data.unit[5] == to

    # testing for `inplace=True` - converted values and expected unit
    df.convert_unit(current, to, **kwargs, inplace=True)
    pd.testing.assert_series_equal(df.data.value, exp, **PRECISE_ARG)
    assert df.data.unit[5] == to


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


def test_convert_unit_with_custom_registry(test_df):
    # unit conversion with custom UnitRegistry
    df = get_units_test_df(test_df).rename(unit={'EJ/yr': 'foo'})

    # check that conversion fails with application registry
    with pytest.raises(pint.UndefinedUnitError):
        df.convert_unit('foo', 'baz')

    # define a custom unit registry
    ureg = pint.UnitRegistry()
    ureg.define('baz = [custom]')
    ureg.define('foo =  3 * baz')

    exp = pd.Series([1., 6., 1.5, 9, 6, 21], name='value')
    assert_converted_units(df, 'foo', 'baz', exp, registry=ureg)


@pytest.mark.parametrize('unit', ['{}', 'Mt {}', 'Mt {} / yr', '{} / yr'])
def test_convert_unit_with_context(test_df, unit):
    # unit conversion with contexts in application registry
    df = test_df.copy()
    df['variable'] = [i.replace('Primary Energy', 'Emissions|CH4')
                      for i in df['variable']]
    current = unit.format('CH4')
    to = unit.format('CO2e')
    df['unit'] = current

    # assert that conversion fails without context
    with pytest.raises(pint.DimensionalityError):
        df.convert_unit(current, to)

    # test conversion for multiple contexts
    for (c, v) in [('AR5GWP100', 28), ('AR4GWP100', 25), ('SARGWP100', 21)]:
        exp = test_df.data.value * v
        assert_converted_units(df.copy(), current, to, exp, context=f'gwp_{c}')


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
