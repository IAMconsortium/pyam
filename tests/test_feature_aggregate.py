import pytest
import logging

import numpy as np
import pandas as pd
from pyam import check_aggregate, IamDataFrame, IAMC_IDX
from pyam.testing import assert_iamframe_equal

from conftest import DTS_MAPPING

LONG_IDX = IAMC_IDX + ['year']

PE_MAX_DF = pd.DataFrame([
    ['model_a', 'scen_a', 'World', 'Primary Energy', 'EJ/yr', 2005, 9.0],
    ['model_a', 'scen_a', 'World', 'Primary Energy', 'EJ/yr', 2010, 10.0],
    ['model_a', 'scen_a', 'reg_a', 'Primary Energy', 'EJ/yr', 2005, 6.0],
    ['model_a', 'scen_a', 'reg_a', 'Primary Energy', 'EJ/yr', 2010, 6.0],
    ['model_a', 'scen_a', 'reg_b', 'Primary Energy', 'EJ/yr', 2005, 3.0],
    ['model_a', 'scen_a', 'reg_b', 'Primary Energy', 'EJ/yr', 2010, 4.0],

],
    columns=LONG_IDX + ['value']
)

CO2_MAX_DF = pd.DataFrame([
    ['model_a', 'scen_a', 'World', 'Emissions|CO2', 'EJ/yr', 2005, 6.0],
    ['model_a', 'scen_a', 'World', 'Emissions|CO2', 'EJ/yr', 2010, 8.0],
    ['model_a', 'scen_a', 'reg_a', 'Emissions|CO2', 'EJ/yr', 2005, 4.0],
    ['model_a', 'scen_a', 'reg_a', 'Emissions|CO2', 'EJ/yr', 2010, 5.0],
    ['model_a', 'scen_a', 'reg_b', 'Emissions|CO2', 'EJ/yr', 2005, 2.0],
    ['model_a', 'scen_a', 'reg_b', 'Emissions|CO2', 'EJ/yr', 2010, 3.0],
],
    columns=LONG_IDX + ['value']
)

PRICE_MAX_DF = pd.DataFrame([
    ['model_a', 'scen_a', 'World', 'Price|Carbon', 'USD/tCO2', 2005, 10.0],
    ['model_a', 'scen_a', 'World', 'Price|Carbon', 'USD/tCO2', 2010, 30.0],
],
    columns=LONG_IDX + ['value']
)


@pytest.mark.parametrize("variable,data", (
    ('Primary Energy', PE_MAX_DF),
    (['Primary Energy', 'Emissions|CO2'], pd.concat([PE_MAX_DF, CO2_MAX_DF])),
))
def test_aggregate(simple_df, variable, data):
    # check that `variable` is a a direct sum and matches given total
    exp = simple_df.filter(variable=variable)
    assert_iamframe_equal(simple_df.aggregate(variable), exp)

    # use other method (max) both as string and passing the function
    _df = data.copy()
    if simple_df.time_col == 'time':
        _df.year = _df.year.replace(DTS_MAPPING)
        _df.rename({'year': 'time'}, axis='columns', inplace=True)
    exp = IamDataFrame(_df)
    for m in ['max', np.max]:
        assert_iamframe_equal(simple_df.aggregate(variable, method=m), exp)


def test_check_aggregate(simple_df):
    # assert that `check_aggregate` returns None for full data
    assert simple_df.check_aggregate('Primary Energy') is None

    # assert that `check_aggregate` returns non-matching data
    obs = (
        simple_df
        .filter(variable='Primary Energy|Coal', region='World', keep=False)
        .check_aggregate('Primary Energy')
    )
    exp = pd.DataFrame([[12., 3.], [15., 5.]])
    np.testing.assert_array_equal(obs.values, exp.values)


def test_check_aggregate_top_level(simple_df):
    # assert that `check_aggregate` returns None for full data
    assert check_aggregate(simple_df, variable='Primary Energy',
                           year=2005) is None

    # duplicate scenario, assert `check_aggregate` returns non-matching data
    _df = (
        simple_df
        .rename(scenario={'scen_a': 'foo'}, append=True)
        .filter(scenario='foo', variable='Primary Energy|Coal', keep=False)
    )

    obs = check_aggregate(_df, variable='Primary Energy', year=2005,
                          exclude_on_fail=True)
    exp = pd.DataFrame([[12., 3.], [8., 2.], [4., 1.]])
    np.testing.assert_array_equal(obs.values, exp.values)

    # assert that scenario `foo` has correctly been assigned as `exclude=True`
    np.testing.assert_array_equal(_df.meta.exclude.values, [True, False])


@pytest.mark.parametrize("variable", (
    ('Primary Energy'),
    (['Primary Energy', 'Emissions|CO2']),
))
def test_aggregate_append(simple_df, variable):
    # remove `variable`, do aggregate and append, check equality to original
    _df = simple_df.filter(variable=variable, keep=False)
    _df.aggregate(variable, append=True)
    assert_iamframe_equal(_df, simple_df)


def test_aggregate_with_components(simple_df):
    # rename sub-category to test setting components explicitly as list
    df = simple_df.rename(variable={'Primary Energy|Wind': 'foo'})
    assert df.check_aggregate('Primary Energy') is not None
    components = ['Primary Energy|Coal', 'foo']
    assert df.check_aggregate('Primary Energy', components=components) is None


def test_aggregate_by_list_with_components_raises(simple_df):
    # using list of variables and components raises an error
    v = ['Primary Energy', 'Emissions|CO2']
    components = ['Primary Energy|Coal', 'Primary Energy|Wind']
    pytest.raises(ValueError, simple_df.aggregate, v, components=components)


def test_aggregate_empty(simple_df):
    assert simple_df.aggregate('foo') is None


def test_aggregate_unknown_method(simple_df):
    # using unknown string as method raises an error
    v = 'Primary Energy'
    pytest.raises(ValueError, simple_df.aggregate_region, v, method='foo')


@pytest.mark.parametrize("variable", (
    ('Primary Energy'),
    (['Primary Energy', 'Primary Energy|Coal', 'Primary Energy|Wind']),
))
def test_aggregate_region(simple_df, variable):
    # check that `variable` is a a direct sum across regions
    exp = simple_df.filter(variable=variable, region='World')
    assert_iamframe_equal(simple_df.aggregate_region(variable), exp)

    # check custom `region` (will include `World`, so double-count values)
    foo = exp.rename(region={'World': 'foo'})
    foo.data.value = foo.data.value * 2
    assert_iamframe_equal(simple_df.aggregate_region(variable, region='foo'),
                          foo)


def test_aggregate_region_log(simple_df, caplog):
    # verify that `check_aggregate_region()` writes log on empty assertion
    caplog.set_level(logging.INFO, logger="pyam._aggregate")
    simple_df.aggregate_region('foo')
    msg = ("cannot aggregate variable `foo` to `World` "
           "because it does not exist in any subregion")
    idx = caplog.messages.index(msg)
    assert caplog.records[idx].levelname == "INFO"


def test_check_aggregate_region(simple_df):
    # assert that `check_aggregate_region` returns None for full data
    assert simple_df.check_aggregate_region('Primary Energy') is None

    # assert that `check_aggregate_region` returns non-matching data
    obs = (
        simple_df
        .filter(variable='Primary Energy', region='reg_a', keep=False)
        .check_aggregate_region('Primary Energy')
    )
    exp = pd.DataFrame([[12., 4.], [15., 6.]])
    np.testing.assert_array_equal(obs.values, exp.values)


def test_check_aggregate_region_log(simple_df, caplog):
    # verify that `check_aggregate_region()` writes log on empty assertion
    caplog.set_level(logging.INFO, logger="pyam.core")
    (
        simple_df.filter(variable='Primary Energy', region='World', keep=False)
        .check_aggregate_region('Primary Energy')
    )
    print(caplog.messages)
    msg = "variable `Primary Energy` does not exist in region `World`"
    idx = caplog.messages.index(msg)
    assert caplog.records[idx].levelname == "INFO"


@pytest.mark.parametrize("variable", (
    ('Primary Energy'),
    (['Primary Energy', 'Primary Energy|Coal', 'Primary Energy|Wind']),
))
def test_aggregate_region_append(simple_df, variable):
    # remove `variable`, do aggregate and append, check equality to original
    _df = simple_df.filter(variable=variable, region='World', keep=False)
    _df.aggregate_region(variable, append=True)
    assert_iamframe_equal(_df, simple_df)


@pytest.mark.parametrize("variable", (
    ('Primary Energy'),
    (['Primary Energy', 'Primary Energy|Coal', 'Primary Energy|Wind']),
))
def test_aggregate_region_with_subregions(simple_df, variable):
    # check that custom `subregions` works (assumes only `reg_a` is in `World`)
    exp = (
        simple_df.filter(variable=variable, region='reg_a')
        .rename(region={'reg_a': 'World'})
    )
    obs = simple_df.aggregate_region(variable, subregions='reg_a')
    assert_iamframe_equal(obs, exp)

    # check that both custom `region` and `subregions` work
    foo = exp.rename(region={'World': 'foo'})
    obs = simple_df.aggregate_region(variable, region='foo',
                                     subregions='reg_a')
    assert_iamframe_equal(obs, foo)

    # check that invalid list of subregions returns empty
    assert simple_df.aggregate_region(variable, subregions=['reg_c']).empty


@pytest.mark.parametrize("variable,data", (
    ('Price|Carbon', PRICE_MAX_DF),
    (['Price|Carbon', 'Emissions|CO2'], pd.concat([PRICE_MAX_DF, CO2_MAX_DF]))
))
def test_aggregate_region_with_other_method(simple_df, variable, data):
    # use other method (max) both as string and passing the function
    _df = data.copy()
    if simple_df.time_col == 'time':
        _df.year = _df.year.replace(DTS_MAPPING)
        _df.rename({'year': 'time'}, axis='columns', inplace=True)
    exp = IamDataFrame(_df).filter(region='World')
    for m in ['max', np.max]:
        assert_iamframe_equal(simple_df.aggregate_region(variable, method=m),
                              exp)


def test_aggregate_region_with_components(simple_df):
    # CO2 emissions have "bunkers" only defined at the region level
    v = 'Emissions|CO2'
    assert simple_df.check_aggregate_region(v) is not None
    assert simple_df.check_aggregate_region(v, components=True) is None

    # rename emissions of bunker to test setting components as list
    _df = simple_df.rename(variable={'Emissions|CO2|Bunkers': 'foo'})
    assert _df.check_aggregate_region(v, components=['foo']) is None


def test_aggregate_region_with_weights(simple_df):
    # carbon price shouldn't be summed but be weighted by emissions
    v = 'Price|Carbon'
    w = 'Emissions|CO2'
    assert simple_df.check_aggregate_region(v) is not None
    assert simple_df.check_aggregate_region(v, weight=w) is None

    exp = simple_df.filter(variable=v, region='World')
    assert_iamframe_equal(simple_df.aggregate_region(v, weight=w), exp)

    # inconsistent index of variable and weight raises an error
    _df = simple_df.filter(variable=w, region='reg_b', keep=False)
    pytest.raises(ValueError, _df.aggregate_region, v, weight=w)

    # using weight and method other than 'sum' raises an error
    pytest.raises(ValueError, simple_df.aggregate_region, v, method='max',
                  weight='bar')


def test_aggregate_region_with_components_and_weights_raises(simple_df):
    # setting both weight and components raises an error
    pytest.raises(ValueError, simple_df.aggregate_region, 'Emissions|CO2',
                  components=True, weight='bar')


def test_aggregate_region_empty(simple_df):
    assert simple_df.aggregate_region('foo') is None


def test_aggregate_region_unknown_method(simple_df):
    # using unknown string as method raises an error
    v = 'Emissions|CO2'
    pytest.raises(ValueError, simple_df.aggregate_region, v, method='foo')


def test_check_internal_consistency(simple_df):
    _df = simple_df.filter(variable='Price|Carbon', keep=False)

    # assert that test data is consistent (except for `Price|Carbon`)
    assert _df.check_internal_consistency(components=True) is None

    # assert removing a specific subsector causes inconsistencies
    obs = (
        _df.filter(variable='Primary Energy|Coal', region='reg_a', keep=False)
        .check_internal_consistency(components=True)
    )

    # test reported inconsistency
    exp = pd.DataFrame([[np.nan, np.nan, 9., 3.], [np.nan, np.nan, 10., 4.],
                        [8., 2., np.nan, np.nan], [9., 3., np.nan, np.nan]])
    np.testing.assert_array_equal(obs.values, exp.values)
