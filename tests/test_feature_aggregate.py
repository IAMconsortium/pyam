import pytest
import logging

import numpy as np
import pandas as pd
from pyam import check_aggregate, IamDataFrame, IAMC_IDX


LONG_IDX = IAMC_IDX + ['year']

PE_MAX_DF = pd.DataFrame([
    ['model_a', 'scen_a', 'World', 'Primary Energy', 'EJ/y', 2005, 9.0],
    ['model_a', 'scen_a', 'World', 'Primary Energy', 'EJ/y', 2010, 10.0],
    ['model_a', 'scen_a', 'reg_a', 'Primary Energy', 'EJ/y', 2005, 6.0],
    ['model_a', 'scen_a', 'reg_a', 'Primary Energy', 'EJ/y', 2010, 6.0],
    ['model_a', 'scen_a', 'reg_b', 'Primary Energy', 'EJ/y', 2005, 3.0],
    ['model_a', 'scen_a', 'reg_b', 'Primary Energy', 'EJ/y', 2010, 4.0],

],
    columns=LONG_IDX + ['value']
)

CO2_MAX_DF = pd.DataFrame([
    ['model_a', 'scen_a', 'World', 'Emissions|CO2', 'EJ/y', 2005, 6.0],
    ['model_a', 'scen_a', 'World', 'Emissions|CO2', 'EJ/y', 2010, 8.0],
    ['model_a', 'scen_a', 'reg_a', 'Emissions|CO2', 'EJ/y', 2005, 4.0],
    ['model_a', 'scen_a', 'reg_a', 'Emissions|CO2', 'EJ/y', 2010, 5.0],
    ['model_a', 'scen_a', 'reg_b', 'Emissions|CO2', 'EJ/y', 2005, 2.0],
    ['model_a', 'scen_a', 'reg_b', 'Emissions|CO2', 'EJ/y', 2010, 3.0],
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
    assert simple_df.aggregate(variable).equals(exp)

    # use other method (max) both as string and passing the function
    exp = IamDataFrame(data)
    assert simple_df.aggregate(variable, method='max').equals(exp)
    assert simple_df.aggregate(variable, method=np.max).equals(exp)


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
    assert _df.equals(simple_df)


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
    assert simple_df.aggregate_region(variable).equals(exp)

    # check custom `region` (will include `World`, so double-count values)
    exp_foo = exp.rename(region={'World': 'foo'})
    exp_foo.data.value = exp_foo.data.value * 2
    assert simple_df.aggregate_region(variable, region='foo').equals(exp_foo)


def test_aggregate_region_log(simple_df, caplog):
    # verify that `check_aggregate_region()` writes log on empty assertion
    caplog.set_level(logging.INFO, logger="pyam.aggregate")
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
    assert _df.equals(simple_df)


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
    assert simple_df.aggregate_region(variable, subregions='reg_a').equals(exp)

    # check that both custom `region` and `subregions` work
    exp_foo = exp.rename(region={'World': 'foo'})
    assert simple_df.aggregate_region(variable, region='foo',
                                      subregions='reg_a').equals(exp_foo)

    # check that invalid list of subregions returns empty
    assert simple_df.aggregate_region(variable, subregions=['reg_c']).empty


@pytest.mark.parametrize("variable,data", (
        ('Price|Carbon', PRICE_MAX_DF),
        (['Price|Carbon', 'Emissions|CO2'],
         pd.concat([PRICE_MAX_DF, CO2_MAX_DF]))
))
def test_aggregate_region_with_other_method(simple_df, variable, data):
    # use other method (max) both as string and passing the function
    exp = IamDataFrame(data).filter(region='World')
    assert simple_df.aggregate_region(variable, method='max').equals(exp)
    assert simple_df.aggregate_region(variable, method=np.max).equals(exp)


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
    assert simple_df.aggregate_region(v, weight=w).equals(exp)

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
    pytest.raises(ValueError, simple_df.aggregate_region, v,  method='foo')


def test_check_internal_consistency_no_world_for_variable(
    check_aggregate_df, caplog
):
    assert check_aggregate_df.check_internal_consistency() is None
    test_df = check_aggregate_df.filter(
        variable='Emissions|CH4', region='World', keep=False
    )
    caplog.set_level(logging.INFO, logger="pyam.core")
    test_df.check_internal_consistency()
    warn_idx = caplog.messages.index("variable `Emissions|CH4` does not exist "
                                     "in region `World`")
    assert caplog.records[warn_idx].levelname == "INFO"


def run_check_agg_fail(pyam_df, tweak_dict, test_type):
    mr = pyam_df.data.model == tweak_dict['model']
    sr = pyam_df.data.scenario == tweak_dict['scenario']
    rr = pyam_df.data.region == tweak_dict['region']
    vr = pyam_df.data.variable == tweak_dict['variable']
    ur = pyam_df.data.unit == tweak_dict['unit']

    row_to_tweak = mr & sr & rr & vr & ur
    assert row_to_tweak.any()

    pyam_df.data.value.iloc[np.where(row_to_tweak)[0]] *= 0.99

    # the error variable is always the top level one
    expected_index = tweak_dict
    agg_test = test_type == 'aggregate'
    region_world_only_contrib = test_type == 'region-world-only-contrib'
    if agg_test or region_world_only_contrib:
        expected_index['variable'] = '|'.join(
            expected_index['variable'].split('|')[:2]
        )
    elif 'region' in test_type:
        expected_index['region'] = 'World'

    expected_index = [v for k, v in expected_index.items()]

    for variable in pyam_df.variables():
        if test_type == 'aggregate':
            obs = pyam_df.check_aggregate(
                variable,
            )
        elif 'region' in test_type:
            obs = pyam_df.check_aggregate_region(
                variable, components=True
            )

        if obs is not None:
            assert len(obs.columns) == 2
            assert set(obs.index.get_values()[0]) == set(expected_index)


def test_df_check_aggregate_fail(check_aggregate_df):
    to_tweak = {
        'model': 'IMG',
        'scenario': 'a_scen_2',
        'region': 'R5REF',
        'variable': 'Emissions|CO2',
        'unit': 'Mt CO2/yr',
    }
    run_check_agg_fail(check_aggregate_df, to_tweak, 'aggregate')


def test_df_check_aggregate_fail_no_regions(check_aggregate_df):
    to_tweak = {
        'model': 'MSG-GLB',
        'scenario': 'a_scen_2',
        'region': 'World',
        'variable': 'Emissions|C2F6|Solvents',
        'unit': 'kt C2F6/yr',
    }
    run_check_agg_fail(check_aggregate_df, to_tweak, 'aggregate')


def test_df_check_aggregate_region_fail(check_aggregate_df):
    to_tweak = {
        'model': 'IMG',
        'scenario': 'a_scen_2',
        'region': 'World',
        'variable': 'Emissions|CO2',
        'unit': 'Mt CO2/yr',
    }
    run_check_agg_fail(check_aggregate_df, to_tweak, 'region')


def test_df_check_aggregate_region_fail_no_subsector(check_aggregate_df):
    to_tweak = {
        'model': 'MSG-GLB',
        'scenario': 'a_scen_2',
        'region': 'R5REF',
        'variable': 'Emissions|CH4',
        'unit': 'Mt CH4/yr',
    }
    run_check_agg_fail(check_aggregate_df, to_tweak, 'region')


def test_df_check_aggregate_region_fail_world_only_var(check_aggregate_df):
    to_tweak = {
        'model': 'MSG-GLB',
        'scenario': 'a_scen_2',
        'region': 'World',
        'variable': 'Emissions|CO2|Agg Agg',
        'unit': 'Mt CO2/yr',
    }

    run_check_agg_fail(
        check_aggregate_df, to_tweak, 'region-world-only-contrib'
    )


def test_df_check_aggregate_region_errors(check_aggregate_regional_df):
    # these tests should fail because our dataframe has continents and regions
    # so checking without providing components leads to double counting and
    # hence failure
    obs = check_aggregate_regional_df.check_aggregate_region(
        'Emissions|N2O', 'World'
    )

    assert len(obs.columns) == 2
    assert obs.index.get_values()[0] == (
        'AIM', 'cscen', 'World', 'Emissions|N2O', 'Mt N/yr'
    )

    obs = check_aggregate_regional_df.check_aggregate_region(
        'Emissions|N2O', 'REUROPE'
    )

    assert len(obs.columns) == 2
    assert obs.index.get_values()[0] == (
        'AIM', 'cscen', 'REUROPE', 'Emissions|N2O', 'Mt N/yr'
    )


def test_df_check_aggregate_region_components(check_aggregate_regional_df):
    obs = check_aggregate_regional_df.check_aggregate_region(
        'Emissions|N2O', 'World', subregions=['REUROPE', 'RASIA'],
        components=True
    )
    assert obs is None

    obs = check_aggregate_regional_df.check_aggregate_region(
        'Emissions|N2O|Ind|Solvents', 'World', subregions=['REUROPE', 'RASIA'],
        components=True
    )
    assert obs is None

    obs = check_aggregate_regional_df.check_aggregate_region(
        'Emissions|N2O', 'REUROPE', subregions=['Germany', 'UK'],
        components=True
    )
    assert obs is None

    obs = check_aggregate_regional_df.check_aggregate_region(
        'Emissions|N2O', 'RASIA', subregions=['China', 'Japan'],
        components=True
    )
    assert obs is None

    obs = check_aggregate_regional_df.check_aggregate_region(
        'Emissions|N2O|Ind|Transport', 'REUROPE', subregions=['Germany', 'UK'],
        components=True
    )
    assert obs is None


@pytest.mark.parametrize("components,exp_vals", (
    # should find sub-components including nested bunkers
    (True, [1.9, 15.7]),
    # should only add AFOLU onto regional sum, not Shipping emissions
    (["Emissions|N2O|AFOLU"], [0.9, 9.7]),
    # specifying Ind leads to double counting (and not skipping AFOLU) but as
    # it's user specified no warning etc. is raised
    (["Emissions|N2O|Ind"], [2.6, 25.2]),
))
def test_aggregate_region_components_handling(check_aggregate_regional_df,
                                              components, exp_vals):
    tdf = check_aggregate_regional_df.filter(variable="*N2O*")
    # only get Europe and Asia to avoid double counting
    res = tdf.aggregate_region("Emissions|N2O", components=components,
                               subregions=["REUROPE", "RASIA"])
    exp_idx = pd.MultiIndex.from_product(
        [["AIM"], ["cscen"], ['Emissions|N2O'], ["Mt N/yr"], [2005, 2010]],
        names=["model", "scenario", "variable", "unit", "year"]
    )
    exp = pd.Series(exp_vals, index=exp_idx)
    exp.name = "value"

    res.equals(IamDataFrame(exp, region='World'))