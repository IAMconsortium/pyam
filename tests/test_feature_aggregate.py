import pytest
import logging

import numpy as np
import pandas as pd
from pyam import check_aggregate, IamDataFrame, IAMC_IDX

from conftest import TEST_DTS


LONG_IDX = IAMC_IDX + ['year']

PE_MAX_DF = pd.DataFrame([
    ['model_a', 'scen_a', 'World', 'Primary Energy', 'EJ/y', 2005, 7.0],
    ['model_a', 'scen_a', 'World', 'Primary Energy', 'EJ/y', 2010, 10.0],
    ['model_a', 'scen_a', 'reg_a', 'Primary Energy', 'EJ/y', 2005, 5.0],
    ['model_a', 'scen_a', 'reg_a', 'Primary Energy', 'EJ/y', 2010, 7.0],
    ['model_a', 'scen_a', 'reg_b', 'Primary Energy', 'EJ/y', 2005, 2.0],
    ['model_a', 'scen_a', 'reg_b', 'Primary Energy', 'EJ/y', 2010, 3.0],

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

REG_IDX = ['model', 'scenario', 'variable', 'unit', 'year']

PRICE_MAX_DF = pd.DataFrame([
    ['model_a', 'scen_a', 'Price|Carbon', 'USD/tCO2', 2005, 10.0],
    ['model_a', 'scen_a', 'Price|Carbon', 'USD/tCO2', 2010, 30.0],
],
    columns=REG_IDX + ['value']
)


def test_aggregate(aggregate_df):
    df = aggregate_df

    # primary energy is a direct sum (within each region)
    assert df.check_aggregate('Primary Energy') is None

    # rename sub-category to test setting components as list
    _df = df.rename(variable={'Primary Energy|Wind': 'foo'})
    assert _df.check_aggregate('Primary Energy') is not None
    components = ['Primary Energy|Coal', 'foo']
    assert _df.check_aggregate('Primary Energy', components=components) is None

    # use other method (max) both as string and passing the function
    exp = PE_MAX_DF.set_index(LONG_IDX).value

    obs = df.aggregate('Primary Energy', method='max')
    pd.testing.assert_series_equal(obs, exp)

    obs = df.aggregate('Primary Energy', method=np.max)
    pd.testing.assert_series_equal(obs, exp)

    # using illegal method raises an error
    pytest.raises(ValueError, df.aggregate, 'Primary Energy', method='foo')


def test_aggregate_by_list(aggregate_df):
    df = aggregate_df
    var_list = ['Primary Energy', 'Emissions|CO2']

    # primary energy and emissions are a direct sum (within each region)
    assert df.check_aggregate(var_list) is None

    # use other method (max) both as string and passing the function
    exp = (
        pd.concat([PE_MAX_DF, CO2_MAX_DF])
        .set_index(LONG_IDX).value
        .sort_index()
    )

    obs = df.aggregate(var_list, method='max')
    pd.testing.assert_series_equal(obs, exp)

    obs = df.aggregate(var_list, method=np.max)
    pd.testing.assert_series_equal(obs, exp)

    # using list of variables and components raises an error
    components = ['Primary Energy|Coal', 'Primary Energy|Wind']
    pytest.raises(ValueError, df.aggregate, var_list, components=components)


def test_aggregate_region(aggregate_df):
    df = aggregate_df

    # primary energy is a direct sum (across regions)
    assert df.check_aggregate_region('Primary Energy') is None

    # CO2 emissions have "bunkers" only defined at the region level
    v = 'Emissions|CO2'
    assert df.check_aggregate_region(v) is not None
    assert df.check_aggregate_region(v, components=True) is None

    # rename emissions of bunker to test setting components as list
    _df = df.rename(variable={'Emissions|CO2|Bunkers': 'foo'})
    assert _df.check_aggregate_region(v, components=['foo']) is None

    # carbon price shouldn't be summed but be weighted by emissions
    assert df.check_aggregate_region('Price|Carbon') is not None
    assert df.check_aggregate_region('Price|Carbon', weight=v) is None

    # inconsistent index of variable and weight raises an error
    _df = df.filter(variable='Emissions|CO2', region='reg_b', keep=False)
    pytest.raises(ValueError, _df.aggregate_region, 'Price|Carbon',
                  weight='Emissions|CO2')

    # setting both weight and components raises an error
    pytest.raises(ValueError, df.aggregate_region, v, components=True,
                  weight='bar')

    # use other method (max) both as string and passing the function
    exp = PRICE_MAX_DF.set_index(REG_IDX).value
    obs = df.aggregate_region('Price|Carbon', method='max')
    pd.testing.assert_series_equal(obs, exp)

    obs = df.aggregate_region('Price|Carbon', method=np.max)
    pd.testing.assert_series_equal(obs, exp)

    # using illegal method raises an error
    pytest.raises(ValueError, df.aggregate_region, v, method='foo')

    # using weight and method other than 'sum' raises an error
    pytest.raises(ValueError, df.aggregate_region, v, method='max',
                  weight='bar')


def test_aggregate_region_by_list(aggregate_df):
    df = aggregate_df
    var_list = ['Primary Energy', 'Primary Energy|Coal', 'Primary Energy|Wind']

    # primary energy and sub-categories are a direct sum (across regions)
    assert df.check_aggregate_region(var_list) is None

    # emissions and carbon price are _not_ a direct sum (across regions)
    var_list = ['Price|Carbon', 'Emissions|CO2']
    assert df.check_aggregate_region(var_list) is not None

    # using list of variables and components raises an error
    pytest.raises(ValueError, df.aggregate_region, var_list, components=True)

    # using list of variables and weight raises an error (inconsistent weight)
    pytest.raises(ValueError, df.aggregate_region, var_list, weight=True)

    # use other method (max) both as string and passing the function
    _co2_df = CO2_MAX_DF[CO2_MAX_DF.region=='World'].drop(columns='region')
    exp = pd.concat([_co2_df, PRICE_MAX_DF]).set_index(REG_IDX).value

    obs = df.aggregate_region(var_list, method='max')
    pd.testing.assert_series_equal(obs, exp)

    obs = df.aggregate_region(var_list, method=np.max)
    pd.testing.assert_series_equal(obs, exp)


def test_missing_region(check_aggregate_df):
    # for now, this test makes sure that this operation works as expected
    exp = check_aggregate_df.aggregate_region('Primary Energy', region='foo')
    assert len(exp) == 8
    # # this test should be updated to the below after the return type of
    # # aggregate_region() is updated
    # exp = check_aggregate_df.aggregate_region(
    #     'Primary Energy', region='foo', append=False
    # ).data
    # check_aggregate_df.aggregate_region(
    #     'Primary Energy', region='foo', append=True
    # )
    # obs = check_aggregate_df.filter(region='foo').data
    # assert len(exp) > 0
    # pd.testing.assert_frame_equal(obs.reset_index(drop=True),
    #                               exp.reset_index(drop=True))


def test_aggregate_region_extra_subregion():
    cols = ['model', 'scenario', 'region', 'variable', 'unit', 2005, 2010]
    data = pd.DataFrame([
        ['model_a', 'scen_a', 'foo', 'Primary Energy', 'EJ/y', 1, 6],
        ['model_a', 'scen_a', 'bar', 'Primary Energy', 'EJ/y', 0.75, 5]],
        columns=cols)
    df = IamDataFrame(data=data)
    obs = df.aggregate_region(variable='Primary Energy',
                              region='R5ASIA',
                              subregions=['foo', 'bar', 'baz'],
                              components=[], append=False)
    assert len(obs) == 2


def test_aggregate_region_missing_all_subregions():
    cols = ['model', 'scenario', 'region', 'variable', 'unit', 2005, 2010]
    data = pd.DataFrame([
        ['model_a', 'scen_a', 'foo', 'Primary Energy', 'EJ/y', 1, 6],
        ['model_a', 'scen_a', 'bar', 'Primary Energy', 'EJ/y', 0.75, 5]],
        columns=cols)
    df = IamDataFrame(data=data)
    obs = df.aggregate_region(variable='Primary Energy',
                              region='R5ASIA',
                              subregions=['China', 'Vietnam', 'Japan']
                              )
    assert len(obs) == 0


def test_do_aggregate_append(test_df):
    test_df.rename({'variable': {'Primary Energy': 'Primary Energy|Gas'}},
                   inplace=True)
    test_df.aggregate('Primary Energy', append=True)
    df = test_df.filter(variable='Primary Energy')

    times = [2005, 2010] if "year" in test_df.data else TEST_DTS
    exp = pd.DataFrame([
        ['model_a', 'scen_a', 'World', 'Primary Energy', 'EJ/y', 1.5, 9.],
        ['model_a', 'scen_b', 'World', 'Primary Energy', 'EJ/y', 2, 7],
    ],
        columns=['model', 'scenario', 'region', 'variable', 'unit'] + times
    ).set_index(IAMC_IDX)
    if "year" in test_df.data:
        exp.columns = list(map(int, exp.columns))
    else:
        df.data.time = df.data.time.dt.normalize()
        exp.columns = pd.to_datetime(exp.columns)

    pd.testing.assert_frame_equal(df.timeseries(), exp)


def test_aggregate_unknown_method(reg_df):
    pytest.raises(ValueError, reg_df.aggregate_region, 'Primary Energy',
                  method='foo')


def test_check_aggregate_pass(check_aggregate_df):
    obs = check_aggregate_df.filter(
        scenario='a_scen'
    ).check_aggregate('Primary Energy')
    assert obs is None


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


def test_check_aggregate_fail(test_df):
    obs = test_df.check_aggregate('Primary Energy', exclude_on_fail=True)
    assert len(obs.columns) == 2
    assert obs.index.get_values()[0] == (
        'model_a', 'scen_a', 'World', 'Primary Energy', 'EJ/y'
    )


def test_check_aggregate_top_level(test_df):
    obs = check_aggregate(test_df, variable='Primary Energy', year=2005)
    assert len(obs.columns) == 1
    assert obs.index.get_values()[0] == (
        'model_a', 'scen_a', 'World', 'Primary Energy', 'EJ/y'
    )


def test_df_check_aggregate_pass(check_aggregate_df):
    obs = check_aggregate_df.check_aggregate('Primary Energy')
    assert obs is None

    for variable in check_aggregate_df.variables():
        obs = check_aggregate_df.check_aggregate(variable)
        assert obs is None


def test_df_check_aggregate_region_pass(check_aggregate_df):
    comp = dict(components=True)
    obs = check_aggregate_df.check_aggregate_region('Primary Energy', **comp)
    assert obs is None

    for variable in check_aggregate_df.variables():
        obs = check_aggregate_df.check_aggregate_region(variable, **comp)
        assert obs is None


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

    pd.testing.assert_series_equal(res, exp)


def test_check_aggregate_region_no_world(check_aggregate_regional_df, caplog):
    test_df = check_aggregate_regional_df.filter(region='World', keep=False)
    caplog.set_level(logging.INFO, logger="pyam.core")
    test_df.check_aggregate_region('Emissions|N2O', region='World')
    warn_idx = caplog.messages.index("variable `Emissions|N2O` does not exist "
                                     "in region `World`")
    assert caplog.records[warn_idx].levelname == "INFO"
