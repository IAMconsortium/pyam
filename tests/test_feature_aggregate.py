import numpy as np
import pandas as pd
from pyam import check_aggregate, IamDataFrame, IAMC_IDX

from conftest import TEST_DTS


def test_missing_region(check_aggregate_df):
    # for now, this test makes sure that this operation works as expected
    exp = check_aggregate_df.aggregate_region('Primary Energy', region='foo')
    assert len(exp) == 8
    # # this test should be updated to the below after the return type of aggregate_region() is updated
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
        ['TEST', 'scen', 'China', 'Primary Energy', 'EJ/y', 1, 6],
        ['TEST', 'scen', 'Vietnam', 'Primary Energy', 'EJ/y', 0.75, 5]],
        columns=cols)
    df = IamDataFrame(data=data)
    obs = df.aggregate_region(variable='Primary Energy',
                              region='R5ASIA',
                              subregions=['China', 'Vietnam', 'Japan'],
                              components=[], append=False)
    assert len(obs) == 2


def test_aggregate_region_missing_all_subregions():
    cols = ['model', 'scenario', 'region', 'variable', 'unit', 2005, 2010]
    data = pd.DataFrame([
        ['TEST', 'scen', 'foo', 'Primary Energy', 'EJ/y', 1, 6],
        ['TEST', 'scen', 'bar', 'Primary Energy', 'EJ/y', 0.75, 5]],
        columns=cols)
    df = IamDataFrame(data=data)
    obs = df.aggregate_region(variable='Primary Energy',
                              region='R5ASIA',
                              subregions=['China', 'Vietnam', 'Japan']
                              )
    assert len(obs) == 0


def test_do_aggregate_append(meta_df):
    meta_df.rename({'variable': {'Primary Energy': 'Primary Energy|Gas'}},
                   inplace=True)
    meta_df.aggregate('Primary Energy', append=True)
    obs = meta_df.filter(variable='Primary Energy').timeseries()

    dts = TEST_DTS
    times = [2005, 2010] if "year" in meta_df.data else dts
    exp = pd.DataFrame([
        ['model_a', 'scen_a', 'World', 'Primary Energy', 'EJ/y', 1.5, 9.],
        ['model_a', 'scen_b', 'World', 'Primary Energy', 'EJ/y', 2, 7],
    ],
        columns=['model', 'scenario', 'region', 'variable', 'unit'] + times
    ).set_index(IAMC_IDX)
    if "year" in meta_df.data:
        exp.columns = list(map(int, exp.columns))
    else:
        exp.columns = pd.to_datetime(exp.columns)

    pd.testing.assert_frame_equal(obs, exp)


def test_check_aggregate_pass(check_aggregate_df):
    obs = check_aggregate_df.filter(
        scenario='a_scen'
    ).check_aggregate('Primary Energy')
    assert obs is None


def test_check_aggregate_fail(meta_df):
    obs = meta_df.check_aggregate('Primary Energy', exclude_on_fail=True)
    assert len(obs.columns) == 2
    assert obs.index.get_values()[0] == (
        'model_a', 'scen_a', 'World', 'Primary Energy', 'EJ/y'
    )


def test_check_aggregate_top_level(meta_df):
    obs = check_aggregate(meta_df, variable='Primary Energy', year=2005)
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
    obs = check_aggregate_df.check_aggregate_region('Primary Energy')
    assert obs is None

    for variable in check_aggregate_df.variables():
        obs = check_aggregate_df.check_aggregate_region(variable)
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
                variable,
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
        'Emissions|N2O', 'World', subregions=['REUROPE', 'RASIA']
    )
    assert obs is None

    obs = check_aggregate_regional_df.check_aggregate_region(
        'Emissions|N2O|Solvents', 'World', subregions=['REUROPE', 'RASIA']
    )
    assert obs is None

    obs = check_aggregate_regional_df.check_aggregate_region(
        'Emissions|N2O', 'REUROPE', subregions=['Germany', 'UK']
    )
    assert obs is None

    obs = check_aggregate_regional_df.check_aggregate_region(
        'Emissions|N2O|Transport', 'REUROPE', subregions=['Germany', 'UK']
    )
    assert obs is None
