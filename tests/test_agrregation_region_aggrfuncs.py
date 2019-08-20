import pyam
import pandas as pd
import numpy as np
import math
import pytest


all_regs = ['AFR', 'CPA', 'EEU', 'FSU', 'LAM']
r2_northregs = ['EEU', 'FSU']
r2_otherregs = ['AFR', 'CPA', 'LAM']

WAVG_DF = pd.DataFrame([
    ['model_a', 'scen_a', 'region_a', 'Primary Energy', 'EJ/y', 1, 6.],
    ['model_a', 'scen_a', 'region_b', 'Primary Energy', 'EJ/y', 2, 12.],
    ['model_a', 'scen_a', 'region_a', 'Price', 'EJ/y', 10, 5.],
    ['model_a', 'scen_a', 'region_b', 'Price', 'EJ/y', 4, 5.],
], columns=['model', 'scenario', 'region', 'variable', 'unit', 2005, 2010])


@pytest.fixture(scope="function")
def wavg_df():
    df = pyam.IamDataFrame(data=WAVG_DF)
    yield df


TWO_STAGE_DF = pd.DataFrame([
    ['MyModel', 'MyScen', 'AFR', 'FE', 'EJ/yr', 11.2, 16.8, 20.9],
    ['MyModel', 'MyScen', 'CPA', 'FE', 'EJ/yr', 47.9, 66.7, 84.1],
    ['MyModel', 'MyScen', 'EEU', 'FE', 'EJ/yr', 8.6, 8.6, 8.9],
    ['MyModel', 'MyScen', 'FSU', 'FE', 'EJ/yr', 24.8, 26, 29.5],
    ['MyModel', 'MyScen', 'LAM', 'FE', 'EJ/yr', 20.6, 24.3, 29.5],
    ['MyModel', 'MyScen', 'AFR', 'POP', 'million', 719, 811, 1021],
    ['MyModel', 'MyScen', 'CPA', 'POP', 'million', 1443, 1460, 1510],
    ['MyModel', 'MyScen', 'EEU', 'POP', 'million', 126, 125, 124],
    ['MyModel', 'MyScen', 'FSU', 'POP', 'million', 278, 279, 281],
    ['MyModel', 'MyScen', 'LAM', 'POP', 'million', 552, 584, 643],
    ['MyModel', 'MyScen', 'World', 'FE', 'EJ/yr', 113.1, 142.4, 172.9],
    ['MyModel', 'MyScen', 'World', 'POP', 'million', 3118, 3259, 3579],
], columns=['model', 'scenario', 'region', 'variable', 'unit',
            2005, 2010, 2020])


@pytest.fixture(scope="function")
def two_stage_df():
    df = pyam.IamDataFrame(data=TWO_STAGE_DF)
    yield df


def test_single_region_agg(reg_df):
    """
        aggregation over single region (no NaN values in data)
        is expected to just "copy" this regions values
        for aggregators min, max, avg, sum
    """
    for m in ['min', 'max', 'avg', 'sum']:
        df = reg_df.copy()  # pyam.IamDataFrame(reg_df.data)
        df.aggregate_region('Primary Energy', region='A1_MEA', method=m,
                            subregions=['MEA'], append=True)
        assert list(df.filter(region='A1_MEA').data.value) == \
            list(df.filter(region='MEA',
                           variable='Primary Energy').data.value)
        # pd.testing.assert_series_equal(
        #       df.filter(region='A1_MEA').data.value,
        #       df.filter(region='MEA',  variable='Primary Energy').data.value,
        #       check_names=False,
        #       check_categorical=False)
        np.testing.assert_array_equal(
                df.filter(region='A1_MEA').data.value,
                df.filter(region='MEA',  variable='Primary Energy').data.value)

def test_single_region_agg_nan_value(reg_df):
    """
        aggregation over single region (havind NaN values in data)
        is expected to just "copy" this regions values
        for aggregators min, max, avg the np.nan "data points" are not copied
        (sum behaves a bt different)
    """
    for m in ['min', 'max', 'avg']:
        df = reg_df.copy()
        df.data.loc[df.data['year'] == 2005, 'value'] = np.nan
        df.aggregate_region('Primary Energy', region='A1_MEA', method=m,
                            subregions=['MEA'], append=True)
        assert list(df.filter(region='A1_MEA').data.value) == \
            list(df.filter(region='MEA',
                           variable='Primary Energy').data.dropna()['value'])


def test_single_region_sum_nan_value(reg_df):
    """
        aggregation over single region (havind NaN values in data)
        is expected to just "copy" this regions values
        for aggregator sum. Please note that sum(NaN) = 0. so the np.nan
        "data point" ends up as 0. (zero)
    """
    for m in ['sum']:
        df = reg_df.copy()
        df.data.loc[df.data['year'] == 2010, 'value'] = np.nan
        df.aggregate_region('Primary Energy', region='A1_MEA', method=m,
                            subregions=['MEA'], append=True)
        assert list(df.filter(region='A1_MEA').data.value) == [1., 0.]


def test_weighted_average_region_basic(wavg_df):
    """
        weighted average basic test Price weighted by Priamry Energy
    """
    df = wavg_df.copy()
    df.weighted_average_region('Price', append=True, weight='Primary Energy')
    assert list(df.filter(region='World').data.value) == [6, 5]


def test_weighted_average_region_self_weight(wavg_df):
    """
        weighted average basic test Price weighted by Prise (i.e. itself)
    """
    df = wavg_df.copy()
    df.weighted_average_region('Price', append=True, weight='Price')
    assert list(df.filter(region='World').data.value) == [116. / 14., 5]


def test_aggregate_empty_subregions(reg_df, caplog):
    """
        agregate regions w/ empty subregion list is expected
        to deliver empty result and issue warning
    """
    for m in ['sum', 'min', 'max', 'avg']:
        df = reg_df.copy()
        caplog.clear()
        df.aggregate_region('Primary Energy', region='foo+bar',
                            subregions=['foo', 'bar'],
                            append=True, method=m)
        assert list(df.filter(region='foo+bar').data.value) == []
        assert 'Filtered IamDataFrame is empty!' in caplog.text


def test_aggregate_non_existing_subregion(wavg_df, caplog):
    """
        agregate region over non existent subregions is expected
        to deliver empty result and issue warning
    """
    for m in ['sum', 'min', 'max', 'avg']:
        df = wavg_df.copy()
        caplog.clear()
        df.aggregate_region('FE', region='foo',
                            subregions=['bar'], append=True, method=m)
        assert list(df.filter(region='foo').data.value) == []
        assert 'Filtered IamDataFrame is empty!' in caplog.text


def test_default_method_equals_sum(reg_df, caplog):
    """
        verify default aggregation method is 'sum' (np.sum)
    """
    df1 = reg_df.copy()
    df1.aggregate_region('Primary Energy', region='R1_world', append=True)
    df2 = reg_df.copy()
    df2.aggregate_region('Primary Energy', region='R1_world', method='sum',
                         append=True)
    # assert  default method yields same resualt as "sum"
    pd.testing.assert_frame_equal(df1.data, df2.data)
    # check sum works as expected
    np.testing.assert_allclose(
        list(df1.filter(region='R1_world').data.value), [6., 26.,  6., 26.])


def test_aggregate_region_two_stage_sum(two_stage_df, caplog):
    """
        sum in two stages
        All := sum(north) + sum (other)
    """
    df1 = two_stage_df.copy()
    df1.aggregate_region('FE', region='A1_All',
                         subregions=all_regs, append=True)
    df2 = two_stage_df.copy()
    df2.aggregate_region('FE', region='A2_North', method='sum',
                         subregions=r2_northregs, append=True)
    df2.aggregate_region('FE', region='A2_Other', method='sum',
                         subregions=r2_otherregs, append=True)
    df2.aggregate_region('FE', region='A1_All', method='sum',
                         subregions=['A2_North', 'A2_Other'], append=True)
    pd.testing.assert_frame_equal(df1.filter(region='A1_All').data,
                                  df2.filter(region='A1_All').data)
    np.testing.assert_allclose(
        list(df1.filter(region='A1_All').data.value), [113.1, 142.4, 172.9])


def test_aggregate_region_avg_uses_np_mean(two_stage_df, caplog):
    """
    'avg' is expected to yield same results as np.mean
    """
    df1 = two_stage_df.copy()
    df1.aggregate_region('FE', region='R1_world', method=np.mean,
                         subregions=all_regs, append=True)
    df2 = two_stage_df.copy()
    df2.aggregate_region('FE', region='R1_world', method='avg',
                         subregions=all_regs, append=True)
    pd.testing.assert_frame_equal(df1.data, df2.data)
    # assert averaging yields expected values
    np.testing.assert_allclose(
        list(df1.filter(region='R1_world').data.value), [22.62, 28.48, 34.58])


def test_weighted_average_region_with_var_nan(two_stage_df, caplog):
    """
    weighted_average_region: variable set to None or np.nan:
    as weight is non-zero a non issue
    """
    df = two_stage_df.copy()
    df.data.loc[df.data['year'] == 2005, 'value'] = np.nan # None
    df.weighted_average_region('FE', region='A1_CPA',
                               subregions=['CPA'],
                               append=True, weight='POP')
    exp_l = [v for v in list(df.filter(region='CPA',
             variable='FE').data.value) if not math.isnan(v)]
    obj_l = [v for v in list(df.filter(region='A1_CPA').data.value)
             if not math.isnan(v)]
    np.testing.assert_allclose(obj_l, exp_l)


def test_weighted_average_region_with_var_zero(two_stage_df, caplog):
    """
    weighted_average_region: variable set to zero:
    as weight is non-zero a non issue
    """
    df = two_stage_df.copy()
    df.data.loc[df.data['variable'] == 'FE', 'value'] = 0.
    df.weighted_average_region('FE', region='A1_CPA',
                               subregions=['CPA'],
                               append=True, weight='POP')
    exp_l = [v for v in list(df.filter(region='CPA',
             variable='FE').data.value) if not math.isnan(v)]
    obj_l = [v for v in list(df.filter(region='A1_CPA').data.value)
             if not math.isnan(v)]
    np.testing.assert_allclose(obj_l, exp_l)


@pytest.mark.xfail(reason="w.avg does not work with zero or None weight var")
def test__weighted_average_region_with_weight_zero(two_stage_df, caplog):
    """
    weighted_average_region: weight set to zero:
    is expected to fail
    """
    df = two_stage_df.copy()
    df.data.loc[df.data['variable'] == 'POP', 'value'] = 0.
    df.weighted_average_region('FE', region='A1_CPA',
                               subregions=['CPA'],
                               append=True, weight='POP')
    exp_l = [v for v in list(df.filter(region='CPA',
             variable='FE').data.value) if not math.isnan(v)]
    obj_l = [v for v in list(df.filter(region='A1_CPA').data.value)
             if not math.isnan(v)]
    np.testing.assert_allclose(obj_l, exp_l)
