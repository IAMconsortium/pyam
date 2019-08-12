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


TEST2_DF = pd.DataFrame([
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


# for record in caplog.records:
#    # assert record.levelname != 'CRITICAL'
#    assert record.levelname == 'INFO'  # level == logging.level.ERROR
# assert 'cannot aggregate variable' in caplog.text
# assert ' because it does not exist in any subregion' in caplog.text
# assert caplog.records[0].levelname == 'ERROR' # actually INFO
# assert len(idf.data.columns) == 7
# pd.testing.assert_frame_equal(TEST_DF, df.data)
# @pytest.mark.xfail(reason="xxx")


def test_single_region_agg(reg_df):
    for m in ['min', 'max', 'avg', 'sum']:
        my_df = pyam.IamDataFrame(reg_df.data)
        my_df.aggregate_region('Primary Energy', region='A1_MEA', method=m,
                               subregions=['MEA'], append=True)
        np.testing.assert_allclose(
            list(my_df.filter(region='A1_MEA').data.value),
            list(my_df.filter(region='MEA',
                              variable='Primary Energy').data.value))
        assert list(my_df.filter(region='A1_MEA').data.value) == \
            list(my_df.filter(region='MEA',
                              variable='Primary Energy').data.value)


def test_single_region_agg_nan_value(reg_df):
    for m in ['min', 'max', 'avg']:
        my_df = pyam.IamDataFrame(reg_df.data)
        my_df.data.loc[my_df.data['year'] == 2005, 'value'] = np.nan
        my_df.aggregate_region('Primary Energy', region='A1_MEA', method=m,
                               subregions=['MEA'], append=True)
        assert list(my_df.filter(region='A1_MEA').data.value) == \
            list(my_df.filter(region='MEA',
                              variable='Primary Energy').data.dropna()[
                'value'])


def test_single_region_sum_nan_value(reg_df):
    for m in ['sum']:
        my_df = pyam.IamDataFrame(reg_df.data)
        my_df.data.loc[my_df.data['year'] == 2010, 'value'] = np.nan
        my_df.aggregate_region('Primary Energy', region='A1_MEA', method=m,
                               subregions=['MEA'], append=True)
        assert list(my_df.filter(region='A1_MEA').data.value) == [1., 0.]


def test_weighted_average_region_basic(wavg_df):
    wavg_df.weighted_average_region('Price', append=True,
                                    weight='Primary Energy')
    assert list(wavg_df.filter(region='World').data.value) == [6, 5]


def test_weighted_average_region_self_weight(wavg_df):
    wavg_df.weighted_average_region('Price', append=True, weight='Price')
    assert list(wavg_df.filter(region='World').data.value) == [116. / 14., 5]


def test_aggregate_empty_subregions_3(wavg_df, caplog):
    caplog.clear()
    wavg_df.aggregate_region('FE', region='foo',
                             subregions=[], append=True, method='avg')
    assert list(wavg_df.filter(region='foo').data.value) == []
    assert 'cannot aggregate variable' in caplog.text


def test_aggregate_empty_subregions_4(wavg_df, caplog):
    for m in ['sum', 'min', 'max', 'avg']:
        caplog.clear()
        wavg_df.aggregate_region('FE', region='foo',
                                 subregions=['bar'], append=True, method=m)
        assert list(wavg_df.filter(region='foo').data.value) == []
        assert 'Filtered IamDataFrame is empty!' in caplog.text


def test_aggregate_empty_subregions_5(wavg_df, caplog):
    for m in ['sum', 'min', 'max', 'avg']:
        caplog.clear()
        wavg_df.aggregate_region('foo', region='foo',
                                 subregions=[], append=True, method=m)
        assert list(wavg_df.filter(region='foo').data.value) == []
        assert 'Filtered IamDataFrame is empty!' in caplog.text


# @pytest.mark.xfail(reason="xxx")
def test_aggregate_empty_subregions_2(reg_df, caplog):
    for m in ['sum', 'min', 'max', 'avg']:
        caplog.clear()
        reg_df.aggregate_region('Primary Energy', region='foo+bar',
                                subregions=['foo', 'bar'],
                                append=True, method=m)
        assert list(reg_df.filter(region='foo+bar').data.value) == []
        assert 'Filtered IamDataFrame is empty!' in caplog.text


def test_df2_sum(caplog):
    df1 = pyam.IamDataFrame(TEST2_DF)
    df1.aggregate_region('FE', region='R1_world',
                         subregions=all_regs, append=True)
    df2 = pyam.IamDataFrame(TEST2_DF)
    df2.aggregate_region('FE', region='R1_world', method='sum',
                         subregions=all_regs, append=True)
    pd.testing.assert_frame_equal(df1.data, df2.data)
    np.testing.assert_allclose(
        list(df1.filter(region='R1_world').data.value), [113.1, 142.4, 172.9])


def test_df2_two_stage_sum(caplog):
    """
        sum in two stages
        All := sum(north) + sum (othet)
    """
    r2_northregs = ['EEU', 'FSU']
    r2_otherregs = ['AFR', 'CPA', 'LAM']

    df1 = pyam.IamDataFrame(TEST2_DF)
    df1.aggregate_region('FE', region='A1_All',
                         subregions=all_regs, append=True)
    df2 = pyam.IamDataFrame(TEST2_DF)
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


def test_df2_avg(caplog):
    df1 = pyam.IamDataFrame(TEST2_DF)
    df1.aggregate_region('FE', region='R1_world', method=np.mean,
                         subregions=all_regs, append=True)
    df2 = pyam.IamDataFrame(TEST2_DF)
    df2.aggregate_region('FE', region='R1_world', method='avg',
                         subregions=all_regs, append=True)
    pd.testing.assert_frame_equal(df1.data, df2.data)
    np.testing.assert_allclose(
        list(df1.filter(region='R1_world').data.value), [22.62, 28.48, 34.58])


def test_df2_single_region_w_agg_with_None(caplog):
    df = pyam.IamDataFrame(TEST2_DF)
    df.data.loc[df.data['year'] == 2005, 'value'] = None
    df.weighted_average_region('FE', region='A1_CPA',
                               subregions=['CPA'],
                               append=True, weight='POP')
    exp_l = [v for v in list(df.filter(region='CPA',
             variable='FE').data.value) if not math.isnan(v)]
    obj_l = [v for v in list(df.filter(region='A1_CPA').data.value)
             if not math.isnan(v)]
    np.testing.assert_allclose(obj_l, exp_l)


def test_df2_single_region_w_agg_with_var_zero(caplog):
    df = pyam.IamDataFrame(TEST2_DF)
    df.data.loc[df.data['variable'] == 'FE', 'value'] = 0.  # None
    df.weighted_average_region('FE', region='A1_CPA',
                               subregions=['CPA'],
                               append=True, weight='POP')
    exp_l = [v for v in list(df.filter(region='CPA',
             variable='FE').data.value) if not math.isnan(v)]
    obj_l = [v for v in list(df.filter(region='A1_CPA').data.value)
             if not math.isnan(v)]
    np.testing.assert_allclose(obj_l, exp_l)


# @pytest.mark.skip(reason="w.avg does not work with zero or None values")
@pytest.mark.xfail(reason="w.avg does not work with zero or None values")
def test_df2_single_region_w_agg_with_weight_zero(caplog):
    df = pyam.IamDataFrame(TEST2_DF)
    df.data.loc[df.data['variable'] == 'POP', 'value'] = 0.  # None
    df.weighted_average_region('FE', region='A1_CPA',
                               subregions=['CPA'],
                               append=True, weight='POP')
    exp_l = [v for v in list(df.filter(region='CPA',
             variable='FE').data.value) if not math.isnan(v)]
    obj_l = [v for v in list(df.filter(region='A1_CPA').data.value)
             if not math.isnan(v)]
    np.testing.assert_allclose(obj_l, exp_l)


def test_df2_single_region_agg_with_None(caplog):
    """ 'sum' failes as sum(None) := 0 """
    for m in ['min', 'max', 'avg']:  # 'sum',
        df = pyam.IamDataFrame(TEST2_DF)
        # df.data.loc[df.data['region']='CPA', 2005] = None
        df.data.loc[df.data['year'] == 2005, 'value'] = np.nan
        df.aggregate_region('FE', region='A1_CPA', method=m,
                            subregions=['CPA'], append=True)
        exp_l = [v for v in list(df.filter(region='CPA',
                 variable='FE').data.value) if not math.isnan(v)]
        obj_l = [v for v in list(df.filter(region='A1_CPA').data.value)
                 if not math.isnan(v)]
        np.testing.assert_allclose(obj_l, exp_l)
        # assert list(df.filter(region='A1_CPA').data.value) == \
        #    list(df.filter(region='CPA',
        #                   variable='FE').data.value)
