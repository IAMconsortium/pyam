import copy
import pytest

import numpy as np
import pandas as pd
from numpy import testing as npt


from pyam import IamDataFrame, META_IDX, IAMC_IDX


def test_append_other_scenario(meta_df):
    other = meta_df.filter(scenario='a_scenario2')\
        .rename({'scenario': {'a_scenario2': 'a_scenario3'}})

    meta_df.set_meta([0, 1], name='col1')
    meta_df.set_meta(['a', 'b'], name='col2')

    other.set_meta(2, name='col1')
    other.set_meta('x', name='col3')

    df = meta_df.append(other)

    # check that the original meta dataframe is not updated
    obs = meta_df.meta.index.get_level_values(1)
    npt.assert_array_equal(obs, ['a_scenario', 'a_scenario2'])

    # assert that merging of meta works as expected
    exp = pd.DataFrame([
        ['a_model', 'a_scenario', False, 0, 'a', np.nan],
        ['a_model', 'a_scenario2', False, 1, 'b', np.nan],
        ['a_model', 'a_scenario3', False, 2, np.nan, 'x'],
    ], columns=['model', 'scenario', 'exclude', 'col1', 'col2', 'col3']
    ).set_index(['model', 'scenario'])

    # sort columns for assertion in older pandas versions
    df.meta = df.meta.reindex(columns=exp.columns)
    pd.testing.assert_frame_equal(df.meta, exp)

    # assert that appending data works as expected
    ts = df.timeseries()
    npt.assert_array_equal(ts.iloc[2].values, ts.iloc[3].values)


def test_append_same_scenario(meta_df):
    other = meta_df.filter(scenario='a_scenario2')\
        .rename({'variable': {'Primary Energy': 'Primary Energy clone'}})

    meta_df.set_meta([0, 1], name='col1')

    other.set_meta(2, name='col1')
    other.set_meta('b', name='col2')

    # check that non-matching meta raise an error
    pytest.raises(ValueError, meta_df.append, other=other)

    # check that ignoring meta conflict works as expetced
    df = meta_df.append(other, ignore_meta_conflict=True)

    # check that the new meta.index is updated, but not the original one
    npt.assert_array_equal(meta_df.meta.columns, ['exclude', 'col1'])

    # assert that merging of meta works as expected
    exp = meta_df.meta.copy()
    exp['col2'] = [np.nan, 'b']
    pd.testing.assert_frame_equal(df.meta, exp)

    # assert that appending data works as expected
    ts = df.timeseries()
    npt.assert_array_equal(ts.iloc[2], ts.iloc[3])


def test_append_duplicates(test_df_year):
    other = copy.deepcopy(test_df_year)
    pytest.raises(ValueError, test_df_year.append, other=other)


def test_rename_data_cols():
    df = IamDataFrame(pd.DataFrame([
        ['model', 'scen', 'region_a', 'test_1', 'unit', 1, 5],
        ['model', 'scen', 'region_a', 'test_2', 'unit', 2, 6],
        ['model', 'scen', 'region_a', 'test_3', 'unit', 3, 7],
        ['model', 'scen', 'region_b', 'test_3', 'unit', 4, 8],
    ], columns=['model', 'scenario', 'region',
                'variable', 'unit', 2005, 2010],
    ))

    mapping = {'region': {'region_a': 'region_c'},
               'variable': {'test_1': 'test', 'test_3': 'test'}}

    obs = df.rename(mapping).data.reset_index(drop=True)

    exp = IamDataFrame(pd.DataFrame([
        ['model', 'scen', 'region_c', 'test', 'unit', 4, 12],
        ['model', 'scen', 'region_a', 'test_2', 'unit', 2, 6],
        ['model', 'scen', 'region_b', 'test_3', 'unit', 4, 8],
    ], columns=['model', 'scenario', 'region',
                'variable', 'unit', 2005, 2010],
    )).data.sort_values(by='region').reset_index(drop=True)

    pd.testing.assert_frame_equal(obs, exp, check_index_type=False)


def test_rename_index_data_fail(meta_df):
    mapping = {'scenario': {'a_scenario': 'a_scenario2'},
               'variable': {'Primary Energy|Coal': 'Primary Energy|Gas'}}
    pytest.raises(ValueError, meta_df.rename, mapping)


def test_rename_index_fail_duplicates(meta_df):
    mapping = {'scenario': {'a_scenario': 'a_scenario2'}}
    pytest.raises(ValueError, meta_df.rename, mapping)


def test_rename_index(meta_df):
    mapping = {'model': {'a_model': 'b_model'},
               'scenario': {'a_scenario': 'b_scen'}}
    obs = meta_df.rename(mapping)

    # test data changes
    exp = pd.DataFrame([
       ['b_model', 'b_scen', 'World', 'Primary Energy', 'EJ/y', 1, 6.],
       ['b_model', 'b_scen', 'World', 'Primary Energy|Coal', 'EJ/y', 0.5, 3],
       ['a_model', 'a_scenario2', 'World', 'Primary Energy', 'EJ/y', 2, 7],
    ], columns=['model', 'scenario', 'region', 'variable', 'unit', 2005, 2010]
    ).set_index(IAMC_IDX).sort_index()
    exp.columns = exp.columns.map(int)
    pd.testing.assert_frame_equal(obs.timeseries().sort_index(), exp)

    # test meta changes
    exp = pd.DataFrame([
        ['b_model', 'b_scen', False],
        ['a_model', 'a_scenario2', False],
    ], columns=['model', 'scenario', 'exclude']
    ).set_index(META_IDX)
    pd.testing.assert_frame_equal(obs.meta, exp)


def test_rename_append(meta_df):
    mapping = {'model': {'a_model': 'b_model'},
               'scenario': {'a_scenario': 'b_scen'}}
    obs = meta_df.rename(mapping, append=True)

    # test data changes
    exp = pd.DataFrame([
        ['a_model', 'a_scenario', 'World', 'Primary Energy', 'EJ/y', 1, 6.],
        ['a_model', 'a_scenario', 'World', 'Primary Energy|Coal', 'EJ/y', 0.5, 3],
        ['a_model', 'a_scenario2', 'World', 'Primary Energy', 'EJ/y', 2, 7],
        ['b_model', 'b_scen', 'World', 'Primary Energy', 'EJ/y', 1., 6.],
        ['b_model', 'b_scen', 'World', 'Primary Energy|Coal', 'EJ/y', .5, 3.],
    ], columns=['model', 'scenario', 'region', 'variable', 'unit', 2005, 2010]
    ).set_index(IAMC_IDX).sort_index()
    exp.columns = exp.columns.map(int)
    pd.testing.assert_frame_equal(obs.timeseries().sort_index(), exp)

    # test meta changes
    exp = pd.DataFrame([
        ['a_model', 'a_scenario', False],
        ['a_model', 'a_scenario2', False],
        ['b_model', 'b_scen', False],
    ], columns=['model', 'scenario', 'exclude']
    ).set_index(META_IDX)
    pd.testing.assert_frame_equal(obs.meta, exp)


def test_convert_unit():
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
