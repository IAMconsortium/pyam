import copy
import pytest

import numpy as np
import pandas as pd
from numpy import testing as npt


from pyam import IamDataFrame, META_IDX, IAMC_IDX, compare


RENAME_DF = IamDataFrame(pd.DataFrame([
    ['model', 'scen', 'region_a', 'test_1', 'unit', 1, 5],
    ['model', 'scen', 'region_a', 'test_2', 'unit', 2, 6],
    ['model', 'scen', 'region_a', 'test_3', 'unit', 3, 7],
    ['model', 'scen', 'region_b', 'test_3', 'unit', 4, 8],
], columns=['model', 'scenario', 'region',
            'variable', 'unit', 2005, 2010],
))

# expected output
EXP_RENAME_DF = IamDataFrame(pd.DataFrame([
    ['model', 'scen', 'region_c', 'test', 'unit', 4, 12],
    ['model', 'scen', 'region_a', 'test_2', 'unit', 2, 6],
    ['model', 'scen', 'region_b', 'test_3', 'unit', 4, 8],
], columns=['model', 'scenario', 'region',
            'variable', 'unit', 2005, 2010],
)).data.sort_values(by='region').reset_index(drop=True)


def test_append_other_scenario(meta_df):
    other = meta_df.filter(scenario='scen_b')\
        .rename({'scenario': {'scen_b': 'scen_c'}})

    meta_df.set_meta([0, 1], name='col1')
    meta_df.set_meta(['a', 'b'], name='col2')

    other.set_meta(2, name='col1')
    other.set_meta('x', name='col3')

    df = meta_df.append(other)

    # check that the original meta dataframe is not updated
    obs = meta_df.meta.index.get_level_values(1)
    npt.assert_array_equal(obs, ['scen_a', 'scen_b'])

    # assert that merging of meta works as expected
    exp = pd.DataFrame([
        ['model_a', 'scen_a', False, 0, 'a', np.nan],
        ['model_a', 'scen_b', False, 1, 'b', np.nan],
        ['model_a', 'scen_c', False, 2, np.nan, 'x'],
    ], columns=['model', 'scenario', 'exclude', 'col1', 'col2', 'col3']
    ).set_index(['model', 'scenario'])

    # sort columns for assertion in older pandas versions
    df.meta = df.meta.reindex(columns=exp.columns)
    pd.testing.assert_frame_equal(df.meta, exp)

    # assert that appending data works as expected
    ts = df.timeseries()
    npt.assert_array_equal(ts.iloc[2].values, ts.iloc[3].values)


def test_append_same_scenario(meta_df):
    other = meta_df.filter(scenario='scen_b')\
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


def test_rename_data_cols_by_dict():
    args = {'mapping': {'variable': {'test_1': 'test', 'test_3': 'test'},
                        'region': {'region_a': 'region_c'}}}
    obs = RENAME_DF.rename(**args).data.reset_index(drop=True)
    pd.testing.assert_frame_equal(obs, EXP_RENAME_DF, check_index_type=False)


def test_rename_data_cols_by_kwargs():
    args = {'variable': {'test_1': 'test', 'test_3': 'test'},
            'region': {'region_a': 'region_c'}}
    obs = RENAME_DF.rename(**args).data.reset_index(drop=True)
    pd.testing.assert_frame_equal(obs, EXP_RENAME_DF, check_index_type=False)


def test_rename_data_cols_by_mixed():
    args = {'mapping': {'variable': {'test_1': 'test', 'test_3': 'test'}},
            'region': {'region_a': 'region_c'}}
    obs = RENAME_DF.rename(**args).data.reset_index(drop=True)
    pd.testing.assert_frame_equal(obs, EXP_RENAME_DF, check_index_type=False)


def test_rename_conflict(meta_df):
    mapping = {'scenario': {'scen_a': 'scen_b'}}
    pytest.raises(ValueError, meta_df.rename, mapping, **mapping)


def test_rename_index_data_fail(meta_df):
    mapping = {'scenario': {'scen_a': 'scen_c'},
               'variable': {'Primary Energy|Coal': 'Primary Energy|Gas'}}
    pytest.raises(ValueError, meta_df.rename, mapping)


def test_rename_index_fail_duplicates(meta_df):
    mapping = {'scenario': {'scen_a': 'scen_b'}}
    pytest.raises(ValueError, meta_df.rename, mapping)


def test_rename_index(meta_df):
    mapping = {'model': {'model_a': 'model_b'}}
    obs = meta_df.rename(mapping, scenario={'scen_a': 'scen_c'})

    # test data changes
    exp = pd.DataFrame([
        ['model_b', 'scen_c', 'World', 'Primary Energy', 'EJ/y', 1, 6.],
        ['model_b', 'scen_c', 'World', 'Primary Energy|Coal', 'EJ/y', 0.5, 3],
        ['model_a', 'scen_b', 'World', 'Primary Energy', 'EJ/y', 2, 7],
    ], columns=['model', 'scenario', 'region', 'variable', 'unit', 2005, 2010]
    ).set_index(IAMC_IDX).sort_index()
    exp.columns = exp.columns.map(int)
    pd.testing.assert_frame_equal(obs.timeseries().sort_index(), exp)

    # test meta changes
    exp = pd.DataFrame([
        ['model_b', 'scen_c', False],
        ['model_a', 'scen_b', False],
    ], columns=['model', 'scenario', 'exclude']
    ).set_index(META_IDX)
    pd.testing.assert_frame_equal(obs.meta, exp)


def test_rename_append(meta_df):
    mapping = {'model': {'model_a': 'model_b'},
               'scenario': {'scen_a': 'scen_c'}}
    obs = meta_df.rename(mapping, append=True)

    # test data changes
    exp = pd.DataFrame([
        ['model_a', 'scen_a', 'World', 'Primary Energy', 'EJ/y', 1, 6.],
        ['model_a', 'scen_a', 'World', 'Primary Energy|Coal', 'EJ/y', 0.5, 3],
        ['model_a', 'scen_b', 'World', 'Primary Energy', 'EJ/y', 2, 7],
        ['model_b', 'scen_c', 'World', 'Primary Energy', 'EJ/y', 1, 6.],
        ['model_b', 'scen_c', 'World', 'Primary Energy|Coal', 'EJ/y', 0.5, 3],
    ], columns=['model', 'scenario', 'region', 'variable', 'unit', 2005, 2010]
    ).set_index(IAMC_IDX).sort_index()
    exp.columns = exp.columns.map(int)
    pd.testing.assert_frame_equal(obs.timeseries().sort_index(), exp)

    # test meta changes
    exp = pd.DataFrame([
        ['model_a', 'scen_a', False],
        ['model_a', 'scen_b', False],
        ['model_b', 'scen_c', False],
    ], columns=['model', 'scenario', 'exclude']
    ).set_index(META_IDX)
    pd.testing.assert_frame_equal(obs.meta, exp)


def test_rename_duplicates():
    mapping = {'variable': {'test_1': 'test_3'}}
    pytest.raises(ValueError, RENAME_DF.rename, **mapping)

    obs = RENAME_DF.rename(check_duplicates=False, **mapping)

    exp = IamDataFrame(pd.DataFrame([
        ['model', 'scen', 'region_a', 'test_2', 'unit', 2, 6],
        ['model', 'scen', 'region_a', 'test_3', 'unit', 4, 12],
        ['model', 'scen', 'region_b', 'test_3', 'unit', 4, 8],
    ], columns=['model', 'scenario', 'region',
                'variable', 'unit', 2005, 2010],
    ))

    assert compare(obs, exp).empty
    pd.testing.assert_frame_equal(obs.data, exp.data)


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
