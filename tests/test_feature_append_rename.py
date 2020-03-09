import copy
import pytest
import datetime as dt

import numpy as np
import pandas as pd
from numpy import testing as npt

from pyam import IamDataFrame, META_IDX, IAMC_IDX, compare

from conftest import TEST_DTS


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


def test_append_other_scenario(test_df):
    other = test_df.filter(scenario='scen_b')\
        .rename({'scenario': {'scen_b': 'scen_c'}})

    test_df.set_meta([0, 1], name='col1')
    test_df.set_meta(['a', 'b'], name='col2')

    other.set_meta(2, name='col1')
    other.set_meta('x', name='col3')

    df = test_df.append(other)

    # check that the original meta dataframe is not updated
    obs = test_df.meta.index.get_level_values(1)
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


def test_append_same_scenario(test_df):
    other = test_df.filter(scenario='scen_b')\
        .rename({'variable': {'Primary Energy': 'Primary Energy clone'}})

    test_df.set_meta([0, 1], name='col1')

    other.set_meta(2, name='col1')
    other.set_meta('b', name='col2')

    # check that non-matching meta raise an error
    pytest.raises(ValueError, test_df.append, other=other)

    # check that ignoring meta conflict works as expetced
    df = test_df.append(other, ignore_meta_conflict=True)

    # check that the new meta.index is updated, but not the original one
    npt.assert_array_equal(test_df.meta.columns, ['exclude', 'col1'])

    # assert that merging of meta works as expected
    exp = test_df.meta.copy()
    exp['col2'] = [np.nan, 'b']
    pd.testing.assert_frame_equal(df.meta, exp)

    # assert that appending data works as expected
    ts = df.timeseries()
    npt.assert_array_equal(ts.iloc[2], ts.iloc[3])


@pytest.mark.parametrize("shuffle_cols", [True, False])
def test_append_extra_col(test_df, shuffle_cols):
    base_data = test_df.data.copy()

    base_data["col_1"] = "hi"
    base_data["col_2"] = "bye"
    base_df = IamDataFrame(base_data)

    other_data = base_data[base_data["variable"] == "Primary Energy"].copy()
    other_data["variable"] = "Primary Energy|Gas"
    other_df = IamDataFrame(other_data)

    if shuffle_cols:
        c1_idx = other_df._LONG_IDX.index("col_1")
        c2_idx = other_df._LONG_IDX.index("col_2")
        other_df._LONG_IDX[c1_idx] = "col_2"
        other_df._LONG_IDX[c2_idx] = "col_1"

    res = base_df.append(other_df)

    def check_meta_is(iamdf, meta_col, val):
        for checker in [iamdf.timeseries().reset_index(), iamdf.data]:
            meta_vals = checker[meta_col].unique()
            assert len(meta_vals) == 1, meta_vals
            assert meta_vals[0] == val, meta_vals

    # ensure meta merged correctly
    check_meta_is(res, "col_1", "hi")
    check_meta_is(res, "col_2", "bye")


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


def test_rename_conflict(test_df):
    mapping = {'scenario': {'scen_a': 'scen_b'}}
    pytest.raises(ValueError, test_df.rename, mapping, **mapping)


def test_rename_index_data_fail(test_df):
    mapping = {'scenario': {'scen_a': 'scen_c'},
               'variable': {'Primary Energy|Coal': 'Primary Energy|Gas'}}
    pytest.raises(ValueError, test_df.rename, mapping)


def test_rename_index_fail_duplicates(test_df):
    mapping = {'scenario': {'scen_a': 'scen_b'}}
    pytest.raises(ValueError, test_df.rename, mapping)


def test_rename_index(test_df):
    mapping = {'model': {'model_a': 'model_b'}}
    obs = test_df.rename(mapping, scenario={'scen_a': 'scen_c'})

    # test data changes
    times = [2005, 2010] if 'year' in test_df.data else TEST_DTS
    exp = pd.DataFrame([
        ['model_b', 'scen_c', 'World', 'Primary Energy', 'EJ/yr', 1, 6.],
        ['model_b', 'scen_c', 'World', 'Primary Energy|Coal', 'EJ/yr', 0.5, 3],
        ['model_a', 'scen_b', 'World', 'Primary Energy', 'EJ/yr', 2, 7],
    ], columns=['model', 'scenario', 'region', 'variable', 'unit'] + times
    ).set_index(IAMC_IDX).sort_index()
    if "year" in test_df.data:
        exp.columns = list(map(int, exp.columns))
    else:
        obs.data.time = obs.data.time.dt.normalize()
        exp.columns = pd.to_datetime(exp.columns)
    pd.testing.assert_frame_equal(obs.timeseries().sort_index(), exp)

    # test meta changes
    exp = pd.DataFrame([
        ['model_b', 'scen_c', False],
        ['model_a', 'scen_b', False],
    ], columns=['model', 'scenario', 'exclude']
    ).set_index(META_IDX)
    pd.testing.assert_frame_equal(obs.meta, exp)


def test_rename_append(test_df):
    mapping = {'model': {'model_a': 'model_b'},
               'scenario': {'scen_a': 'scen_c'}}
    obs = test_df.rename(mapping, append=True)

    # test data changes
    times = [2005, 2010] if "year" in test_df.data else TEST_DTS
    exp = pd.DataFrame([
        ['model_a', 'scen_a', 'World', 'Primary Energy', 'EJ/yr', 1, 6.],
        ['model_a', 'scen_a', 'World', 'Primary Energy|Coal', 'EJ/yr', 0.5, 3],
        ['model_a', 'scen_b', 'World', 'Primary Energy', 'EJ/yr', 2, 7],
        ['model_b', 'scen_c', 'World', 'Primary Energy', 'EJ/yr', 1, 6.],
        ['model_b', 'scen_c', 'World', 'Primary Energy|Coal', 'EJ/yr', 0.5, 3],
    ], columns=['model', 'scenario', 'region', 'variable', 'unit'] + times
    ).set_index(IAMC_IDX).sort_index()
    if "year" in test_df.data:
        exp.columns = list(map(int, exp.columns))
    else:
        obs.data.time = obs.data.time.dt.normalize()
        exp.columns = pd.to_datetime(exp.columns)
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
