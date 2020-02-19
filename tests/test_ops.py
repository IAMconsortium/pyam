import pytest

import pandas as pd
from pandas.testing import assert_frame_equal

from pyam.core import IamDataFrame
from pyam.ops import BinaryOp, subtract as subtract_op
from pyam.utils import META_IDX, LONG_IDX

from pyam.testing import assert_iamframe_equal

class MockOpDf(IamDataFrame):

    def __init__(self, df, meta=pd.DataFrame()):
        self.data = df
        self.meta = meta


def test_constructor_meta():
    # meta values are the same, so all is good. data doesn't matter
    meta = pd.DataFrame({'scenario': ['foo',]})
    a = MockOpDf(1, meta=meta)
    b = MockOpDf(2, meta=meta)
    op = BinaryOp(a, b)
    
    obs = op.meta
    exp = meta
    assert_frame_equal(obs, exp)

def test_constructor_meta_raises():
    # meta values are different, which is a no go - data doesn't matter
    a = MockOpDf(1, meta=pd.DataFrame({'scenario': ['foo',]}))
    b = MockOpDf(2, meta=pd.DataFrame({'scenario': ['bar',]}))
    pytest.raises(ValueError, BinaryOp, a, b)

def test_constructor_meta_doesnt_raise():
    # meta values are different, but should be ignored - data doesn't matter
    meta = pd.DataFrame({'scenario': ['foo',]})
    a = MockOpDf(1, meta=pd.DataFrame({'scenario': ['foo',]}))
    b = MockOpDf(2, meta=pd.DataFrame({'scenario': ['bar',]}))
    op = BinaryOp(a, b, ignore_meta_conflict=True)

    # when ignored, meta takes a-value
    obs = op.meta
    exp = meta
    assert_frame_equal(obs, exp)

    
def test_op_data():
    # checking to make sure data is in the right format (index is correct,
    # columns are correct)
    axis = 'variable'
    a = MockOpDf(pd.DataFrame({
        'value': [1,],
        axis: ['var1',],
        'region': ['foo',],
    }))
    b = MockOpDf(pd.DataFrame({
        'value': [2,],
        axis: ['var2',],
        'region': ['foo',],
    }))
    op = BinaryOp(a, b)
    obs_a, obs_b = op.op_data(axis=axis)

    exp_a = pd.DataFrame({'value': [1,]}, index=pd.Index(['foo',], name='region'))
    assert_frame_equal(obs_a, exp_a)

    exp_b = pd.DataFrame({'value': [2,]}, index=pd.Index(['foo',], name='region'))
    assert_frame_equal(obs_b, exp_b)

def test_op_data_raises_too_many_a():
    # test that you can only have one ``a`` value along axis of interest
    axis = 'variable'
    a = MockOpDf(pd.DataFrame({
        'value': [1, 1,],
        axis: ['var1', 'var2',],
        'region': ['foo', 'foo',],
    }))
    b = MockOpDf(pd.DataFrame({
        'value': [2,],
        axis: ['var2',],
        'region': ['foo',],
    }))
    op = BinaryOp(a, b)
    pytest.raises(ValueError, op.op_data, axis=axis)


def test_op_data_raises_too_many_b():
    # test that you can only have one ``b`` value along axis of interest
    axis = 'variable'
    a = MockOpDf(pd.DataFrame({
        'value': [1,],
        axis: ['var1',],
        'region': ['foo',],
    }))
    b = MockOpDf(pd.DataFrame({
        'value': [2, 2,],
        axis: ['var1', 'var2',],
        'region': ['foo', 'foo',],
    }))
    op = BinaryOp(a, b)
    pytest.raises(ValueError, op.op_data, axis=axis)

def test_calc_meta(test_df_year):
    a = test_df_year.filter(scenario='scen_a')
    b = test_df_year.filter(scenario='scen_b')
    op = BinaryOp(a, b)

    res_meta = pd.DataFrame({
        'model': ['model_a', 'model_a'],
        'scenario': ['scen_a', 'scen_b'],
    })

    exp = op.meta
    obs = op.calc_meta(res_meta)
    assert_frame_equal(obs, exp)

    
def test_calc_meta_reduced_result(test_df_year):
    a = test_df_year.filter(scenario='scen_a')
    b = test_df_year.filter(scenario='scen_b')
    op = BinaryOp(a, b)

    res_meta = pd.DataFrame({
        'model': ['model_a',],
        'scenario': ['scen_a',],
    })

    obs = op.calc_meta(res_meta)
    res_meta['exclude'] = False
    exp = res_meta.set_index(META_IDX)
    assert_frame_equal(obs, exp)

def test_calc_subtract_inplace_default(test_df_year):
    a = test_df_year.filter(scenario='scen_a', variable='Primary Energy')
    b = test_df_year.filter(scenario='scen_a', variable='Primary Energy|Coal')
    
    op = BinaryOp(a, b)
    axis = 'variable'
    axis_value = 'Non-Coal PE'
    obs_data, obs_meta = op.calc(subtract_op, axis=axis, axis_value=axis_value)

    # test data
    exp = a.copy()
    exp['value'] = [0.5, 3.0]
    exp['variable'] = [axis_value, axis_value]
    idx = LONG_IDX + ['value']
    assert_frame_equal(obs_data[idx], exp.data[idx])

    # test meta
    assert_frame_equal(obs_meta, exp.meta)

def test_calc_subtract_self(test_df_year):
    a = test_df_year.filter(scenario='scen_a', variable='Primary Energy')
    
    op = BinaryOp(a, a)
    axis = 'variable'
    axis_value = 'Non-Coal PE'
    obs_data, obs_meta = op.calc(subtract_op, axis=axis, axis_value=axis_value)

    # test data
    exp = a.copy()
    exp['value'] = [0.0, 0.0]
    exp['variable'] = [axis_value, axis_value]
    idx = LONG_IDX + ['value']
    assert_frame_equal(obs_data[idx], exp.data[idx])

    # test meta
    assert_frame_equal(a.meta, exp.meta)
    
    
def test_calc_subtract_inplace_scenario(test_df_year):
    a = test_df_year.filter(scenario='scen_a', variable='Primary Energy')
    b = test_df_year.filter(scenario='scen_b', variable='Primary Energy')
    
    op = BinaryOp(a, b)
    axis = 'scenario'
    axis_value = 'scen_a - scen_b'
    obs_data, obs_meta = op.calc(subtract_op, axis=axis, axis_value=axis_value)

    exp = a.copy()
    exp['value'] = [-1.0, -1.0]
    exp['scenario'] = [axis_value, axis_value]
    idx = LONG_IDX + ['value']
    assert_frame_equal(obs_data[idx], exp.data[idx])
    # expected to be empty, but with correct structure
    assert obs_meta.empty
    assert obs_meta.columns == ['exclude']
    assert obs_meta.index.names == ['model', 'scenario']

def test_calc_subtract_iamdataframe(test_df_year):
    a = test_df_year.filter(scenario='scen_a', variable='Primary Energy')
    b = test_df_year.filter(scenario='scen_a', variable='Primary Energy|Coal')

    axis = 'variable'
    axis_value = 'Non-Coal PE'
    obs = a.subtract(b, axis=axis, new_name=axis_value)
    
    # test data
    exp = a.copy()
    exp['value'] = [0.5, 3.0]
    exp['variable'] = [axis_value, axis_value]
    idx = LONG_IDX + ['value']
    assert_iamframe_equal(obs, exp)

def test_calc_subtract_inplace(test_df_year):
    df = test_df_year.filter(scenario='scen_a')
    a = 'Primary Energy'
    b = 'Primary Energy|Coal'
    axis = 'variable'
    axis_value = 'Non-Coal PE'

    # exp value
    exp = df.filter(variable='Primary Energy').copy()
    exp['value'] = [0.5, 3.0]
    exp['variable'] = [axis_value, axis_value]

    # test append=False
    idx = LONG_IDX + ['value']
    obs = df.subtract_inplace(a, b, axis=axis, new_name=axis_value,
                              append=False)
    assert_iamframe_equal(obs, exp)

    # test append=True
    obs = df.subtract_inplace(a, b, axis=axis, new_name=axis_value,
                              append=True)
    assert obs == None
    obs = df.filter(variable=axis_value)
    assert_iamframe_equal(obs, exp)
