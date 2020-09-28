from datetime import datetime
import pandas as pd
from pyam import IamDataFrame, validate, categorize, require_variable
from conftest import TEST_DTS


def test_require_variable(test_df):
    obs = test_df.require_variable(variable='Primary Energy|Coal',
                                   exclude_on_fail=True)
    assert len(obs) == 1
    assert obs.loc[0, 'scenario'] == 'scen_b'

    assert list(test_df['exclude']) == [False, True]


def test_require_variable_top_level(test_df):
    obs = require_variable(test_df, variable='Primary Energy|Coal',
                           exclude_on_fail=True)
    assert len(obs) == 1
    assert obs.loc[0, 'scenario'] == 'scen_b'

    assert list(test_df['exclude']) == [False, True]


def test_require_variable_year_list(test_df):
    years = [2005, 2010]

    # checking for variables that have ANY of the years in the list
    df = IamDataFrame(test_df.data[1:])
    df.require_variable(variable='Primary Energy', year=years,
                        exclude_on_fail=True)
    df.filter(exclude=False, inplace=True)

    assert len(df.variables()) == 2
    assert len(df.scenarios()) == 2

    # checking for variables that have ALL of the years in the list
    df = IamDataFrame(test_df.data[1:])
    for y in years:
        df.require_variable(variable='Primary Energy', year=y,
                            exclude_on_fail=True)
    df.filter(exclude=False, inplace=True)

    assert len(df.variables()) == 1
    assert len(df.scenarios()) == 1


def test_validate_all_pass(test_df):
    obs = test_df.validate(
        {'Primary Energy': {'up': 10}}, exclude_on_fail=True)
    assert obs is None
    assert len(test_df.data) == 6  # data unchanged

    assert list(test_df['exclude']) == [False, False]  # none excluded


def test_validate_nonexisting(test_df):
    obs = test_df.validate({'Primary Energy|Coal': {'up': 2}},
                           exclude_on_fail=True)
    assert len(obs) == 1
    assert obs.index.get_level_values('scenario').values[0] == 'scen_a'

    assert list(test_df['exclude']) == [True, False]  # scenario with failed
    # validation excluded, scenario with non-defined value passes validation


def test_validate_up(test_df):
    obs = test_df.validate({'Primary Energy': {'up': 6.5}},
                           exclude_on_fail=False)
    assert len(obs) == 1
    if 'year' in test_df.data:
        assert obs.index.get_level_values('year').values[0] == 2010
    else:
        exp_time = pd.to_datetime(datetime(2010, 7, 21))
        assert pd.to_datetime(obs.index.get_level_values('time')
                              .values[0]).date() == exp_time

    assert list(test_df['exclude']) == [False, False]  # assert none excluded


def test_validate_lo(test_df):
    obs = test_df.validate({'Primary Energy': {'up': 8, 'lo': 2.0}})
    assert len(obs) == 1
    if 'year' in test_df.data:
        assert obs.index.get_level_values('year').values[0] == 2005
    else:
        exp_year = pd.to_datetime(datetime(2005, 6, 17))
        assert pd.to_datetime(obs.index.get_level_values('time')
                              .values[0]).date() == exp_year

    assert list(obs.index.get_level_values('scenario').values) == ['scen_a']


def test_validate_both(test_df):
    obs = test_df.validate({'Primary Energy': {'up': 6.5, 'lo': 2.0}})
    assert len(obs) == 2
    if 'year' in test_df.data:
        assert list(obs.index.get_level_values('year').values) == [2005, 2010]
    else:
        exp_time = pd.to_datetime(TEST_DTS)
        obs.index.set_levels(obs.index.get_level_values('time')
                                .normalize(), level=5)
        assert (pd.to_datetime(obs.index.get_level_values('time').values)
                .date == exp_time).all()

    exp = ['scen_a', 'scen_b']
    assert list(obs.index.get_level_values('scenario').values) == exp


def test_validate_year(test_df):
    obs = test_df.validate({'Primary Energy': {'up': 5.0, 'year': 2005}},
                           exclude_on_fail=False)
    assert obs is None

    obs = test_df.validate({'Primary Energy': {'up': 5.0, 'year': 2010}},
                           exclude_on_fail=False)
    assert len(obs) == 2


def test_validate_exclude(test_df):
    test_df.validate({'Primary Energy': {'up': 6.0}}, exclude_on_fail=True)
    assert list(test_df['exclude']) == [False, True]


def test_validate_top_level(test_df):
    obs = validate(test_df, criteria={'Primary Energy': {'up': 6.0}},
                   exclude_on_fail=True, variable='Primary Energy')
    assert len(obs) == 1
    if 'year' in test_df._data.index.names:
        assert obs.index.get_level_values('year').values[0] == 2010
    else:
        exp_time = datetime(2010, 7, 21).date()
        obs_time = obs.index.get_level_values('time')[0].date()
        assert exp_time == obs_time
    assert list(test_df['exclude']) == [False, True]


def test_category_none(test_df):
    test_df.categorize('category', 'Testing', {'Primary Energy': {'up': 0.8}})
    assert 'category' not in test_df.meta.columns


def test_category_pass(test_df):
    dct = {'model': ['model_a', 'model_a'],
           'scenario': ['scen_a', 'scen_b'],
           'category': ['foo', None]}
    exp = pd.DataFrame(dct).set_index(['model', 'scenario'])['category']

    test_df.categorize('category', 'foo', {'Primary Energy':
                                           {'up': 6, 'year': 2010}})
    obs = test_df['category']
    pd.testing.assert_series_equal(obs, exp)


def test_category_top_level(test_df):
    dct = {'model': ['model_a', 'model_a'],
           'scenario': ['scen_a', 'scen_b'],
           'category': ['foo', None]}
    exp = pd.DataFrame(dct).set_index(['model', 'scenario'])['category']

    categorize(test_df, 'category', 'foo',
               criteria={'Primary Energy': {'up': 6, 'year': 2010}},
               variable='Primary Energy')
    obs = test_df['category']
    pd.testing.assert_series_equal(obs, exp)
