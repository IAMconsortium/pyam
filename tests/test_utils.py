import pandas as pd

from pyam_analysis import utils


def test_pattern_match_none():
    data = pd.Series(['foo', 'bar'])
    values = ['baz']

    obs = utils.pattern_match(data, values)
    exp = [False, False]

    assert (obs == exp).all()


def test_pattern_match_one():
    data = pd.Series(['foo', 'bar'])
    values = ['foo']

    obs = utils.pattern_match(data, values)
    exp = [True, False]

    assert (obs == exp).all()


def test_pattern_match_one_regex():
    data = pd.Series(['foo', 'bar'])
    values = ['f*']

    obs = utils.pattern_match(data, values, pseudo_regex=True)
    exp = [True, False]

    assert (obs == exp).all()


def test_pattern_match_plus():
    data = pd.Series(['foo+', '+bar', 'b+az'])
    values = ['*+*']

    obs = utils.pattern_match(data, values, pseudo_regex=True)
    exp = [True, True, True]

    assert (obs == exp).all()
