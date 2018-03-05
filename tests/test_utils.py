import pandas as pd

from pyam import utils


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


def test_pattern_match_str_regex():
    data = pd.Series(['foo', 'foo2', 'bar'])
    values = ['foo']

    obs = utils.pattern_match(data, values)
    exp = [True, False, False]

    assert (obs == exp).all()


def test_pattern_match_ast_regex():
    data = pd.Series(['foo', 'foo2', 'bar'])
    values = ['foo*']

    obs = utils.pattern_match(data, values)
    exp = [True, True, False]

    assert (obs == exp).all()


def test_pattern_match_plus():
    data = pd.Series(['foo', 'foo+', '+bar', 'b+az'])
    values = ['*+*']

    obs = utils.pattern_match(data, values)
    exp = [False, True, True, True]

    assert (obs == exp).all()
