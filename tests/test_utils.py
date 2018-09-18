import pandas as pd
import numpy as np

from pyam import utils


def test_pattern_match_none():
    data = pd.Series(['foo', 'bar'])
    values = ['baz']

    obs = utils.pattern_match(data, values)
    exp = [False, False]

    assert (obs == exp).all()


def test_pattern_match_nan():
    data = pd.Series(['foo', np.nan])
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


def test_pattern_match_dot():
    data = pd.Series(['foo', 'fo.'])
    values = ['fo.']

    obs = utils.pattern_match(data, values)
    exp = [False, True]

    assert (obs == exp).all()


def test_pattern_match_brackets():
    data = pd.Series(['foo (bar)', 'foo bar'])
    values = ['foo (bar)']

    obs = utils.pattern_match(data, values)
    exp = [True, False]

    assert (obs == exp).all()


def test_pattern_match_dollar():
    data = pd.Series(['foo$bar', 'foo'])
    values = ['foo$bar']

    obs = utils.pattern_match(data, values)
    exp = [True, False]

    assert (obs == exp).all()


def test_pattern_regexp():
    data = pd.Series(['foo', 'foa', 'foo$'])
    values = ['fo.$']

    obs = utils.pattern_match(data, values, regexp=True)
    exp = [True, True, False]

    assert (obs == exp).all()
