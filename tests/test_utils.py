import pytest
import pandas as pd
import numpy as np
from pandas import testing as pdt
from pandas import Timestamp
from datetime import datetime

from pyam.utils import (
    META_IDX,
    pattern_match,
    merge_meta,
    to_time,
)


def test_pattern_match_none():
    data = pd.Series(["foo", "bar"])
    values = ["baz"]

    obs = pattern_match(data, values)
    assert (obs == [False, False]).all()


def test_pattern_match_nan():
    data = pd.Series(["foo", np.nan])
    values = ["baz"]

    obs = pattern_match(data, values, has_nan=True)
    assert (obs == [False, False]).all()


def test_pattern_match_one():
    data = pd.Series(["foo", "bar"])
    values = ["foo"]

    obs = pattern_match(data, values)
    assert (obs == [True, False]).all()


def test_pattern_match_str_regex():
    data = pd.Series(["foo", "foo2", "bar"])
    values = ["foo"]

    obs = pattern_match(data, values)
    assert (obs == [True, False, False]).all()


def test_pattern_match_ast_regex():
    data = pd.Series(["foo", "foo2", "bar"])
    values = ["foo*"]

    obs = pattern_match(data, values)
    assert (obs == [True, True, False]).all()


def test_pattern_match_ast2_regex():
    data = pd.Series(["foo|bar", "foo", "bar"])
    values = ["*o*b*"]

    obs = pattern_match(data, values)
    assert (obs == [True, False, False]).all()


def test_pattern_match_plus():
    data = pd.Series(["foo", "foo+", "+bar", "b+az"])
    values = ["*+*"]

    obs = pattern_match(data, values)
    assert (obs == [False, True, True, True]).all()


def test_pattern_match_dot():
    data = pd.Series(["foo", "fo."])
    values = ["fo."]

    obs = pattern_match(data, values)
    assert (obs == [False, True]).all()


def test_pattern_match_brackets():
    data = pd.Series(["foo (bar)", "foo bar"])
    values = ["foo (bar)"]

    obs = pattern_match(data, values)
    assert (obs == [True, False]).all()


def test_pattern_match_dollar():
    data = pd.Series(["foo$bar", "foo"])
    values = ["foo$bar"]

    obs = pattern_match(data, values)
    assert (obs == [True, False]).all()


def test_pattern_regexp():
    data = pd.Series(["foo", "foa", "foo$"])
    values = ["fo.$"]

    obs = pattern_match(data, values, regexp=True)
    assert (obs == [True, True, False]).all()


def test_merge_meta():
    # test merging of two meta tables
    left = pd.DataFrame(
        [
            ["model_a", "scen_a", "foo", 1],
            ["model_a", "scen_b", "bar", 2],
        ],
        columns=META_IDX + ["string", "value"],
    ).set_index(META_IDX)
    right = pd.DataFrame(
        [
            ["model_a", "scen_a", "bar", 2],
            ["model_b", "scen_a", "baz", 3],
        ],
        columns=META_IDX + ["string", "value2"],
    ).set_index(META_IDX)

    # merge conflict raises an error
    pytest.raises(ValueError, merge_meta, left, right)

    # merge conflict ignoring errors yields expected results
    exp = pd.DataFrame(
        [
            ["model_a", "scen_a", "foo", 1, 2],
            ["model_a", "scen_b", "bar", 2, np.nan],
            ["model_b", "scen_a", "baz", np.nan, 3],
        ],
        columns=META_IDX + ["string", "value", "value2"],
    ).set_index(META_IDX)

    obs = merge_meta(left, right, ignore_conflict=True)
    pdt.assert_frame_equal(exp, obs)


@pytest.mark.parametrize(
    "x, exp",
    [
        ("2", 2),
        ("2010-07-10", Timestamp("2010-07-10 00:00")),
        (datetime(2010, 7, 10), Timestamp("2010-07-10 00:00")),
    ],
)
def test_to_time(x, exp):
    assert to_time(x) == exp


@pytest.mark.parametrize("x", [2.5, "2010-07-10 foo"])
def test_to_time_raises(x):
    with pytest.raises(ValueError, match=f"Invalid time domain: {x}"):
        to_time(x)
