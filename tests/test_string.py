import numpy as np
import pandas as pd
import pytest

from pyam.str import (
    find_depth,
    get_variable_components,
    reduce_hierarchy,
    concat_with_pipe,
)

TEST_VARS = ["foo", "foo|bar", "foo|bar|baz"]
TEST_CONCAT_SERIES = pd.Series(["foo", "bar", "baz"], index=["f", "b", "z"])


def test_find_depth_as_list():
    obs = find_depth(TEST_VARS)
    assert obs == [0, 1, 2]


def test_find_depth_as_str():
    assert find_depth("foo|bar|baz") == 2


def test_find_depth_with_str():
    data = pd.Series(["foo", "foo|bar|baz", "bar|baz", "bar|baz|foo"])
    obs = find_depth(data, "bar")
    assert obs == [None, None, 1, 2]


def test_find_depth_with_str_1():
    data = pd.Series(["foo", "foo|bar|baz", "bar|baz", "bar|baz|foo"])
    obs = find_depth(data, "bar|", 1)
    assert obs == [False, False, False, True]


def test_find_depth_with_str_0():
    data = pd.Series(["foo", "foo|bar|baz", "bar|baz", "bar|baz|foo"])
    obs = find_depth(data, "*bar|", 0)
    assert obs == [False, True, True, False]


def test_find_depth_0():
    obs = find_depth(TEST_VARS, level=0)
    assert obs == [True, False, False]


def test_find_depth_0_minus():
    obs = find_depth(TEST_VARS, level="0-")
    assert obs == [True, False, False]


def test_find_depth_0_plus():
    obs = find_depth(TEST_VARS, level="0+")
    assert obs == [True, True, True]


def test_find_depth_1():
    obs = find_depth(TEST_VARS, level=1)
    assert obs == [False, True, False]


def test_find_depth_1_minus():
    obs = find_depth(TEST_VARS, level="1-")
    assert obs == [True, True, False]


def test_find_depth_1_plus():
    obs = find_depth(TEST_VARS, level="1+")
    assert obs == [False, True, True]


def test_concat_with_pipe_all():
    obs = concat_with_pipe(TEST_CONCAT_SERIES)
    assert obs == "foo|bar|baz"


def test_concat_with_pipe_exclude_none():
    s = TEST_CONCAT_SERIES.copy()
    s["b"] = None
    obs = concat_with_pipe(s)
    assert obs == "foo|baz"


def test_concat_with_pipe_exclude_nan():
    s = TEST_CONCAT_SERIES.copy()
    s["b"] = np.nan
    obs = concat_with_pipe(s)
    assert obs == "foo|baz"


def test_concat_with_pipe_by_name():
    obs = concat_with_pipe(TEST_CONCAT_SERIES, cols=["f", "z"])
    assert obs == "foo|baz"


def test_concat_list_with_pipe():
    obs = concat_with_pipe(["foo", "bar"])
    assert obs == "foo|bar"


def test_concat_list_with_pipe_by_cols():
    obs = concat_with_pipe(["foo", "bar", "baz"], cols=[0, 2])
    assert obs == "foo|baz"


def test_concat_args_with_pipe():
    obs = concat_with_pipe("foo", "bar")
    assert obs == "foo|bar"


def test_concat_args_with_pipe_by_cols():
    obs = concat_with_pipe("foo", "bar", "baz", cols=[0, 2])
    assert obs == "foo|baz"


def test_concat_args_deprecated():
    # test error message for legacy-issues when introducing `*args` (#778)
    with pytest.raises(DeprecationWarning, match="Please use `cols=\[0, 2\]`."):
        concat_with_pipe(["foo", "bar", "baz"], [0, 2])


def test_reduce_hierarchy_0():
    assert reduce_hierarchy("foo|bar|baz", 0) == "foo"


def test_reduce_hierarchy_1():
    assert reduce_hierarchy("foo|bar|baz", 1) == "foo|bar"


def test_reduce_hierarchy_neg1():
    assert reduce_hierarchy("foo|bar|baz", -1) == "foo|bar"


def test_reduce_hierarchy_neg2():
    assert reduce_hierarchy("foo|bar|baz", -2) == "foo"


def test_get_variable_components_int():
    assert get_variable_components("foo|bar|baz", 1) == "bar"


def test_get_variable_components_list():
    assert get_variable_components("foo|bar|baz", [1, 2]) == ["bar", "baz"]


def test_get_variable_components_indexError():
    with pytest.raises(IndexError):
        get_variable_components("foo|bar|baz", 3)


def test_get_variable_components_join_true():
    assert get_variable_components("foo|bar|baz", [0, 2], join=True) == "foo|baz"


def test_get_variable_components_join_str():
    assert get_variable_components("foo|bar|baz", [2, 1], join="_") == "baz_bar"
