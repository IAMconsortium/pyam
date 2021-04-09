import pytest
import pandas as pd
import numpy as np
from pandas import testing as pdt

from pyam import utils, META_IDX

TEST_VARS = ["foo", "foo|bar", "foo|bar|baz"]
TEST_CONCAT_SERIES = pd.Series(["foo", "bar", "baz"], index=["f", "b", "z"])


def test_pattern_match_none():
    data = pd.Series(["foo", "bar"])
    values = ["baz"]

    obs = utils.pattern_match(data, values)
    assert (obs == [False, False]).all()


def test_pattern_match_nan():
    data = pd.Series(["foo", np.nan])
    values = ["baz"]

    obs = utils.pattern_match(data, values, has_nan=True)
    assert (obs == [False, False]).all()


def test_pattern_match_one():
    data = pd.Series(["foo", "bar"])
    values = ["foo"]

    obs = utils.pattern_match(data, values)
    assert (obs == [True, False]).all()


def test_pattern_match_str_regex():
    data = pd.Series(["foo", "foo2", "bar"])
    values = ["foo"]

    obs = utils.pattern_match(data, values)
    assert (obs == [True, False, False]).all()


def test_pattern_match_ast_regex():
    data = pd.Series(["foo", "foo2", "bar"])
    values = ["foo*"]

    obs = utils.pattern_match(data, values)
    assert (obs == [True, True, False]).all()


def test_pattern_match_ast2_regex():
    data = pd.Series(["foo|bar", "foo", "bar"])
    values = ["*o*b*"]

    obs = utils.pattern_match(data, values)
    assert (obs == [True, False, False]).all()


def test_pattern_match_plus():
    data = pd.Series(["foo", "foo+", "+bar", "b+az"])
    values = ["*+*"]

    obs = utils.pattern_match(data, values)
    assert (obs == [False, True, True, True]).all()


def test_pattern_match_dot():
    data = pd.Series(["foo", "fo."])
    values = ["fo."]

    obs = utils.pattern_match(data, values)
    assert (obs == [False, True]).all()


def test_pattern_match_brackets():
    data = pd.Series(["foo (bar)", "foo bar"])
    values = ["foo (bar)"]

    obs = utils.pattern_match(data, values)
    assert (obs == [True, False]).all()


def test_pattern_match_dollar():
    data = pd.Series(["foo$bar", "foo"])
    values = ["foo$bar"]

    obs = utils.pattern_match(data, values)
    assert (obs == [True, False]).all()


def test_pattern_regexp():
    data = pd.Series(["foo", "foa", "foo$"])
    values = ["fo.$"]

    obs = utils.pattern_match(data, values, regexp=True)
    assert (obs == [True, True, False]).all()


def test_find_depth_as_list():
    obs = utils.find_depth(TEST_VARS)
    assert obs == [0, 1, 2]


def test_find_depth_as_str():
    assert utils.find_depth("foo|bar|baz") == 2


def test_find_depth_with_str():
    data = pd.Series(["foo", "foo|bar|baz", "bar|baz", "bar|baz|foo"])
    obs = utils.find_depth(data, "bar")
    assert obs == [None, None, 1, 2]


def test_find_depth_with_str_1():
    data = pd.Series(["foo", "foo|bar|baz", "bar|baz", "bar|baz|foo"])
    obs = utils.find_depth(data, "bar|", 1)
    assert obs == [False, False, False, True]


def test_find_depth_with_str_0():
    data = pd.Series(["foo", "foo|bar|baz", "bar|baz", "bar|baz|foo"])
    obs = utils.find_depth(data, "*bar|", 0)
    assert obs == [False, True, True, False]


def test_find_depth_0():
    obs = utils.find_depth(TEST_VARS, level=0)
    assert obs == [True, False, False]


def test_find_depth_0_minus():
    obs = utils.find_depth(TEST_VARS, level="0-")
    assert obs == [True, False, False]


def test_find_depth_0_plus():
    obs = utils.find_depth(TEST_VARS, level="0+")
    assert obs == [True, True, True]


def test_find_depth_1():
    obs = utils.find_depth(TEST_VARS, level=1)
    assert obs == [False, True, False]


def test_find_depth_1_minus():
    obs = utils.find_depth(TEST_VARS, level="1-")
    assert obs == [True, True, False]


def test_find_depth_1_plus():
    obs = utils.find_depth(TEST_VARS, level="1+")
    assert obs == [False, True, True]


def test_concat_with_pipe_all():
    obs = utils.concat_with_pipe(TEST_CONCAT_SERIES)
    assert obs == "foo|bar|baz"


def test_concat_with_pipe_exclude_none():
    s = TEST_CONCAT_SERIES.copy()
    s["b"] = None
    obs = utils.concat_with_pipe(s)
    assert obs == "foo|baz"


def test_concat_with_pipe_exclude_nan():
    s = TEST_CONCAT_SERIES.copy()
    s["b"] = np.nan
    obs = utils.concat_with_pipe(s)
    assert obs == "foo|baz"


def test_concat_with_pipe_by_name():
    obs = utils.concat_with_pipe(TEST_CONCAT_SERIES, ["f", "z"])
    assert obs == "foo|baz"


def test_reduce_hierarchy_0():
    assert utils.reduce_hierarchy("foo|bar|baz", 0) == "foo"


def test_reduce_hierarchy_1():
    assert utils.reduce_hierarchy("foo|bar|baz", 1) == "foo|bar"


def test_reduce_hierarchy_neg1():
    assert utils.reduce_hierarchy("foo|bar|baz", -1) == "foo|bar"


def test_reduce_hierarchy_neg2():
    assert utils.reduce_hierarchy("foo|bar|baz", -2) == "foo"


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
    pytest.raises(ValueError, utils.merge_meta, left, right)

    # merge conflict ignoring errors yields expected results
    exp = pd.DataFrame(
        [
            ["model_a", "scen_a", "foo", 1, 2],
            ["model_a", "scen_b", "bar", 2, np.nan],
            ["model_b", "scen_a", "baz", np.nan, 3],
        ],
        columns=META_IDX + ["string", "value", "value2"],
    ).set_index(META_IDX)

    obs = utils.merge_meta(left, right, ignore_conflict=True)
    pdt.assert_frame_equal(exp, obs)


def test_get_variable_components_int():
    assert utils.get_variable_components("foo|bar|baz", 1) == "bar"


def test_get_variable_components_list():
    assert utils.get_variable_components("foo|bar|baz", [1, 2]) == ["bar", "baz"]


def test_get_variable_components_indexError():
    with pytest.raises(IndexError):
        utils.get_variable_components("foo|bar|baz", 3)


def test_get_variable_components_joinTRUE():
    assert utils.get_variable_components("foo|bar|baz", [0, 2], join=True) == "foo|baz"


def test_get_variable_components_joinstr():
    assert utils.get_variable_components("foo|bar|baz", [2, 1], join="_") == "baz_bar"
