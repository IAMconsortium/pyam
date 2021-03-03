import pandas as pd
import pandas.testing as pdt
from pyam import IamDataFrame, validate, categorize, require_variable, META_IDX


def test_require_variable_pass(test_df):
    # checking that the return-type is correct
    obs = test_df.require_variable(variable="Primary Energy", exclude_on_fail=True)
    assert obs is None
    assert list(test_df["exclude"]) == [False, False]


def test_require_variable(test_df):
    exp = pd.DataFrame([["model_a", "scen_b"]], columns=META_IDX)

    # checking that the return-type is correct
    obs = test_df.require_variable(variable="Primary Energy|Coal")
    pdt.assert_frame_equal(obs, exp)
    assert list(test_df["exclude"]) == [False, False]

    # checking exclude on fail
    obs = test_df.require_variable(variable="Primary Energy|Coal", exclude_on_fail=True)
    pdt.assert_frame_equal(obs, exp)
    assert list(test_df["exclude"]) == [False, True]


def test_require_variable_top_level(test_df):
    exp = pd.DataFrame([["model_a", "scen_b"]], columns=META_IDX)

    # checking that the return-type is correct
    obs = require_variable(test_df, variable="Primary Energy|Coal")
    pdt.assert_frame_equal(obs, exp)
    assert list(test_df["exclude"]) == [False, False]

    # checking exclude on fail
    obs = require_variable(
        test_df, variable="Primary Energy|Coal", exclude_on_fail=True
    )
    pdt.assert_frame_equal(obs, exp)
    assert list(test_df["exclude"]) == [False, True]


def test_require_variable_year_list(test_df):
    # drop first data point
    df = IamDataFrame(test_df.data[1:])
    # checking for variables that have data for ANY of the years in the list
    obs = df.require_variable(variable="Primary Energy", year=[2005, 2010])
    assert obs is None

    # checking for variables that have data for ALL of the years in the list
    df = IamDataFrame(test_df.data[1:])
    exp = pd.DataFrame([["model_a", "scen_a"]], columns=META_IDX)

    obs = df.require_variable(variable="Primary Energy", year=[2005])
    pdt.assert_frame_equal(obs, exp)


def test_validate_pass(test_df):
    obs = test_df.validate({"Primary Energy": {"up": 10}}, exclude_on_fail=True)
    assert obs is None
    assert list(test_df["exclude"]) == [False, False]  # none excluded


def test_validate_nonexisting(test_df):
    # checking that a scenario with no relevant value does not fail validation
    obs = test_df.validate({"Primary Energy|Coal": {"up": 2}}, exclude_on_fail=True)
    # checking that the return-type is correct
    pdt.assert_frame_equal(obs, test_df.data[3:4].reset_index(drop=True))
    # scenario with failed validation excluded, scenario with no value passes
    assert list(test_df["exclude"]) == [True, False]


def test_validate_up(test_df):
    # checking that the return-type is correct
    obs = test_df.validate({"Primary Energy": {"up": 6.5}})
    pdt.assert_frame_equal(obs, test_df.data[5:6].reset_index(drop=True))
    assert list(test_df["exclude"]) == [False, False]

    # checking exclude on fail
    obs = test_df.validate({"Primary Energy": {"up": 6.5}}, exclude_on_fail=True)
    pdt.assert_frame_equal(obs, test_df.data[5:6].reset_index(drop=True))
    assert list(test_df["exclude"]) == [False, True]


def test_validate_lo(test_df):
    # checking that the return-type is correct
    obs = test_df.validate({"Primary Energy": {"up": 8, "lo": 2}})
    pdt.assert_frame_equal(obs, test_df.data[0:1].reset_index(drop=True))
    assert list(test_df["exclude"]) == [False, False]

    # checking exclude on fail
    obs = test_df.validate({"Primary Energy": {"up": 8, "lo": 2}}, exclude_on_fail=True)
    pdt.assert_frame_equal(obs, test_df.data[0:1].reset_index(drop=True))
    assert list(test_df["exclude"]) == [True, False]


def test_validate_both(test_df):
    # checking that the return-type is correct
    obs = test_df.validate({"Primary Energy": {"up": 6.5, "lo": 2}})
    pdt.assert_frame_equal(obs, test_df.data[0:6:5].reset_index(drop=True))
    assert list(test_df["exclude"]) == [False, False]

    # checking exclude on fail
    obs = test_df.validate(
        {"Primary Energy": {"up": 6.5, "lo": 2}}, exclude_on_fail=True
    )
    pdt.assert_frame_equal(obs, test_df.data[0:6:5].reset_index(drop=True))
    assert list(test_df["exclude"]) == [True, True]


def test_validate_year(test_df):
    # checking that the year filter works as expected
    obs = test_df.validate({"Primary Energy": {"up": 6, "year": 2005}})
    assert obs is None

    # checking that the return-type is correct
    obs = test_df.validate({"Primary Energy": {"up": 6, "year": 2010}})
    pdt.assert_frame_equal(obs, test_df.data[5:6].reset_index(drop=True))
    assert list(test_df["exclude"]) == [False, False]

    # checking exclude on fail
    obs = test_df.validate(
        {"Primary Energy": {"up": 6, "year": 2010}}, exclude_on_fail=True
    )
    pdt.assert_frame_equal(obs, test_df.data[5:6].reset_index(drop=True))
    assert list(test_df["exclude"]) == [False, True]


def test_validate_top_level(test_df):
    obs = validate(
        test_df,
        criteria={"Primary Energy": {"up": 6}},
        exclude_on_fail=True,
        variable="Primary Energy",
    )
    pdt.assert_frame_equal(obs, test_df.data[5:6].reset_index(drop=True))
    assert list(test_df["exclude"]) == [False, True]


def test_category_none(test_df):
    test_df.categorize("category", "Testing", {"Primary Energy": {"up": 0.8}})
    assert "category" not in test_df.meta.columns


def test_category_pass(test_df):
    dct = {
        "model": ["model_a", "model_a"],
        "scenario": ["scen_a", "scen_b"],
        "category": ["foo", None],
    }
    exp = pd.DataFrame(dct).set_index(["model", "scenario"])["category"]

    test_df.categorize("category", "foo", {"Primary Energy": {"up": 6, "year": 2010}})
    obs = test_df["category"]
    pd.testing.assert_series_equal(obs, exp)


def test_category_top_level(test_df):
    dct = {
        "model": ["model_a", "model_a"],
        "scenario": ["scen_a", "scen_b"],
        "category": ["foo", None],
    }
    exp = pd.DataFrame(dct).set_index(["model", "scenario"])["category"]

    categorize(
        test_df,
        "category",
        "foo",
        criteria={"Primary Energy": {"up": 6, "year": 2010}},
        variable="Primary Energy",
    )
    obs = test_df["category"]
    pd.testing.assert_series_equal(obs, exp)
