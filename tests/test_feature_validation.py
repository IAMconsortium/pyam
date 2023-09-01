import pandas as pd
import pandas.testing as pdt
import pytest

from pyam import IamDataFrame, validate, categorize
from pyam.utils import IAMC_IDX

from .conftest import TEST_YEARS


# add row for `Primary Energy|Gas` to test that all permutations
DATA_GAS = pd.DataFrame(
    [
        ["model_a", "scen_a", "World", "Primary Energy|Gas", "EJ/yr", 0.5, 3],
        ["model_a", "scen_b", "World", "Primary Energy|Gas", "EJ/yr", 1, 2],
    ],
    columns=IAMC_IDX + TEST_YEARS,
)


@pytest.mark.parametrize(
    "kwargs",
    (
        dict(),
        dict(variable="Primary Energy"),
        dict(variable="Primary Energy", year=[2005, 2010]),
        dict(variable=["Primary Energy"], year=[2005, 2010]),
        dict(variable=["Primary Energy", "Primary Energy|Gas"], year=[2005, 2010]),
    ),
)
def test_require_data_pass(test_df_year, kwargs):
    # check that IamDataFrame with all required data returns None
    df = test_df_year.append(IamDataFrame(DATA_GAS))
    assert df.require_data(**kwargs) is None


@pytest.mark.parametrize(
    "kwargs",
    (
        dict(variable="Primary Energy|Coal"),
        dict(variable="Primary Energy", year=[2005, 2010]),
        dict(variable=["Primary Energy"], year=[2005, 2010]),
        dict(variable=["Primary Energy", "Primary Energy|Gas"], year=[2005, 2010]),
    ),
)
@pytest.mark.parametrize("exclude_on_fail", (False, True))
def test_require_data(test_df_year, kwargs, exclude_on_fail):
    # check different ways of failing when not all required data is present

    test_df_year._data = test_df_year._data[0:5]  # remove value for scen_b & 2010
    df = test_df_year.append(IamDataFrame(DATA_GAS))

    obs = df.require_data(**kwargs, exclude_on_fail=exclude_on_fail)

    exp = pd.DataFrame([["model_a", "scen_b"]], columns=["model", "scenario"])
    # add parametrization-dependent columns to expected output
    if kwargs["variable"] == "Primary Energy|Coal":
        exp["variable"] = ["Primary Energy|Coal"]
    else:
        exp["variable"] = ["Primary Energy"]
        exp["year"] = [2010]

    pdt.assert_frame_equal(obs, exp)

    if exclude_on_fail:
        assert list(df.exclude) == [False, True]
    else:
        assert list(df.exclude) == [False, False]


def test_validate_pass(test_df):
    obs = test_df.validate({"Primary Energy": {"up": 10}}, exclude_on_fail=True)
    assert obs is None
    assert list(test_df.exclude) == [False, False]  # none excluded


def test_validate_nonexisting(test_df):
    # checking that a scenario with no relevant value does not fail validation
    obs = test_df.validate({"Primary Energy|Coal": {"up": 2}}, exclude_on_fail=True)
    # checking that the return-type is correct
    pdt.assert_frame_equal(obs, test_df.data[3:4].reset_index(drop=True))
    # scenario with failed validation excluded, scenario with no value passes
    assert list(test_df.exclude) == [True, False]


def test_validate_up(test_df):
    # checking that the return-type is correct
    obs = test_df.validate({"Primary Energy": {"up": 6.5}})
    pdt.assert_frame_equal(obs, test_df.data[5:6].reset_index(drop=True))
    assert list(test_df.exclude) == [False, False]

    # checking exclude on fail
    obs = test_df.validate({"Primary Energy": {"up": 6.5}}, exclude_on_fail=True)
    pdt.assert_frame_equal(obs, test_df.data[5:6].reset_index(drop=True))
    assert list(test_df.exclude) == [False, True]


def test_validate_lo(test_df):
    # checking that the return-type is correct
    obs = test_df.validate({"Primary Energy": {"up": 8, "lo": 2}})
    pdt.assert_frame_equal(obs, test_df.data[0:1].reset_index(drop=True))
    assert list(test_df.exclude) == [False, False]

    # checking exclude on fail
    obs = test_df.validate({"Primary Energy": {"up": 8, "lo": 2}}, exclude_on_fail=True)
    pdt.assert_frame_equal(obs, test_df.data[0:1].reset_index(drop=True))
    assert list(test_df.exclude) == [True, False]


def test_validate_both(test_df):
    # checking that the return-type is correct
    obs = test_df.validate({"Primary Energy": {"up": 6.5, "lo": 2}})
    pdt.assert_frame_equal(obs, test_df.data[0:6:5].reset_index(drop=True))
    assert list(test_df.exclude) == [False, False]

    # checking exclude on fail
    obs = test_df.validate(
        {"Primary Energy": {"up": 6.5, "lo": 2}}, exclude_on_fail=True
    )
    pdt.assert_frame_equal(obs, test_df.data[0:6:5].reset_index(drop=True))
    assert list(test_df.exclude) == [True, True]


def test_validate_year(test_df):
    # checking that the year filter works as expected
    obs = test_df.validate({"Primary Energy": {"up": 6, "year": 2005}})
    assert obs is None

    # checking that the return-type is correct
    obs = test_df.validate({"Primary Energy": {"up": 6, "year": 2010}})
    pdt.assert_frame_equal(obs, test_df.data[5:6].reset_index(drop=True))
    assert list(test_df.exclude) == [False, False]

    # checking exclude on fail
    obs = test_df.validate(
        {"Primary Energy": {"up": 6, "year": 2010}}, exclude_on_fail=True
    )
    pdt.assert_frame_equal(obs, test_df.data[5:6].reset_index(drop=True))
    assert list(test_df.exclude) == [False, True]


def test_validate_top_level(test_df):
    obs = validate(
        test_df,
        criteria={"Primary Energy": {"up": 6}},
        exclude_on_fail=True,
        variable="Primary Energy",
    )
    pdt.assert_frame_equal(obs, test_df.data[5:6].reset_index(drop=True))
    assert list(test_df.exclude) == [False, True]


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
