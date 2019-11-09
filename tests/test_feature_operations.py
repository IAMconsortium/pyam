import re

import numpy as np
import pandas as pd
import pytest

from pyam import IamDataFrame

# list of todo's here:
#   - what happens if indexes are different (NotImplementedError?)

# list of todo's for future PRs:
#   - subtracting Series
#   - subtracting floats
#   - subtracting vectors
#   - subtracting arrays
#   - other operations e.g. multiplication, addition, division

def test_subtraction_with_self(check_aggregate_df):
    tdf = check_aggregate_df.filter(variable="Primary Energy")
    res = tdf.subtract(tdf, "variable", "zero")
    np.testing.assert_array_equal(res.data.value, 0)


def test_subtraction_duplicate_entries_self_error(check_aggregate_df):
    # is this how we want this to work?
    error_msg = re.escape("`self` contains more than one `variable`")
    with pytest.raises(ValueError, match=error_msg):
        check_aggregate_df.subtract(check_aggregate_df, "variable", "error")


def test_subtraction_duplicate_entries_other_error(check_aggregate_df):
    # is this how we want this to work?
    error_msg = re.escape("`other` contains more than one `variable`")
    with pytest.raises(ValueError, match=error_msg):
        check_aggregate_df.filter(variable="Primary Energy").subtract(
            check_aggregate_df, "variable", "error"
        )


def test_subtraction(check_aggregate_df):
    tdf = check_aggregate_df.filter(variable="Primary Energy")
    sdf = check_aggregate_df.filter(variable="Primary Energy|Coal")
    sub_var_name = "Primary Energy - Primary Energy|Coal"

    join_col = "variable"
    tdf_ts = tdf.timeseries()
    sdf_ts = sdf.timeseries()
    idx = tdf_ts.index.names
    idx_tmp = list(set(idx) - set([join_col]) - {"value"})

    tdf_ts = tdf_ts.reset_index().set_index(idx_tmp).drop(join_col, axis="columns")
    sdf_ts = sdf_ts.reset_index().set_index(idx_tmp).drop(join_col, axis="columns")

    exp = (tdf_ts - sdf_ts).reset_index()
    exp[join_col] = sub_var_name
    exp = IamDataFrame(exp)

    res = tdf.subtract(sdf, "variable", sub_var_name)

    pd.testing.assert_frame_equal(exp.timeseries(), res.timeseries(), check_like=True)


def test_subtraction_scenarios(check_aggregate_df):
    tvs = [
        "Emissions|CO2",
        "Emissions|CO2|Cars",
        "Emissions|CO2|Tar",
        "Primary Energy",
        "Primary Energy|Coal",
        "Primary Energy|Gas",
    ]
    tdf = check_aggregate_df.filter(model="MSG-GLB", scenario="a_scen", variable=tvs)
    sdf = check_aggregate_df.filter(model="MSG-GLB", scenario="a_scen_2", variable=tvs)
    sub_scenario_name = "a_scen - a_scen_2"

    join_col = "scenario"
    tdf_ts = tdf.timeseries()
    sdf_ts = sdf.timeseries()
    idx = tdf_ts.index.names
    idx_tmp = list(set(idx) - set([join_col]) - {"value"})

    tdf_ts = tdf_ts.reset_index().set_index(idx_tmp).drop(join_col, axis="columns")
    sdf_ts = sdf_ts.reset_index().set_index(idx_tmp).drop(join_col, axis="columns")

    exp = (tdf_ts - sdf_ts).reset_index()
    exp[join_col] = sub_scenario_name
    exp = IamDataFrame(exp)

    res = tdf.subtract(sdf, "scenario", sub_scenario_name)

    pd.testing.assert_frame_equal(exp.timeseries(), res.timeseries(), check_like=True)


@pytest.mark.parametrize("failing_type", (
    pd.Series([1, 2, 3]),
    2.3,
    1,
    np.array([2, 4, 1]),
    np.array([[2, 4, 1], [2.3, -1, 0.3]]),
))
def test_failing_types_error(test_df, failing_type):
    with pytest.raises(NotImplementedError):
        test_df.subtract(failing_type, "variable", "irrelevant")


@pytest.mark.parametrize("ignore_meta_conflict", (True, False))
def test_different_meta_res(test_df, ignore_meta_conflict):
    tdf = test_df.filter(variable="Primary Energy")
    odf = tdf.copy()
    tdf.set_meta("value", "extra_col")
    odf.set_meta("conflict value", "extra_col")

    if ignore_meta_conflict:
        res = tdf.subtract(
            odf, "variable", "irrelevant", ignore_meta_conflict=ignore_meta_conflict
        )
        pd.testing.assert_frame_equal(res.meta, tdf.meta)
    else:
        error_msg = re.escape(
            "conflict in `meta` for scenarios [('model_a', 'scen_a')]"
        )
        with pytest.raises(ValueError, match=error_msg):
            tdf.subtract(odf, "variable", "irrelevant")


@pytest.mark.parametrize("ignore_meta_conflict", (True, False))
def test_meta_conflict(test_df, ignore_meta_conflict):
    tdf = test_df.filter(variable="Primary Energy")
    odf = tdf.copy()
    tdf.set_meta("value", "extra_col")
    odf.set_meta("conflict value", "extra_col")

    if ignore_meta_conflict:
        res = tdf.subtract(
            odf, "variable", "irrelevant", ignore_meta_conflict=ignore_meta_conflict
        )
        pd.testing.assert_frame_equal(res.meta, tdf.meta)
    else:
        error_msg = re.escape(
            "conflict in `meta` for scenarios [('model_a', 'scen_a')]"
        )
        with pytest.raises(ValueError, match=error_msg):
            tdf.subtract(odf, "variable", "irrelevant")
