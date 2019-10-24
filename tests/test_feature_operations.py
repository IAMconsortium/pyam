import numpy as np
import pandas as pd

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
    res = check_aggregate_df.subtract(check_aggregate_df, "variable", "zero")
    np.testing.assert_array_equal(res.timeseries(), 0)


def test_subtraction(check_aggregate_df):
    tdf = check_aggregate_df.filter(variable="Primary Energy", region="World")
    sdf = check_aggregate_df.filter(variable="Primary Energy|Coal", region="World")
    sub_var_name = "Primary Energy - Primary Energy|Coal"

    join_col = "variable"
    tdf_ts = tdf.timeseries()
    sdf_ts = sdf.timeseries()
    idx = tdf_ts.index.names
    idx_tmp = list(set(idx) - set([join_col]) - {"value"})

    tdf_ts = tdf_ts.reset_index().set_index(idx_tmp).drop(join_col, axis="columns")
    sdf_ts = sdf_ts.reset_index().set_index(idx_tmp).drop(join_col, axis="columns")

    exp = (tdf_ts - sdf_ts).reset_index()
    exp["variable"] = sub_var_name
    exp = IamDataFrame(exp)

    res = tdf.subtract(
        sdf,
        "variable",
        sub_var_name
    )

    pd.testing.assert_frame_equal(exp.timeseries(), res.timeseries(), check_like=True)
