import datetime as dt

import numpy as np
import pandas as pd
from pyam import compare, IAMC_IDX


def test_compare(test_df):
    clone = test_df.copy()
    clone._data.iloc[0] = 2
    clone.rename(variable={"Primary Energy|Coal": "Primary Energy|Gas"}, inplace=True)

    obs = compare(test_df, clone, left_label="test_df", right_label="clone")

    exp = pd.DataFrame(
        [
            ["Primary Energy", "EJ/yr", dt.datetime(2005, 6, 17), 1, 2],
            ["Primary Energy|Coal", "EJ/yr", dt.datetime(2005, 6, 17), 0.5, np.nan],
            ["Primary Energy|Coal", "EJ/yr", dt.datetime(2010, 7, 21), 3, np.nan],
            ["Primary Energy|Gas", "EJ/yr", dt.datetime(2005, 6, 17), np.nan, 0.5],
            ["Primary Energy|Gas", "EJ/yr", dt.datetime(2010, 7, 21), np.nan, 3],
        ],
        columns=["variable", "unit", "time", "test_df", "clone"],
    )
    exp["model"] = "model_a"
    exp["scenario"] = "scen_a"
    exp["region"] = "World"
    time_col = "time"
    if test_df.time_col == "year":
        exp["year"] = exp["time"].apply(lambda x: x.year)
        exp = exp.drop("time", axis="columns")
        time_col = "year"
    else:
        obs = obs.reset_index()
        obs.time = obs.time.dt.normalize()
        obs = obs.set_index(IAMC_IDX + [time_col])

    exp = exp.set_index(IAMC_IDX + [time_col])
    pd.testing.assert_frame_equal(obs, exp)
