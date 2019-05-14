import copy
import datetime as dt


import numpy as np
import pandas as pd
from pyam import compare, IAMC_IDX


def test_compare(meta_df):
    clone = copy.deepcopy(meta_df)
    clone.data.iloc[0, clone.data.columns.get_loc('value')] = 2
    clone.rename({'variable': {'Primary Energy|Coal': 'Primary Energy|Gas'}},
                 inplace=True)

    obs = compare(meta_df, clone, right_label='meta_df', left_label='clone')

    exp = pd.DataFrame([
        ['Primary Energy', 'EJ/y', dt.datetime(2005, 6, 17), 2, 1],
        ['Primary Energy|Coal', 'EJ/y', dt.datetime(2005, 6, 17), np.nan, 0.5],
        ['Primary Energy|Coal', 'EJ/y', dt.datetime(2010, 7, 21), np.nan, 3],
        ['Primary Energy|Gas', 'EJ/y', dt.datetime(2005, 6, 17), 0.5, np.nan],
        ['Primary Energy|Gas', 'EJ/y', dt.datetime(2010, 7, 21), 3, np.nan],
    ],
        columns=['variable', 'unit', 'time', 'meta_df', 'clone'],
    )
    exp['model'] = 'model_a'
    exp['scenario'] = 'scen_a'
    exp['region'] = 'World'
    time_col = 'time'
    if "year" in meta_df.data.columns:
        exp["year"] = exp["time"].apply(lambda x: x.year)
        exp = exp.drop("time", axis="columns")
        time_col = 'year'

    exp = exp.set_index(IAMC_IDX + [time_col])

    pd.testing.assert_frame_equal(obs, exp)
