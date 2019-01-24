import copy
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
        ['Primary Energy', 'EJ/y', 2005, 2, 1],
        ['Primary Energy|Coal', 'EJ/y', 2005, np.nan, 0.5],
        ['Primary Energy|Coal', 'EJ/y', 2010, np.nan, 3],
        ['Primary Energy|Gas', 'EJ/y', 2005, 0.5, np.nan],
        ['Primary Energy|Gas', 'EJ/y', 2010, 3, np.nan],
    ],
        columns=['variable', 'unit', 'year', 'meta_df', 'clone'],
    )
    exp['model'] = 'a_model'
    exp['scenario'] = 'a_scenario'
    exp['region'] = 'World'
    exp = exp.set_index(IAMC_IDX + ['year'])

    pd.testing.assert_frame_equal(obs, exp)
