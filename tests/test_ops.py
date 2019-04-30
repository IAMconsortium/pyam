import pandas as pd

def test_subtract(meta_df):
    x = meta_df.filter(variable='Primary Energy', scenario='scen_a')
    y = meta_df.filter(variable='Primary Energy|Coal', scenario='scen_a')
    obs = (x-y).data

    exp = x.copy().data
    exp['variable'] = 'Primary Energy - Primary Energy|Coal'
    exp['value'] = [0.5, 3.0]

    pd.testing.assert_frame_equal(obs, exp)

