import pytest

import numpy as np
import pandas as pd
import pyam


@pytest.mark.parametrize("variable", (
    ('Primary Energy'),
    (['Primary Energy', 'Primary Energy|Coal']),
))
def test_downscale_region(aggregate_df, variable):
    df = aggregate_df
    df.set_meta([1], name='test')

    regions = ['reg_a', 'reg_b']

    # return as new IamDataFrame
    obs_df = df.downscale_region(variable, proxy='Population')
    exp_df = df.filter(variable=variable, region=regions)
    assert pyam.compare(obs_df, exp_df).empty
    pd.testing.assert_frame_equal(obs_df.meta, exp_df.meta)

    # append to `self` (after removing to-be-downscaled timeseries)
    inplace_df = df.filter(variable=variable, region=regions, keep=False)
    inplace_df.downscale_region(variable, proxy='Population', append=True)
    assert pyam.compare(inplace_df, df).empty
    pd.testing.assert_frame_equal(inplace_df.meta, df.meta)