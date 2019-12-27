import pytest
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
    obs = df.downscale_region(variable, proxy='Population')
    exp = df.filter(variable=variable, region=regions)
    assert exp.equals(obs)

    # append to `self` (after removing to-be-downscaled timeseries)
    inplace = df.filter(variable=variable, region=regions, keep=False)
    inplace.downscale_region(variable, proxy='Population', append=True)
    assert inplace.equals(df)
