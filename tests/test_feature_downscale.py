import pytest
from pyam.testing import assert_iamframe_equal


@pytest.mark.parametrize("variable", (
    ('Primary Energy'),
    (['Primary Energy', 'Primary Energy|Coal']),
))
def test_downscale_region(simple_df, variable):
    simple_df.set_meta([1], name='test')
    regions = ['reg_a', 'reg_b']

    # return as new IamDataFrame
    obs = simple_df.downscale_region(variable, proxy='Population')
    exp = simple_df.filter(variable=variable, region=regions)
    assert_iamframe_equal(exp, obs)

    # append to `self` (after removing to-be-downscaled timeseries)
    inplace = simple_df.filter(variable=variable, region=regions, keep=False)
    inplace.downscale_region(variable, proxy='Population', append=True)
    assert_iamframe_equal(inplace, simple_df)
