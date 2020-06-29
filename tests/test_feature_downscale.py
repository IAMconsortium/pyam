import pytest
from pyam.testing import assert_iamframe_equal


@pytest.mark.parametrize("variable", (
    ('Primary Energy'),
    (['Primary Energy', 'Primary Energy|Coal']),
))
def test_downscale_region_with_proxy(simple_df, variable):
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


@pytest.mark.parametrize("variable, index", (
    ('Primary Energy', ['region']),
    (['Primary Energy', 'Primary Energy|Coal'], ['region']),
    (['Primary Energy', 'Primary Energy|Coal'], ['model', 'region']),
))
def test_downscale_region_with_weight(simple_df, variable, index):
    simple_df.set_meta([1], name='test')
    regions = ['reg_a', 'reg_b']

    # create weighting dataframe
    weight_df = (
        simple_df.filter(variable='Population', region=regions).data
        .pivot_table(index=index, columns=simple_df.time_col,
                     values='value')
    )

    # return as new IamDataFrame
    obs = simple_df.downscale_region(variable, weight=weight_df)
    exp = simple_df.filter(variable=variable, region=regions)
    assert_iamframe_equal(exp, obs)

    # append to `self` (after removing to-be-downscaled timeseries)
    inplace = simple_df.filter(variable=variable, region=regions, keep=False)
    inplace.downscale_region(variable, weight=weight_df, append=True)
    assert_iamframe_equal(inplace, simple_df)