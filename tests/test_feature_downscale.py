import pytest
from pyam.testing import assert_iamframe_equal


def test_downscale_region_without_method_raises(simple_df):
    # downscale_region without specifying a method
    variable = "Primary Energy"
    pytest.raises(ValueError, simple_df.downscale_region, variable=variable)


def test_downscale_region_with_multiple_methods_raises(simple_df):
    # downscale_region with specifying both weight and proxy raises

    # create weighting dataframe
    weight_df = simple_df.filter(variable="Population").data.pivot_table(
        index="region", columns=simple_df.time_col, values="value"
    )

    # call downscale_region with both proxy and weight
    variable = "Primary Energy"
    pytest.raises(
        ValueError,
        simple_df.downscale_region,
        variable=variable,
        proxy="Population",
        weight=weight_df,
    )


@pytest.mark.parametrize(
    "variable",
    (
        ("Primary Energy"),
        (["Primary Energy", "Primary Energy|Coal"]),
    ),
)
def test_downscale_region_with_proxy(simple_df, variable):
    simple_df.set_meta([1], name="test")
    regions = ["reg_a", "reg_b"]

    # return as new IamDataFrame
    obs = simple_df.downscale_region(variable, proxy="Population")
    exp = simple_df.filter(variable=variable, region=regions)
    assert_iamframe_equal(exp, obs)

    # append to `self` (after removing to-be-downscaled timeseries)
    inplace = simple_df.filter(variable=variable, region=regions, keep=False)
    inplace.downscale_region(variable, proxy="Population", append=True)
    assert_iamframe_equal(inplace, simple_df)


@pytest.mark.parametrize(
    "variable, index",
    (
        ("Primary Energy", ["region"]),
        (["Primary Energy", "Primary Energy|Coal"], ["region"]),
        (["Primary Energy", "Primary Energy|Coal"], ["model", "region"]),
    ),
)
def test_downscale_region_with_weight(simple_df, variable, index):
    simple_df.set_meta([1], name="test")
    regions = ["reg_a", "reg_b"]

    # create weighting dataframe
    weight_df = simple_df.filter(
        variable="Population", region=regions
    ).data.pivot_table(index=index, columns=simple_df.time_col, values="value")

    # return as new IamDataFrame
    obs = simple_df.downscale_region(variable, weight=weight_df)
    exp = simple_df.filter(variable=variable, region=regions)
    assert_iamframe_equal(exp, obs)

    # append to `self` (after removing to-be-downscaled timeseries)
    inplace = simple_df.filter(variable=variable, region=regions, keep=False)
    inplace.downscale_region(variable, weight=weight_df, append=True)
    assert_iamframe_equal(inplace, simple_df)


@pytest.mark.parametrize(
    "variable, index",
    (
        ("Primary Energy", ["region"]),
        (["Primary Energy", "Primary Energy|Coal"], ["model", "region"]),
    ),
)
def test_downscale_region_with_weight_subregions(simple_df, variable, index):
    simple_df.set_meta([1], name="test")
    regions = ["reg_a", "reg_b"]

    # create weighting dataframe with an extra "duplicate" region
    weight_df = (
        simple_df.filter(variable="Population")
        .rename(region={"reg_a": "duplicate"}, append=True)  # add extra region
        .data.pivot_table(index=index, columns=simple_df.time_col, values="value")
    )

    # return as new IamDataFrame
    ds_args = dict(variable=variable, weight=weight_df, subregions=regions)
    obs = simple_df.downscale_region(**ds_args)
    exp = simple_df.filter(variable=variable, region=regions)
    assert_iamframe_equal(exp, obs)

    # append to `self` (after removing to-be-downscaled timeseries)
    inplace = simple_df.filter(variable=variable, region=regions, keep=False)
    inplace.downscale_region(**ds_args, append=True)
    assert_iamframe_equal(inplace, simple_df)
