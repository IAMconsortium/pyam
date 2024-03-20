import pytest
from ixmp4.core.region import RegionModel
from ixmp4.core.unit import UnitModel

import pyam
from pyam import read_ixmp4
from pyam.testing import assert_iamframe_equal


def test_to_ixmp4_missing_region_raises(test_platform, test_df_year):
    """Writing to platform raises if region not defined"""
    test_df_year.rename(region={"World": "foo"}, inplace=True)
    with pytest.raises(RegionModel.NotFound, match="foo. Use `Platform.regions."):
        test_df_year.to_ixmp4(platform=test_platform)


def test_to_ixmp4_missing_unit_raises(test_platform, test_df_year):
    """Writing to platform raises if unit not defined"""
    test_df_year.rename(unit={"EJ/yr": "foo"}, inplace=True)
    with pytest.raises(UnitModel.NotFound, match="foo. Use `Platform.units."):
        test_df_year.to_ixmp4(platform=test_platform)


def test_ixmp4_time_not_implemented(test_platform, test_df):
    """Writing an IamDataFrame with datetime-data is not implemented"""
    if test_df.time_domain != "year":
        with pytest.raises(NotImplementedError):
            test_df.to_ixmp4(platform=test_platform)


def test_ixmp4_integration(test_platform, test_df_year):
    """Write an IamDataFrame to the platform"""

    # test writing to platform
    test_df_year.to_ixmp4(platform=test_platform)

    # read only default scenarios (runs) - version number added as meta indicator
    obs = read_ixmp4(platform=test_platform)
    exp = test_df_year.copy()
    exp.set_meta(1, "version")  # add version number added from ixmp4
    assert_iamframe_equal(exp, obs)

    # make one scenario a non-default scenario, make sure that it is not included
    test_platform.runs.get("model_a", "scen_b").unset_as_default()
    obs = read_ixmp4(platform=test_platform)
    assert_iamframe_equal(exp.filter(scenario="scen_a"), obs)

    # read all scenarios (runs) - version number used as additional index dimension
    obs = read_ixmp4(platform=test_platform, default_only=False)
    data = test_df_year.data
    data["version"] = 1
    meta = test_df_year.meta.reset_index()
    meta["version"] = 1
    exp = pyam.IamDataFrame(data, meta=meta, index=["model", "scenario", "version"])
    pyam.assert_iamframe_equal(exp, obs)


@pytest.mark.parametrize(
    "filters",
    (
        dict(model="model_a"),
        dict(scenario="scen_a"),
        dict(scenario="*n_a"),
        dict(model="model_a", scenario="scen_a", region="World", variable="* Energy"),
        dict(scenario="scen_a", region="World", variable="Primary Energy", year=2010),
    ),
)
def test_ixmp4_filters(test_platform, test_df_year, filters):
    """Write an IamDataFrame to the platform and read it back with filters"""

    # test writing to platform
    test_df_year.to_ixmp4(platform=test_platform)

    # add 'version' meta indicator (indicator during imp4 roundtrip)
    test_df_year.set_meta(1, "version")

    # read with filters
    assert_iamframe_equal(
        read_ixmp4(test_platform, **filters),
        test_df_year.filter(**filters),
    )


@pytest.mark.parametrize("drop_meta", (True, False))
def test_ixmp4_reserved_columns(test_platform, test_df_year, drop_meta):
    """Make sure that a 'version' column in `meta` is not written to the platform"""

    if drop_meta:
        test_df_year = pyam.IamDataFrame(test_df_year.data)

    # write to platform with a version-number as meta indicator
    test_df_year.set_meta(1, "version")  # add version number added from ixmp4
    test_df_year.to_ixmp4(platform=test_platform)

    # version is not saved to the platform
    if drop_meta:
        assert len(test_platform.runs.get("model_a", "scen_a").meta) == 0
    else:
        assert "version" not in test_platform.runs.get("model_a", "scen_a").meta

    # version is included when reading again from the platform
    assert_iamframe_equal(test_df_year, pyam.read_ixmp4(test_platform))
