import pytest
from ixmp4.core import Platform
from ixmp4.core.region import RegionModel
from ixmp4.core.unit import UnitModel
from ixmp4.data.backend import SqliteTestBackend

import pyam
from pyam import read_ixmp4


def test_to_ixmp4_missing_region_raises(test_df_year):
    """Writing to platform raises if region not defined"""
    platform = Platform(_backend=SqliteTestBackend())
    with pytest.raises(RegionModel.NotFound, match="World. Use `Platform.regions."):
        test_df_year.to_ixmp4(platform=platform)


def test_to_ixmp4_missing_unit_raises(test_df_year):
    """Writing to platform raises if unit not defined"""
    platform = Platform(_backend=SqliteTestBackend())
    platform.regions.create(name="World", hierarchy="common")
    with pytest.raises(UnitModel.NotFound, match="EJ/yr. Use `Platform.units."):
        test_df_year.to_ixmp4(platform=platform)


def test_ixmp4_integration(test_df):
    """Write an IamDataFrame to the platform"""
    platform = Platform(_backend=SqliteTestBackend())
    platform.regions.create(name="World", hierarchy="common")
    platform.units.create(name="EJ/yr")

    if test_df.time_domain != "year":
        with pytest.raises(NotImplementedError):
            test_df.to_ixmp4(platform=platform)
    else:
        # test writing to platform
        test_df.to_ixmp4(platform=platform)

        # read only default scenarios (runs) - version number added as meta indicator
        obs = read_ixmp4(platform=platform)
        exp = test_df.copy()
        exp.set_meta(1, "version")  # add version number added from ixmp4
        pyam.assert_iamframe_equal(exp, obs)

        # read all scenarios (runs) - version number used as additional index dimension
        obs = read_ixmp4(platform=platform, default_only=False)
        data = test_df.data
        data["version"] = 1
        meta = test_df.meta.reset_index()
        meta["version"] = 1
        exp = pyam.IamDataFrame(data, meta=meta, index=["model", "scenario", "version"])
        pyam.assert_iamframe_equal(exp, obs)
