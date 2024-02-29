import dask
import pytest
from ixmp4.core import Platform
from ixmp4.core.region import RegionModel
from ixmp4.core.unit import UnitModel
from ixmp4.data.backend import SqliteTestBackend

dask.config.set({"dataframe.convert-string": False})


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

    if test_df.time_domain == "year":
        test_df.to_ixmp4(platform=platform)
    else:
        with pytest.raises(NotImplementedError):
            test_df.to_ixmp4(platform=platform)

    # TODO add test for reading data from ixmp4 platform
