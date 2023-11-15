import pytest
from ixmp4.core import Platform
from ixmp4.data.backend import SqliteTestBackend
from ixmp4.core.region import RegionModel
from ixmp4.core.unit import UnitModel


def test_to_ixmp4_missing_region_raises(test_df_year):
    """Writing to platform raises if region not defined"""
    platform = Platform(_backend=SqliteTestBackend())
    with pytest.raises(RegionModel.NotFound):
        test_df_year.to_ixmp4(platform=platform)


def test_to_ixmp4_missing_unit_raises(test_df_year):
    """Writing to platform raises if unit not defined"""
    platform = Platform(_backend=SqliteTestBackend())
    platform.regions.create(name="World", hierarchy="common")
    with pytest.raises(UnitModel.NotFound):
        test_df_year.to_ixmp4(platform=platform)
