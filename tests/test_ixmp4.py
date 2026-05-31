from unittest.mock import patch

import ixmp4 as ixmp4_module
import pytest
from ixmp4.core.region import Region
from ixmp4.core.unit import Unit

import pyam
from pyam import iiasa, read_ixmp4
from pyam.ixmp4 import read_run
from pyam.testing import assert_iamframe_equal


def test_to_ixmp4_missing_region_raises(test_platform, test_df_year):
    """Writing to platform raises if region not defined"""
    test_df_year.rename(region={"World": "foo"}, inplace=True)
    with pytest.raises(Region.NotFound, match="foo. Use `Platform.regions."):
        test_df_year.to_ixmp4(platform=test_platform)


def test_to_ixmp4_missing_unit_raises(test_platform, test_df_year):
    """Writing to platform raises if unit not defined"""
    test_df_year.rename(unit={"EJ/yr": "foo"}, inplace=True)
    with pytest.raises(Unit.NotFound, match="foo. Use `Platform.units."):
        test_df_year.to_ixmp4(platform=test_platform)


def test_ixmp4_subannual_not_implemented(test_platform, test_df_year):
    """Writing an IamDataFrame with subannual timeslices is not implemented"""

    data = test_df_year.data
    data["subannual"] = "summer-day"
    with pytest.raises(NotImplementedError):
        pyam.IamDataFrame(data).to_ixmp4(platform=test_platform)


def test_ixmp4_mixed_time_domain(test_platform, test_df_mixed):
    """Writing an IamDataFrame with mixed time domain to the platform"""

    # test writing to platform
    test_df_mixed.to_ixmp4(platform=test_platform)

    # read only default scenarios (runs) - version number added as meta indicator
    obs = read_ixmp4(platform=test_platform)
    exp = test_df_mixed.copy()
    exp.set_meta(1, "version")  # add version number added from ixmp4
    assert_iamframe_equal(exp, obs)


def test_ixmp4_integration(test_platform, test_df):
    """Write an IamDataFrame to the platform"""

    # test writing to platform
    test_df.to_ixmp4(platform=test_platform)

    # read only default scenarios (runs) - version number added as meta indicator
    obs = read_ixmp4(platform=test_platform)
    exp = test_df.copy()
    exp.set_meta(1, "version")  # add version number added from ixmp4
    assert_iamframe_equal(exp, obs)

    # make one scenario a non-default scenario, make sure that it is not included
    test_platform.runs.get("model_a", "scen_b").unset_as_default()
    obs = read_ixmp4(platform=test_platform)
    assert_iamframe_equal(exp.filter(scenario="scen_a"), obs)

    # read all scenarios (runs) - version number used as additional index dimension
    obs = read_ixmp4(platform=test_platform, default_only=False)
    data = test_df.data
    data["version"] = 1
    meta = test_df.meta.reset_index()
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

    # add 'version' meta indicator (indicator during ixmp4 roundtrip)
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


def test_ixmp4_empty_result(test_platform):
    with pytest.raises(ValueError, match=r"No scenario data with filters \{.*'foo'\}"):
        read_ixmp4(test_platform, variable="foo")


def test_ixmp4_read_run(test_platform, test_df):
    """Initialize an IamDataFrame from an ixmp4 run"""

    # write to platform
    test_df.to_ixmp4(platform=test_platform)

    # get a run and cast to an IamDataFrame
    run = test_platform.runs.get("model_a", "scen_a")
    obs = read_run(run)

    # assert that the returned object and the original IamDataFrame are equal
    exp = test_df.filter(scenario="scen_a")
    exp.set_meta(1, "version")  # the IamDataFrame read from ixmp4 includes the version
    pyam.assert_iamframe_equal(exp, obs)


def test_ixmp4_read_run_with_filters(test_platform, test_df_year):
    """Read a run with filters"""

    test_df_year.to_ixmp4(platform=test_platform)

    run = test_platform.runs.get("model_a", "scen_a")
    obs = read_run(run, region="World", variable="Primary Energy", year=2010)

    exp = test_df_year.filter(
        scenario="scen_a", region="World", variable="Primary Energy", year=2010
    )
    exp.set_meta(1, "version")
    pyam.assert_iamframe_equal(exp, obs)


def test_read_ixmp4_string_platform(test_platform, test_df_year):
    """read_ixmp4 converts a string platform name to a Platform instance"""

    test_df_year.to_ixmp4(platform=test_platform)

    class FakePlatform:
        def __new__(cls, name):
            return test_platform

    with patch.object(ixmp4_module, "Platform", FakePlatform):
        obs = read_ixmp4(platform="test-platform-name")

    test_df_year.set_meta(1, "version")
    pyam.assert_iamframe_equal(test_df_year, obs)


def test_write_to_ixmp4_string_platform(test_platform, test_df_year):
    """write_to_ixmp4 converts a string platform name to a Platform instance"""

    class FakePlatform:
        def __new__(cls, name):
            return test_platform

    with patch.object(ixmp4_module, "Platform", FakePlatform):
        test_df_year.to_ixmp4(platform="test-platform-name")

    # verify data was written correctly
    obs = read_ixmp4(platform=test_platform)
    test_df_year.set_meta(1, "version")
    pyam.assert_iamframe_equal(test_df_year, obs)


class _PlatformInfo:
    def __init__(self, slug, name):
        self.slug = slug
        self.name = name


class _ManagerPlatforms:
    def __init__(self, platforms):
        self._platforms = platforms

    def list_platforms(self):
        return self._platforms


class _Settings:
    def __init__(self, platforms):
        self._manager_platforms = _ManagerPlatforms(platforms)

    def get_manager_platforms(self):
        return self._manager_platforms


def test_read_iiasa_ixmp4_lookup_by_slug(monkeypatch):
    """read_iiasa should identify ixmp4 platforms via platform slug."""

    monkeypatch.setattr(
        iiasa,
        "Settings",
        lambda: _Settings([_PlatformInfo(slug="my-platform", name="My Platform")]),
    )

    calls = {}

    def _read_ixmp4(platform, default_only=True, **kwargs):
        calls["platform"] = platform
        calls["default_only"] = default_only
        calls["kwargs"] = kwargs
        return "ixmp4-result"

    monkeypatch.setattr(iiasa, "read_ixmp4", _read_ixmp4)

    class _Connection:
        def __init__(self, *args, **kwargs):
            pytest.fail("Connection path should not be used for ixmp4 platform slug")

    monkeypatch.setattr(iiasa, "Connection", _Connection)

    obs = iiasa.read_iiasa(
        "my-platform", default_only=False, meta=True, variable="Primary Energy"
    )

    assert obs == "ixmp4-result"
    assert calls == {
        "platform": "my-platform",
        "default_only": False,
        "kwargs": {"variable": "Primary Energy"},
    }


def test_read_iiasa_does_not_match_platform_name(monkeypatch):
    """Platform matching should not use display name; only slug is valid."""

    monkeypatch.setattr(
        iiasa,
        "Settings",
        lambda: _Settings([_PlatformInfo(slug="my-platform", name="My Platform")]),
    )

    def _read_ixmp4(*args, **kwargs):
        pytest.fail("ixmp4 path should not be used when only platform name matches")

    monkeypatch.setattr(iiasa, "read_ixmp4", _read_ixmp4)

    calls = {}

    class _Connection:
        def __init__(self, name, creds):
            calls["name"] = name
            calls["creds"] = creds

        def query(self, default_only=True, meta=True, **kwargs):
            calls["default_only"] = default_only
            calls["meta"] = meta
            calls["kwargs"] = kwargs
            return "legacy-result"

    monkeypatch.setattr(iiasa, "Connection", _Connection)

    obs = iiasa.read_iiasa(
        "My Platform", default_only=False, meta=False, creds="foo", model="bar"
    )

    assert obs == "legacy-result"
    assert calls == {
        "name": "My Platform",
        "creds": "foo",
        "default_only": False,
        "meta": False,
        "kwargs": {"model": "bar"},
    }
