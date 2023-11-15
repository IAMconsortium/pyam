import ixmp4
from ixmp4.core.region import RegionModel
from ixmp4.core.unit import UnitModel

def write_to_ixmp4(df, platform: ixmp4.Platform):
    """Save all scenarios as new default runs in an ixmp4 platform database instance

    Parameters
    ----------
    df : pyam.IamDataFrame
        The IamDataFrame instance with scenario data
    platform : :class:`ixmp4.Platform` or str
        The ixmp4 platform database instance to which the scenario data is saved
    """
    if not isinstance(platform, ixmp4.Platform):
        platform = ixmp4.Platform(platform)

    # TODO: implement a try-except to roll back changes if any error writing to platform
    # depends on https://github.com/iiasa/ixmp4/issues/29
    # quickfix: ensure that units and regions exist before writing

    platform_regions = platform.regions.tabulate().name.values
    if missing := [r for r in df.region if r not in platform_regions]:
        raise RegionModel.NotFound(
            ", ".join(missing) +
            ". Use `Platform.regions.create()` to add the missing region(s)."
        )

    platform_units = platform.units.tabulate().name.values
    if missing := [u for u in df.unit if u not in platform_units]:
        raise UnitModel.NotFound(
            ", ".join(missing) +
            ". Use `Platform.units.create()` to add the missing unit(s)."
    )

    for model, scenario in df.index:
        _df = df.filter(model=model, scenario=scenario)

        run = platform.Run(model=model, scenario=scenario, version="new")
        run.iamc.add(_df.data)
        for key, value in dict(_df.meta.iloc[0]).items():
            run.meta[key] = value
        run.set_as_default()