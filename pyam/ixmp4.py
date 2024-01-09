import ixmp4
from ixmp4.core.region import RegionModel
from ixmp4.core.unit import UnitModel


def write_to_ixmp4(platform: ixmp4.Platform, df):
    """Save all scenarios as new default runs in an ixmp4 platform database instance

    Parameters
    ----------
    platform : :class:`ixmp4.Platform` or str
        The ixmp4 platform database instance to which the scenario data is saved
    df : pyam.IamDataFrame
        The IamDataFrame instance with scenario data
    """
    if not isinstance(platform, ixmp4.Platform):
        platform = ixmp4.Platform(platform)

    # TODO: implement a try-except to roll back changes if any error writing to platform
    # depends on https://github.com/iiasa/ixmp4/issues/29
    # quickfix: ensure that units and regions exist before writing
    for dimension, values, model in [
        ("regions", df.region, RegionModel),
        ("units", df.unit, UnitModel),
    ]:
        platform_values = platform.__getattribute__(dimension).tabulate().name.values
        if missing := [i for i in values if i not in platform_values]:
            raise model.NotFound(
                ", ".join(missing)
                + f". Use `Platform.{dimension}.create()` to add the missing {dimension}."
            )

    for model, scenario in df.index:
        _df = df.filter(model=model, scenario=scenario)

        run = platform.runs.create(model=model, scenario=scenario)
        run.iamc.add(_df.data)
        run.meta = dict(_df.meta.iloc[0])
        run.set_as_default()
