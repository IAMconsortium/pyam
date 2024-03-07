import logging

import ixmp4
import pandas as pd
from ixmp4.core.region import RegionModel
from ixmp4.core.unit import UnitModel

logger = logging.getLogger(__name__)


def read_ixmp4(platform: ixmp4.Platform | str, default_only: bool = True):
    """Read scenario runs from an ixmp4 platform database instance

    Parameters
    ----------
    platform : :class:`ixmp4.Platform` or str
        The ixmp4 platform database instance to which the scenario data is saved
    default_only : :class:`bool`, optional
        Read only default runs
    """
    from pyam import IamDataFrame

    if not isinstance(platform, ixmp4.Platform):
        platform = ixmp4.Platform(platform)

    data = platform.iamc.tabulate(run={"default_only": default_only})
    meta = platform.meta.tabulate(run={"default_only": default_only})

    # if default-only, simplify to standard IAMC index, add `version` as meta indicator
    if default_only:
        index = ["model", "scenario"]
        data.drop(columns="version", inplace=True)
        meta_version = (
            meta[["model", "scenario", "version"]]
            .drop_duplicates()
            .rename(columns={"version": "value"})
        )
        meta_version["key"] = "version"
        meta = pd.concat([meta.drop(columns="version"), meta_version])
    else:
        index = ["model", "scenario", "version"]

    return IamDataFrame(data, meta=meta, index=index)


def write_to_ixmp4(platform: ixmp4.Platform | str, df):
    """Save all scenarios as new default runs in an ixmp4 platform database instance

    Parameters
    ----------
    platform : :class:`ixmp4.Platform` or str
        The ixmp4 platform database instance to which the scenario data is saved
    df : pyam.IamDataFrame
        The IamDataFrame instance with scenario data
    """
    if df.time_domain != "year":
        raise NotImplementedError("Only time_domain='year' is supported for now")

    if not isinstance(platform, ixmp4.Platform):
        platform = ixmp4.Platform(platform)

    # TODO: implement a try-except to roll back changes if any error writing to platform
    # depends on https://github.com/iiasa/ixmp4/issues/29
    # quickfix: ensure that units and regions exist before writing
    for dimension, values, model in [
        ("regions", df.region, RegionModel),
        ("units", df.unit, UnitModel),
    ]:
        platform_values = getattr(platform, dimension).tabulate().name.values
        if missing := set(values).difference(platform_values):
            raise model.NotFound(
                ", ".join(missing)
                + f". Use `Platform.{dimension}.create()` to add the missing "
                f"{dimension}."
            )

    # The "version" meta-indicator should not be written to the database
    if "version" in df.meta.columns:
        logger.warning(
            "The `meta.version` column will be dropped when writing to the ixmp4 database."
        )
        meta = df.meta.drop(columns="version")
    else:
        meta = df.meta.copy()

    # Create runs and add IAMC timeseries data and meta indicators
    for model, scenario in df.index:
        _df = df.filter(model=model, scenario=scenario)

        run = platform.runs.create(model=model, scenario=scenario)
        run.iamc.add(_df.data)
        run.meta = dict(meta.loc[(model, scenario)])
        run.set_as_default()
