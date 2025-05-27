import logging

import ixmp4
import numpy as np
import pandas as pd
from ixmp4.core.region import RegionModel
from ixmp4.core.unit import UnitModel
from ixmp4.data.abstract import DataPoint

from pyam.index import get_index_levels_codes

logger = logging.getLogger(__name__)


def read_ixmp4(
    platform: ixmp4.Platform | str,
    default_only: bool = True,
    model: str | list[str] | None = None,
    scenario: str | list[str] | None = None,
    region: str | list[str] | None = None,
    variable: str | list[str] | None = None,
    unit: str | list[str] | None = None,
    year: int | list[int] | None = None,
):
    """Read scenario runs from an ixmp4 platform database instance

    Parameters
    ----------
    platform : :class:`ixmp4.Platform` or str
        The ixmp4 platform database instance to which the scenario data is saved.
    default_only : :class:`bool`, optional
        Read only default runs.
    model, scenario, region, variable, unit : str or list of str, optional
        Filter by these dimensions.
    year : int or list of int, optional
        Filter by time domain.
    """
    from pyam import IamDataFrame

    if not isinstance(platform, ixmp4.Platform):
        platform = ixmp4.Platform(platform)

    # TODO This may have to be revised, see https://github.com/iiasa/ixmp4/issues/72
    meta_filters = dict(
        run=dict(default_only=default_only, model=model, scenario=scenario)
    )
    iamc_filters = dict(
        run=dict(default_only=default_only),
        model=model,
        scenario=scenario,
        region=region,
        variable=variable,
        unit=unit,
        year=year,
    )
    data = platform.iamc.tabulate(**iamc_filters)
    meta = platform.meta.tabulate(**meta_filters)

    # if default-only, simplify to standard IAMC index, add `version` as meta indicator
    if default_only:
        index = ["model", "scenario"]
        meta_version = (
            data[index + ["version"]]
            .drop_duplicates()
            .rename(columns={"version": "value"})
        )
        meta_version["key"] = "version"
        meta = pd.concat([meta.drop(columns="version"), meta_version])
        data.drop(columns="version", inplace=True)
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
    if df.extra_cols:
        raise NotImplementedError(
            "Only data with standard IAMC columns can be written to an ixmp4 platform."
        )

    if not isinstance(platform, ixmp4.Platform):
        platform = ixmp4.Platform(platform)

    # TODO: implement try-except to roll back changes if any error writing to platform
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
                + f". Use `Platform.{dimension}.create()` to add missing elements."
            )

    # The "version" meta-indicator, added when reading from an ixmp4 platform,
    # should not be written to the platform
    if "version" in df.meta.columns:
        logger.warning(
            "The `meta.version` column was dropped when writing to the ixmp4 platform."
        )
        meta = df.meta.drop(columns="version")
    else:
        meta = df.meta.copy()

    # Create runs and add IAMC timeseries data and meta indicators
    for model, scenario in df.index:
        _df = df.filter(model=model, scenario=scenario)

        run = platform.runs.create(model=model, scenario=scenario)

        if df.time_domain == "year":
              run.iamc.add(_df.data)
        else:
            levels, codes = get_index_levels_codes(_df._data, "time")
            year_rows = np.isin(
                codes,
                [i for (i, label) in enumerate(levels) if isinstance(label, int)],
            )
            data = _df.data.reset_index()
            if any(year_rows):
                run.iamc.add(
                    data[year_rows].rename(columns={"time": "step_year"}),
                    type=DataPoint.Type.ANNUAL,
                )
            if any(~year_rows):
                run.iamc.add(
                    data[~year_rows].rename(columns={"time": "step_datetime"}),
                    type=DataPoint.Type.DATETIME,
                )

        if not meta.empty:
            run.meta = dict(meta.loc[(model, scenario)])
        run.set_as_default()
