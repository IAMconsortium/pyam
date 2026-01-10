import logging

import ixmp4
import pandas as pd
from ixmp4.core.region import RegionModel
from ixmp4.core.unit import UnitModel
from ixmp4.data.abstract import DataPoint

logger = logging.getLogger(__name__)


def read_ixmp4(
    platform: ixmp4.Platform | str,
    *,
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
    iamc_filters = dict(run=dict(default_only=default_only))
    for key, value in (
        ("model", model),
        ("scenario", scenario),
        ("region", region),
        ("variable", variable),
        ("unit", unit),
        ("year", year),
    ):
        if value is not None:
            iamc_filters[key] = value

    data = platform.iamc.tabulate(**iamc_filters)
    meta = platform.meta.tabulate(**meta_filters)

    if data.empty:
        raise ValueError("No scenario data with filters " + str(iamc_filters))

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


def read_run(
    run: ixmp4.Run,
    region: str | list[str] | None = None,
    variable: str | list[str] | None = None,
    unit: str | list[str] | None = None,
    year: int | list[int] | None = None,
):
    """Return timeseries data and meta indicatos from an ixmp4 run as IamDataFrame

    Parameters
    ----------
    run : :class:`ixmp4.Run`
        An ixmp4 run instance from which to read timeseries data and meta indicators
    region, variable, unit : str or list of str, optional
        Filter timeseries data by these dimensions.
    year : int or list of int, optional
        Filter timeseries data by time domain.
    """
    from pyam import IamDataFrame

    if year is not None:
        raise NotImplementedError("Filter by 'year' not implemented in ixmp4.")

    meta = pd.DataFrame.from_dict(run.meta, orient="index").T
    meta.index = pd.MultiIndex.from_tuples(
        [(run.model.name, run.scenario.name)], name=["model", "scenario"]
    )
    meta["version"] = run.version

    return IamDataFrame(
        data=run.iamc.tabulate(
            region=region,
            variable=variable,
            unit=unit,
        ),
        meta=meta,
        model=run.model.name,
        scenario=run.scenario.name,
    )


def write_to_ixmp4(platform: ixmp4.Platform | str, df, checkpoint_message: str):
    """Save all scenarios as new default runs in an ixmp4 platform database instance

    Parameters
    ----------
    platform : :class:`ixmp4.Platform` or str
        The ixmp4 platform database instance to which the scenario data is saved
    df : pyam.IamDataFrame
        The IamDataFrame instance with scenario data
    checkpoint_message : str
        The message for the ixmp4 checkpoint (similar to a commit message).
    """
    if df.extra_cols:
        raise NotImplementedError(
            "Only data with standard IAMC columns can be written to an ixmp4 platform."
        )

    if df.time_domain not in ["year", "datetime"]:
        raise NotImplementedError(
            "Only data with time domain 'year' or 'datetime' can be written to an ixmp4"
            " platform."
        )

    if not isinstance(platform, ixmp4.Platform):
        platform = ixmp4.Platform(platform)

    _validate_dimensions(platform, df)

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

        with run.transact(checkpoint_message):
            if df.time_domain == "year":
                run.iamc.add(_df.data)
            elif df.time_domain == "datetime":
                run.iamc.add(
                    _df.data.rename(columns={"time": "step_datetime"}),
                    type=DataPoint.Type.DATETIME,
                )

            if not meta.empty:
                run.meta = dict(meta.loc[(model, scenario)])
            run.set_as_default()


def _validate_dimensions(platform, df):
    """Ensure that all regions and units in the DataFrame exist in the platform"""
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
