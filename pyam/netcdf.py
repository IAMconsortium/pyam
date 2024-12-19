import datetime as dt

import numpy as np
import pandas as pd

from pyam.index import get_index_levels

try:
    import xarray as xr

    HAS_XARRAY = True
except ModuleNotFoundError:
    xr = None
    HAS_XARRAY = False
from pyam.utils import IAMC_IDX, META_IDX

NETCDF_IDX = ["time", "model", "scenario", "region"]


def read_netcdf(path):
    """Read timeseries data and meta indicators from a netCDF file

    Parameters
    ----------
    path : :class:`pathlib.Path` or file-like object
        Scenario data file in netCDF format.

    Returns
    -------
    :class:`IamDataFrame`

    See Also
    --------
    pyam.IamDataFrame.to_netcdf

    Notes
    -----
    Read the `pyam-netcdf docs <https://pyam-iamc.readthedocs.io/en/stable/api/io.html>`_
    for more information on the expected file format structure.

    """
    from pyam import IamDataFrame

    if not HAS_XARRAY:
        raise ModuleNotFoundError("Reading netcdf files requires 'xarray'.")
    _ds = xr.open_dataset(path)
    _list_variables = [i for i in _ds.to_dict()["data_vars"].keys()]

    # Check if the time coordinate is years (integers) or date time-format
    is_year_based = all(
        isinstance(x, (int, np.integer)) for x in _ds.coords["time"].values
    )
    is_datetime = all(
        isinstance(x, (dt.date, dt.time, np.datetime64))
        for x in _ds.coords["time"].values
    )

    # Check if the xarray dataset has the correct coordinates, then get column names
    if is_year_based:
        _list_cols = IAMC_IDX + ["year", "value"]
    elif is_datetime:
        _list_cols = IAMC_IDX + ["time", "value"]
    else:
        raise TypeError(
            "Time coordinates can year (integer) or datetime format, found: "
            + _ds.coords["time"]
        )

    # read `data` table
    _data = []
    _meta = []
    for _var in _list_variables:
        # Check dimensions, if exactly as in META_IDX is a meta indicator
        # if exactly as in IAMC_IDX is a variable
        if set(_ds[_var].dims) == set(META_IDX):
            _meta.append(_var)
        elif set(_ds[_var].dims) == set(NETCDF_IDX):
            # convert the data into the IamDataframe format
            _tmp = (
                _ds[_var]
                .to_dataframe()
                .rename(columns={_var: "value"})
                .reset_index(drop=False)
            )
            _tmp["variable"] = _ds[_var].long_name
            _tmp["unit"] = _ds[_var].unit
            _data.append(_tmp)
        else:
            raise TypeError(
                f"Cannot define {_var}, different indices from META_IDX and IAMC_IDX."
            )
    data = pd.concat(_data).reset_index(drop=True)
    # if year-based data, get the time coordinate as "year"
    # if timeseries, keep the time coordinate as "time"
    if is_year_based:
        data = data.rename(columns={"time": "year"})

    return IamDataFrame(
        data,
        meta=_ds[_meta].to_dataframe().replace("nan", np.nan) if _meta else None,
    )


def to_xarray(data_series: pd.Series, meta: pd.DataFrame):
    """Convert timeseries data and meta indicators to an xarray Dataset

    Returns
    -------
    :class:`xarray.Dataset`

    """
    if not HAS_XARRAY:
        raise ModuleNotFoundError("Converting to xarray requires 'xarray'.")

    dataset = xr.Dataset()

    # add timeseries data-variables
    for variable, _variable_data in data_series.groupby("variable"):
        unit = get_index_levels(_variable_data, "unit")

        if len(unit) > 1:
            raise ValueError(
                "Cannot write to xarray for non-unique units in '" + variable + "'."
            )

        dataset[variable] = xr.DataArray(
            _variable_data.droplevel(["variable", "unit"]).to_xarray(),
        )
        dataset[variable].attrs = {
            "unit": unit[0],
            "long_name": variable,
        }

    # add meta indicators as data-variables
    for meta_indicator, meta_data in meta.items():
        meta_data = meta_data.replace(np.nan, "nan")
        dataset[meta_indicator] = xr.DataArray(
            meta_data.to_xarray(),
            dims=META_IDX,
            name=meta_indicator,
        )

    return dataset
