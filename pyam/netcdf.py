import numpy as np
import datetime as dt
import pandas as pd

try:
    import xarray as xr

    HAS_XARRAY = True
except ModuleNotFoundError:
    xr = None
    HAS_XARRAY = False
from pyam.core import IamDataFrame
from pyam.utils import META_IDX, IAMC_IDX


def read_netcdf(path):
    """Read timeseries data and meta indicators from a netCDF file

    Parameters
    ----------
    path : :class:`pathlib.Path` or file-like object
        Scenario data file in netCDF format.

    Returns
    ----------
    :class:`IamDataFrame`

    """
    if not HAS_XARRAY:
        raise ModuleNotFoundError("Reading netcdf files requires 'xarray'")
    _ds = xr.open_dataset(path)
    NETCDF_IDX = ["time", "model", "scenario", "region"]
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
            + ds.coords["time"]
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
