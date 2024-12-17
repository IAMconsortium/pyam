import datetime as dt

import numpy as np
import pandas as pd

try:
    import xarray as xr

    HAS_XARRAY = True
except ModuleNotFoundError:
    xr = None
    HAS_XARRAY = False
from pyam.core import IamDataFrame
from pyam.utils import IAMC_IDX, META_IDX

NETCDF_IDX = ["time", "model", "scenario", "region"]


def read_netcdf(_ds=None, path=None):
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
    if path:
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


def to_netcdf(self, path=None, iamc_index=False, **kwargs):
    """Write :meth:`IamDataFrame.timeseries` to a Network Common Data Form (netCDF) file

    Parameters
    ----------
    path : str, path or file-like, optional
        File path as string or :class:`pathlib.Path`, or file-like object.
        If *None*, the result is returned as a csv-formatted string.
        See :meth:`pandas.DataFrame.to_csv` for details.
    Return
    --------
    xarray dataset ready to be saved in netCDF format
    """
    _data = self.data
    _ds = xr.Dataset()

    # if IamDataFrame with a 'year' dimension, rename to 'time' to match NETCDF_IDX
    if "year" in self.dimensions:
        _data = _data.rename(columns={"year": "time"})

    # get data from variables
    for _var in self.variable:

        _tmp = _data[_data["variable"] == _var]
        _ds[_var] = xr.DataArray(
            _tmp.set_index(NETCDF_IDX)["value"].to_xarray(),
            dims=(NETCDF_IDX),
        )

        # variable attributes
        _ds[_var].attrs = {
            "unit": _tmp["unit"].iloc[0],
            "long_name": _var,
        }

    # get data for meta indicators with META_IDX (model and scenario)
    for _var in self.meta.keys():

        _ds[_var] = xr.DataArray(
            self.meta[_var].to_xarray(),
            dims=(META_IDX),
            name=_var,
        )

    # global attributes
    _ds.attrs["Information"] = "netCDF file written from :class:`IamDataFrame`"
    return _ds
