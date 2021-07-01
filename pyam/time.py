import pandas as pd
from pyam.index import get_index_levels, append_index_col
from pyam.utils import _raise_data_error


def swap_time_for_year(df, inplace):
    """Internal implementation to swap 'time' domain to 'year' (as int)"""
    if not df.time_col == "time":
        raise ValueError("Time domain must be datetime to use this method")

    ret = df.copy() if not inplace else df

    index = ret._data.index

    time = pd.Series(index.get_level_values("time"))
    order = [v if v != "time" else "year" for v in index.names]

    index = index.droplevel("time")
    index = append_index_col(index, time.apply(lambda x: x.year), "year", order=order)

    rows = index.duplicated()
    if any(rows):
        error_msg = "Swapping time for year causes duplicates in `data`"
        _raise_data_error(error_msg, index.to_frame().reset_index(drop=True))

    # assign data and other attributes
    ret._LONG_IDX = index.names
    ret._data.index = index
    ret.time_col = "year"
    ret._set_attributes()
    delattr(ret, "time")

    if not inplace:
        return ret