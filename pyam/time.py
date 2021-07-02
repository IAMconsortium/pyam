import pandas as pd
from pyam.index import get_index_levels, append_index_col
from pyam.utils import _raise_data_error


def swap_time_for_year(df, inplace, subannual=False):
    """Internal implementation to swap 'time' domain to 'year' (as int)"""
    if not df.time_col == "time":
        raise ValueError("Time domain must be datetime to use this method")

    ret = df.copy() if not inplace else df

    index = ret._data.index

    time = pd.Series(index.get_level_values("time"))
    order = [v if v != "time" else "year" for v in index.names]

    index = index.droplevel("time")
    index = append_index_col(index, time.apply(lambda x: x.year), "year", order=order)

    if subannual:
        # if subannual is True, default to simple datetime format without year
        if subannual is True:
            subannual = "%m-%d %H:%M%z"
        if isinstance(subannual, str):
            _subannual = time.apply(lambda x: x.strftime(subannual))
        else:
            _subannual = time.apply(subannual)

        index = append_index_col(index, _subannual, "subannual")

    rows = index.duplicated()
    if any(rows):
        error_msg = "Swapping time for year causes duplicates in `data`"
        _raise_data_error(error_msg, index[rows].to_frame().reset_index(drop=True))

    # assign data and other attributes
    ret._LONG_IDX = index.names
    ret._data.index = index
    ret.time_col = "year"
    ret._set_attributes()
    delattr(ret, "time")

    if not inplace:
        return ret
