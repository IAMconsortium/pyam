from pyam.utils import _raise_data_error


def swap_time_for_year(df, inplace):
    """Internal implementation to swap 'time' domain to 'year' (as int)"""
    if not df.time_col == "time":
        raise ValueError("Time domain must be datetime to use this method")

    ret = df.copy() if not inplace else df

    _data = ret.data
    _data["year"] = _data["time"].apply(lambda x: x.year)
    _data = _data.drop("time", axis="columns")
    _index = [v if v != "time" else "year" for v in ret._LONG_IDX]

    rows = _data[_index].duplicated()
    if any(rows):
        error_msg = "Swapping time for year causes duplicates in `data`"
        _raise_data_error(error_msg, _data[_index])

    # assign data and other attributes
    ret._LONG_IDX = _index
    ret._data = _data.set_index(ret._LONG_IDX).value
    ret.time_col = "year"
    ret._set_attributes()
    delattr(ret, "time")

    if not inplace:
        return ret