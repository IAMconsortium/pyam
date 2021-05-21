import operator
from pyam.index import append_index_level, get_index_levels
from pyam.utils import to_list


KNOWN_OPS = {
    "add": operator.add,
    "subtract": operator.sub,
    "multiply": operator.mul,
    "divide": operator.truediv,
}


def _op_data(df, method_args, name, method, axis, **kwds):
    """Internal implementation of numerical operations on timeseries"""

    if axis not in df._data.index.names:
        raise ValueError(f"Unknown axis: {axis}")

    if method in KNOWN_OPS:
        method = KNOWN_OPS[method]
    elif callable(method):
        pass
    else:
        raise ValueError(f"Unknown method: {method}")

    cols = df._data.index.names.difference([axis])
    _args = [_get_values(df, axis, value, cols) for value in method_args]

    for key, value in kwds.items():
        kwds[key] = _get_values(df, axis, value, cols)

    _value = method(*_args, **kwds)
    _value.index = append_index_level(_value.index, codes=0, level=name, name=axis)

    return _value


def _get_values(df, axis, value, cols):
    """Return the value of the time series, if value is in axis. Otherwise
    return values itself.

    Parameters
    ----------
    df : IamDataFrame
        IamDataFrame to select the values from.
    axis : str
        Axis in `df` that contains value.
    value : str or list of str or any
        Either str or list of str in axis or anything else.
    cols : list
        Columns in df that are not `axis`.

    Returns
    -------
    Either filtered timeseries from `df` or `value`

    """
    if any(v in get_index_levels(df._data, axis) for v in to_list(value)):
        return df.filter(**{axis: value})._data.groupby(cols).sum()
    else:
        return value
