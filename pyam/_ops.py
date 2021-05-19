import operator
from pyam.index import append_index_level, get_index_levels


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
    _args = [df.filter(**{axis: i})._data.groupby(cols).sum() for i in method_args]

    for key, value in kwds.items():
        if value in get_index_levels(df._data, axis):
            kwds[key] = df.filter(**{axis: value})._data.groupby(cols).sum()

    _value = method(*_args, **kwds)
    _value.index = append_index_level(_value.index, codes=0, level=name, name=axis)

    return _value
