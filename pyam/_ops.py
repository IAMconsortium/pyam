from pyam.index import append_index_level


def subtract(a, b):
    return a - b


KNOWN_OPS = {
    "subtract": subtract
}


def _op_data(df, a, b, name, method, axis):
    """Internal implementation of numerical operations on timeseries"""

    cols = df._data.index.names.difference([axis])
    _a = df.filter(**{axis: a})._data.groupby(cols).sum()
    _b = df.filter(**{axis: b})._data.groupby(cols).sum()

    if method in KNOWN_OPS:
        method = KNOWN_OPS[method]
    else:
        raise ValueError(f"Unknown method: {method}")

    _value = method(_a, _b)
    _value.index = append_index_level(_value.index, 0, name, axis)

    return _value
