from pyam.index import append_index_level


def subtract(a, b):
    return a - b


def add(a, b):
    return a + b


def divide(a, b):
    return a / b


def multiply(a, b):
    return a * b


KNOWN_OPS = {
    "subtract": subtract,
    "add": add,
    "divide": divide,
    "multiply": multiply,
}


def _op_data(df, a, b, name, method, axis):
    """Internal implementation of numerical operations on timeseries"""

    if axis not in df._data.index.names:
        raise ValueError(f"Unknown axis: {axis}")

    if method in KNOWN_OPS:
        method = KNOWN_OPS[method]
    else:
        raise ValueError(f"Unknown method: {method}")

    cols = df._data.index.names.difference([axis])
    _a = df.filter(**{axis: a})._data.groupby(cols).sum()
    _b = df.filter(**{axis: b})._data.groupby(cols).sum()

    _value = method(_a, _b)
    _value.index = append_index_level(_value.index, codes=0, level=name, name=axis)

    return _value
