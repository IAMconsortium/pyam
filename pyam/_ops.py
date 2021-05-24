import operator
import pandas as pd
from pyam.index import append_index_level, get_index_levels
from pyam.utils import to_list


# these functions have to be defined explicitly to allow calling them with keyword args
def add(a, b):
    return operator.add(a, b)


def subtract(a, b):
    return operator.sub(a, b)


def multiply(a, b):
    return operator.mul(a, b)


def divide(a, b):
    return operator.truediv(a, b)


KNOWN_OPS = {
    "add": add,
    "subtract": subtract,
    "multiply": multiply,
    "divide": divide,
}


def _op_data(df, name, method, axis, fillna=None, args=(), **kwds):
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

    # replace args and and kwds with values of `df._data` if applicable
    # _data_args and _data_kwds track if an argument was replaced by `df._data` values
    _args, _data_args = [None] * len(args), [False] * len(args)
    for i, value in enumerate(args):
        _args[i], _data_args[i] = _get_values(df, axis, value, cols, f"_arg{i}")

    _data_kwds = {}
    for i, (key, value) in enumerate(kwds.items()):
        kwds[key], _data_kwds[key] = _get_values(df, axis, value, cols, key)

    # merge all args and kwds that are based on `df._data` to apply fillna
    if fillna:
        _data_cols = [_args[i] for i, is_data in enumerate(_data_args) if is_data]
        _data_cols += [kwds[key] for key, is_data in _data_kwds.items() if is_data]
        _data = pd.merge(*_data_cols, how="outer", left_index=True, right_index=True)

        _data.fillna(fillna, inplace=True)

        for i, is_data in enumerate(_data_args):
            if is_data:
                _args[i] = _data[f"_arg{i}"]
        for key, is_data in _data_kwds.items():
            if is_data:
                kwds[key] = _data[key]

    # apply method and check that returned object is valid
    _value = method(*_args, **kwds)
    if not isinstance(_value, pd.Series):
        msg = f"Value returned by `{method.__name__}` cannot be cast to an IamDataFrame"
        raise ValueError(f"{msg}: {_value}")

    # insert the index level and reset the series name
    _value.index = append_index_level(_value.index, codes=0, level=name, name=axis)
    return _value.rename(index=None)


def _get_values(df, axis, value, cols, name):
    """Return grouped data if value is in axis. Otherwise return value.

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
    name : str
        Name of the returned pd.Series.

    Returns
    -------
    Either filtered timeseries from `df` or `value`

    """
    if any(v in get_index_levels(df._data, axis) for v in to_list(value)):
        _data = df.filter(**{axis: value})._data.groupby(cols).sum().rename(index=name)
        return _data, True
    else:
        return value, False
