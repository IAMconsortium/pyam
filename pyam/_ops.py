import operator
import pandas as pd
from pyam.index import append_index_level, get_index_levels, replace_index_values
from pyam.utils import to_list
from iam_units import registry
from pint import Quantity


# these functions have to be defined explicitly to allow calling them with keyword args
def add(a, b):
    return operator.add(*_make_series(a, b))


def subtract(a, b):
    return operator.sub(*_make_series(a, b))


def multiply(a, b):
    return operator.mul(*_make_series(a, b))


def divide(a, b):
    return operator.truediv(*_make_series(a, b))


def _make_series(a, b):
    """Cast instances of a pint.Quantity to a pd.Series

    Calling an operation on a pd.Series and a pint.Quantity removes the units,
    therefore a pint.Quantity is transformed to a pd.Series of quantities.
    """
    if isinstance(a, Quantity):
        return pd.Series([a] * len(b), index=b.index), b
    if isinstance(b, Quantity):
        return a, pd.Series([b] * len(a), index=a.index)
    return a, b


KNOWN_OPS = {
    "add": add,
    "subtract": subtract,
    "multiply": multiply,
    "divide": divide,
}


def _op_data(df, name, method, axis, fillna=None, args=(), ignore_units=False, **kwds):
    """Internal implementation of numerical operations on timeseries"""

    if axis not in df.dimensions:
        raise ValueError(f"Unknown axis: {axis}")

    if method in KNOWN_OPS:
        method = KNOWN_OPS[method]
    elif callable(method):
        pass
    else:
        raise ValueError(f"Unknown method: {method}")

    cols = [d for d in df.dimensions if d != axis]

    # replace args and and kwds with values of `df._data` if applicable
    # _data_args and _data_kwds track if an argument was replaced by `df._data` values
    n = len(args)
    _args, _data_args, _units_args = [None] * n, [False] * n, [None] * n
    for i, value in enumerate(args):
        _args[i], _units_args[i], _data_args[i] = _get_values(
            df, axis, value, cols, f"_arg{i}"
        )

    _data_kwds, _unit_kwds = {}, {}
    for i, (key, value) in enumerate(kwds.items()):
        kwds[key], _unit_kwds[key], _data_kwds[key] = _get_values(
            df, axis, value, cols, key
        )

    # fast-pass on units: override pint for some methods if all kwds have the same unit
    if (
        method in [add, subtract, divide]
        and ignore_units is False
        and fillna is None
        and len(_unit_kwds["a"]) == 1
        and len(_unit_kwds["b"]) == 1
        and registry.Unit(_unit_kwds["a"][0]) == registry.Unit(_unit_kwds["b"][0])
    ):
        # activate ignore-units feature
        ignore_units = _unit_kwds["a"][0] if method in [add, subtract] else ""
        # downcast `pint.Quantity` to numerical value
        kwds["a"], kwds["b"] = _to_value(kwds["a"]), _to_value(kwds["b"])

    # cast args and kwds to pd.Series of pint.Quantity
    if ignore_units is False:
        for i, is_data in enumerate(_data_args):
            _args[i] = _to_quantity(_args[i]) if is_data else _args[i]
        for key, value in kwds.items():
            kwds[key] = _to_quantity(value) if _data_kwds[key] else value
    # else remove units from pd.Series
    else:
        for i, is_data in enumerate(_data_args):
            _args[i] = _args[i].reset_index("unit", drop=True) if is_data else _args[i]
        for key, value in kwds.items():
            kwds[key] = (
                value.reset_index("unit", drop=True) if _data_kwds[key] else value
            )

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
    result = method(*_args, **kwds)
    if not isinstance(result, pd.Series):
        msg = f"Value returned by `{method.__name__}` cannot be cast to an IamDataFrame"
        raise ValueError(f"{msg}: {result}")

    # separate pint quantities into numerical value and unit (as index)
    if ignore_units is False:
        _value = pd.DataFrame(
            [[i.magnitude, "{:~}".format(i.units)] for i in result.values],
            columns=["value", "unit"],
            index=result.index,
        ).set_index("unit", append=True)
        _value.index = replace_index_values(_value, "unit", {"dimensionless": ""})

    # otherwise, set unit (as index) to "unknown" or the value given by "ignore_units"
    else:
        index = append_index_level(
            result.index,
            codes=0,
            level="unknown" if ignore_units is True else ignore_units,
            name="unit",
        )
        _value = pd.Series(result.values, index=index, name="value")

    # append the `name` to the index on the `axis`
    _value.index = append_index_level(_value.index, codes=0, level=name, name=axis)
    return _value


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
    Tuple of the following:
     - Either `df.data` downselected by `{axis: value}` or `value`
     - List of units of the timeseries data or `value`
     - Bool whether first item was derived from `df.data`

    """
    # check if `value` is a `pint.Quantity` and return unit specifically
    if isinstance(value, Quantity):
        return value, [value.units], False
    # try selecting from `df.data`
    if any(v in get_index_levels(df._data, axis) for v in to_list(value)):
        _df = df.filter(**{axis: value})
        return _df._data.groupby(cols).sum().rename(index=name), _df.unit, True
    # else, return value
    return value, [], False


def _to_quantity(data):
    """Convert the values of an indexed pd.Series into pint.Quantity instances"""
    return pd.Series(
        [
            registry.Quantity(v, u)
            for v, u in zip(data.values, data.index.get_level_values("unit"))
        ],
        index=data.reset_index("unit", drop=True).index,
        name=data.name,
    )


def _to_value(x):
    """Return the value of a pint.Quantity"""
    if isinstance(x, Quantity):
        return x.magnitude
    return x
