import datetime
import time

import numpy as np
import pandas as pd

from pyam.index import get_index_levels_codes, get_keep_col
from pyam.utils import is_str, pattern_match, to_list

FILTER_DATETIME_ATTRS = {
    "month": (["%b", "%B"], "tm_mon", "months"),
    "day": (["%a", "%A"], "tm_wday", "days"),
}


def filter_by_col(data, col, values, regexp, level=None):
    levels, codes = get_index_levels_codes(data, col)
    matches = pattern_match(
        levels,
        values,
        regexp=regexp,
        level=level,
        has_nan=True,
        return_codes=True,
    )
    return get_keep_col(codes, matches)


def filter_by_measurand(data, values, regexp, level=None):
    variable, unit = values
    if is_str(variable) and is_str(unit):
        return np.logical_and(
            filter_by_col(data, "variable", variable, regexp, level),
            filter_by_col(data, "unit", unit, regexp),
        )
    # values is an iterable of measurands
    keep_col = np.zeros(len(data), dtype=bool)
    for measurand in values:
        keep_col = np.logical_or(
            keep_col, filter_by_measurand(data, measurand, regexp, level)
        )
    return keep_col


def filter_by_time_domain(values, levels, codes):
    """Internal implementation to filter by time domain"""

    if values == "year":
        matches = [i for (i, label) in enumerate(levels) if isinstance(label, int)]
    elif values == "datetime":
        matches = [i for (i, label) in enumerate(levels) if not isinstance(label, int)]
    else:
        raise ValueError(f"Filter by `time_domain='{values}'` not supported!")

    return np.isin(codes, matches)


def filter_by_year(time_col, values, levels, codes):
    """Internal implementation to filter by time domain"""

    if time_col == "time":
        levels = [i.year if isinstance(i, pd.Timestamp) else i for i in levels]
    return get_keep_col(codes, years_match(levels, values))


def filter_by_dt_arg(col, values, data):
    """Internal implementation to filter by datetime arguments"""

    def time_col(x, col):
        return getattr(x, col) if isinstance(x, pd.Timestamp) else None

    if col == "day":
        if isinstance(values, str):
            wday = True
        elif isinstance(values, list) and isinstance(values[0], str):
            wday = True
        else:
            wday = False

        if wday:
            data = data.apply(lambda x: x.weekday())
        else:  # ints or list of ints
            data = data.apply(lambda x: x.day)

    else:
        data = data.apply(lambda x: time_col(x, col))

    if col in FILTER_DATETIME_ATTRS:
        return time_match(data, values, *FILTER_DATETIME_ATTRS[col])
    else:
        return np.isin(data, values)


def years_match(levels, years):
    """Return rows where data matches year"""
    years = to_list(years)
    if not all([pd.api.types.is_integer(y) for y in years]):
        raise TypeError("Filter by `year` requires integers!")

    return np.isin(levels, years)


def time_match(data, times, conv_codes, strptime_attr, name):
    """Return rows where data matches a timestamp"""

    def conv_strs(strs_to_convert, conv_codes, name):
        for conv_code in conv_codes:
            try:
                res = [
                    getattr(time.strptime(t, conv_code), strptime_attr)
                    for t in strs_to_convert
                ]
                break
            except ValueError:
                continue

        try:
            return res
        except NameError:
            raise ValueError(f"Could not convert {name} to integer: {times}")

    times = [times] if isinstance(times, (int, str)) else times
    if isinstance(times[0], str):
        to_delete = []
        to_append = []
        for i, timeset in enumerate(times):
            if "-" in timeset:
                ints = conv_strs(timeset.split("-"), conv_codes, name)
                if ints[0] > ints[1]:
                    error_msg = (
                        "string ranges must lead to increasing integer ranges,"
                        " {} becomes {}".format(timeset, ints)
                    )
                    raise ValueError(error_msg)

                # + 1 to include last month
                to_append += [j for j in range(ints[0], ints[1] + 1)]
                to_delete.append(i)

        for i in to_delete:
            del times[i]

        times = conv_strs(times, conv_codes, name)
        times += to_append

    return np.isin(data, times)


def datetime_match(data, dts):
    """Matching of datetimes in time columns for data filtering"""
    dts = to_list(dts)
    if any([not (isinstance(i, (datetime.datetime, np.datetime64))) for i in dts]):
        error_msg = "`time` can only be filtered by datetimes and datetime64s"
        raise TypeError(error_msg)
    return data.isin(dts).values
