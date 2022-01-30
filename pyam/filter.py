import datetime
import time
import numpy as np
import pandas as pd

from pyam.utils import to_list


FILTER_DATETIME_ATTRS = {
    "month": (["%b", "%B"], "tm_mon", "months"),
}


def filter_by_time_domain(values, levels, codes):
    """Internal implementation to filter by time domain"""

    if values == "year":
        matches = [i for (i, label) in enumerate(levels) if isinstance(label, int)]
    elif values == "datetime":
        matches = [i for (i, label) in enumerate(levels) if not isinstance(label, int)]
    else:
        raise ValueError(f"Filter by `datetime='{values}'` not supported!")

    return np.isin(codes, matches)


def years_match(levels, years):
    """Return rows where data matches year"""
    years = to_list(years)
    if not all([pd.api.types.is_integer(y) for y in years]):
        raise TypeError("Filter by `year` requires integers!")
    return np.isin(levels, years)


def day_match(data, days):
    """Return rows where data matches days"""
    return time_match(data, days, ["%a", "%A"], "tm_wday", "days")


def hour_match(data, hours):
    """Return rows where data matches hours"""
    hours = [hours] if isinstance(hours, int) else hours
    return np.isin(data, hours)


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
