import logging
import math
import numpy as np
import pandas as pd
from pyam.utils import isstr, to_int, remove_from_list

logger = logging.getLogger(__name__)


def fill_series(x, time):
    """Returns the timeseries value at a point in time by linear interpolation

    Parameters
    ----------
    x : pandas.Series
        a timeseries to be interpolated
    time : int or pandas.datetime
        year or datetime to interpolate
    """
    x = x.dropna()
    if time in x.index and not np.isnan(x[time]):
        return x[time]
    else:
        prev = [i for i in x.index if i < time]
        nxt = [i for i in x.index if i > time]
        if prev and nxt:
            p = max(prev)
            n = min(nxt)
            return ((n - time) * x[p] + (time - p) * x[n]) / (n - p)
        else:
            return np.nan


def cumulative(x, first_year, last_year):
    """Returns the cumulative sum of a timeseries

    This function implements linear interpolation between years
    and ignores nan's in the range.
    The function includes the last-year value of the series, and
    raises a warning if start_year or last_year is outside of
    the timeseries range and returns nan

    Parameters
    ----------
    x : pandas.Series
        a timeseries to be summed over time
    first_year : int
        first year of the sum
    last_year : int
        last year of the sum (inclusive)
    """
    # if the timeseries does not cover the range `[first_year, last_year]`,
    # return nan to avoid erroneous aggregation
    if min(x.index) > first_year:
        logger.warning(f"Start of period {first_year} outside of range.")
        return np.nan
    if max(x.index) < last_year:
        logger.warning(f"End of period {last_year} outside of range.")
        return np.nan

    # make sure we're using integers
    to_int(x, index=True)

    x[first_year] = fill_series(x, first_year)
    x[last_year] = fill_series(x, last_year)

    years = [
        i for i in x.index if i >= first_year and i <= last_year and ~np.isnan(x[i])
    ]
    years.sort()

    # loop over years
    if not np.isnan(x[first_year]) and not np.isnan(x[last_year]):
        value = 0
        for (i, yr) in enumerate(years[:-1]):
            next_yr = years[i + 1]
            # the summation is shifted to include the first year fully in sum,
            # otherwise, would return a weighted average of `yr` and `next_yr`
            value += ((next_yr - yr - 1) * x[next_yr] + (next_yr - yr + 1) * x[yr]) / 2

        # the loop above does not include the last element in range
        # (`last_year`), therefore added explicitly
        value += x[last_year]

        return value


def cross_threshold(
    x, threshold=0, direction=["from above", "from below"], return_type=int
):
    """Returns a list of the years in which a timeseries crosses a threshold

    Parameters
    ----------
    x : :class:`pandas.Series`
        A timeseries indexed over years (as integers)
    threshold : float, optional
        The threshold that the timeseries is checked against
    direction : str, optional
        Whether to return all years where the threshold is crossed
        or only where threshold is crossed in a specific direction
    return_type : type, optional
        Whether to cast the returned values to integer (years)
    """
    direction = [direction] if isstr(direction) else list(direction)
    if not set(direction).issubset(set(["from above", "from below"])):
        raise ValueError(f"Invalid direction: {direction}")

    # get the values and time-domain index
    x = x.dropna()
    values, index = x.values - threshold, x.index.to_numpy()
    positive, negative = (values >= 0), (values < 0)

    # determine all indices before crossing the threshold
    pre = [False] * (len(x) - 1)
    if "from above" in direction:
        pre |= positive[:-1] & negative[1:]
    if "from below" in direction:
        pre |= positive[1:] & negative[:-1]
    pre = np.argwhere(pre)
    # determine all indices after crossing the threshold
    post = pre + 1

    # compute the index value where the threshold is crossed
    change = (values[post] - values[pre]) / (index[post] - index[pre])
    years = index[pre] - values[pre] / change

    # it year (as int) is returned, add one because int() rounds down
    if return_type == int:
        return [y + 1 for y in map(int, years)]
    return years


def compute_learning_rate(x, performance, experience):
    """Compute the implicit learning rate from timeseries data

    Experience curves are based on the concept that a technology's performance improves
    as experience with this technology grows.

    The "learning rate" indicates the performance improvement (e.g., cost reduction) for
    each doubling of the accumulated experience (e.g., cumulative installed capacity).

    The experience curve parameter *b* is equivalent to the (linear) slope when plotting
    performance and experience timeseries on double-logarithmic scales.
    The learning rate can be computed from the experience curve parameter as
    :math:`1 - 2^{b}`.

    The learning rate parameter in period *t* is computed based on the changes
    to the subsequent period, i.e., from period *t* to period *t+1*.

    Parameters
    ----------
    x : :class:`pandas.Series`
        Timeseries data of the *performance* and *experience* variables
        indexed over the time domain.
    performance : str
        Variable of the "performance" timeseries (e.g., specific investment costs).
    experience : str
        Variable of the "experience" timeseries (e.g., cumulative installed capacity).

    Returns
    -------
    Indexed :class:`pandas.Series` of implicit learning rates
    """
    # drop all index dimensions other than "variable" and "year"
    x.index = x.index.droplevel(
        [i for i in x.index.names if i not in ["variable", "year"]]
    )

    # apply log, dropping all values that are zero or negative
    x = x[x > 0].apply(math.log10)

    # return empty pd.Series if not all relevant variables exist
    if not all([v in x.index for v in [performance, experience]]):
        names = remove_from_list(x.index.names, "variable")
        empty_list = [[]] * len(names)
        return pd.Series(
            index=pd.MultiIndex(levels=empty_list, codes=empty_list, names=names),
            dtype="float64",
        )

    # compute the "experience parameter" (slope of experience curve on double-log scale)
    b = (x[performance] - x[performance].shift(periods=-1)) / (
        x[experience] - x[experience].shift(periods=-1)
    )

    # translate to "learning rate" (e.g., cost reduction per doubling of capacity)
    return b.apply(lambda y: 1 - math.pow(2, y))
