import itertools
import logging
import pandas as pd

from pyam.utils import META_IDX, make_index, s

logger = logging.getLogger(__name__)


def _check_rows(rows, check, in_range=True, return_test="any"):
    """Check all rows to be in/out of a certain range and provide testing on
    return values based on provided conditions

    Parameters
    ----------
    rows : pd.DataFrame
        data rows
    check : dict
        dictionary with possible values of 'up', 'lo', and 'year'
    in_range : bool, optional
        check if values are inside or outside of provided range
    return_test : str, optional
        possible values:
            - 'any': default, return scenarios where check passes for any entry
            - 'all': test if all values match checks, if not, return empty set
    """
    valid_checks = set(["up", "lo", "year"])
    if not set(check.keys()).issubset(valid_checks):
        msg = "Unknown checking type: {}"
        raise ValueError(msg.format(check.keys() - valid_checks))
    if "year" not in check:
        where_idx = set(rows.index)
    else:
        if "time" in rows.index.names:
            _years = rows.index.get_level_values("time").year
        else:
            _years = rows.index.get_level_values("year")
        where_idx = set(rows.index[_years == check["year"]])
        rows = rows.loc[list(where_idx)]

    up_op = rows.values.__le__ if in_range else rows.values.__gt__
    lo_op = rows.values.__ge__ if in_range else rows.values.__lt__

    check_idx = []
    for bd, op in [("up", up_op), ("lo", lo_op)]:
        if bd in check:
            check_idx.append(set(rows.index[op(check[bd])]))

    if return_test == "any":
        ret = where_idx & set.union(*check_idx)
    elif return_test == "all":
        ret = where_idx if where_idx == set.intersection(*check_idx) else set()
    else:
        raise ValueError("Unknown return test: {}".format(return_test))
    return ret


def _apply_criteria(df, criteria, **kwargs):
    """Apply criteria individually to every model/scenario instance"""
    idxs = []
    for var, check in criteria.items():
        _df = df[df.index.get_level_values("variable") == var]
        for group in _df.groupby(META_IDX):
            grp_idxs = _check_rows(group[-1], check, **kwargs)
            idxs.append(grp_idxs)
    df = df.loc[itertools.chain(*idxs)]
    return df


def _exclude_on_fail(df, index):
    """Assign a selection of scenarios as `exclude: True`"""

    if not isinstance(index, pd.MultiIndex):
        index = make_index(index, cols=df.index.names)

    df.exclude[index] = True
    n = len(index)
    logger.info(
        f"{n} scenario{s(n)} failed validation and will be set as `exclude=True`."
    )
