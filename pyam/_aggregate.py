import numpy as np
import logging

from pyam.logging import adjust_log_level
from pyam.utils import (
    islistable,
    isstr,
    KNOWN_FUNCS,
    META_IDX,
)

logger = logging.getLogger(__name__)


def _aggregate(df, variable, components=None, method=np.sum):
    """Internal implementation of the `aggregate` function"""
    # list of variables require default components (no manual list)
    if islistable(variable) and components is not None:
        raise ValueError('aggregating by list of variables cannot use '
                         'custom components')

    mapping = {}
    msg = 'cannot aggregate variable `{}` because it has no components'
    # if single variable
    if isstr(variable):
        # default components to all variables one level below `variable`
        components = components or df._variable_components(variable)

        if not len(components):
            logger.info(msg.format(variable))
            return

        for c in components:
            mapping[c] = variable

    # else, use all variables one level below `variable` as components
    else:
        for v in variable if islistable(variable) else [variable]:
            _components = df._variable_components(v)
            if not len(_components):
                logger.info(msg.format(v))
                continue

            for c in _components:
                mapping[c] = v

    # rename all components to `variable` and aggregate
    _df = df.data[df._apply_filters(variable=mapping.keys())].copy()
    _df['variable'].replace(mapping, inplace=True)
    return _group_and_agg(_df, [], method)


def _aggregate_region(self, variable, region, subregions=None,
                      components=False, method='sum', weight=None):
    """Internal implementation for aggregating data over subregions"""
    if not isstr(variable) and components is not False:
        msg = 'aggregating by list of variables with components ' \
              'is not supported'
        raise ValueError(msg)

    if weight is not None and components is not False:
        msg = 'using weights and components in one operation not supported'
        raise ValueError(msg)

    # default subregions to all regions other than `region`
    subregions = subregions or self._all_other_regions(region, variable)

    if not len(subregions):
        msg = 'cannot aggregate variable `{}` to `{}` because it does not'\
              ' exist in any subregion'
        logger.info(msg.format(variable, region))

        return

    # compute aggregate over all subregions
    subregion_df = self.filter(region=subregions)
    rows = subregion_df._apply_filters(variable=variable)
    if weight is None:
        col = 'region'
        _data = _group_and_agg(subregion_df.data[rows], col, method=method)
    else:
        weight_rows = subregion_df._apply_filters(variable=weight)
        _data = _agg_weight(subregion_df.data[rows],
                            subregion_df.data[weight_rows], method)

    # if not `components=False`, add components at the `region` level
    if components is not False:
        with adjust_log_level(logger):
            region_df = self.filter(region=region)

        # if `True`, auto-detect `components` at the `region` level,
        # defaults to variables below `variable` only present in `region`
        if components is True:
            level = dict(level=None)
            r_comps = region_df._variable_components(variable, **level)
            sr_comps = subregion_df._variable_components(variable, **level)
            components = set(r_comps).difference(sr_comps)

        if len(components):
            # rename all components to `variable` and aggregate
            rows = region_df._apply_filters(variable=components)
            _df = region_df.data[rows].copy()
            _df['variable'] = variable
            _data = _data.add(_group_and_agg(_df, 'region'), fill_value=0)

    return _data


def _group_and_agg(df, by, method=np.sum):
    """Groupby & aggregate `df` by column(s), return indexed `pd.Series`"""
    by = [by] if isstr(by) else by
    cols = [c for c in list(df.columns) if c not in ['value'] + by]
    # pick aggregator func (default: sum)
    return df.groupby(cols)['value'].agg(_get_method_func(method))


def _agg_weight(df, weight, method):
    """Aggregate `df` by regions with weights, return indexed `pd.Series`"""
    # only summation allowed with weights
    if method not in ['sum', np.sum]:
        raise ValueError('only method `np.sum` allowed for weighted average')

    cols = _list_diff(df.columns, ['variable', 'unit', 'value'])
    _weight = _get_value_col(weight, cols)

    if not _get_value_col(df, cols).index.equals(_weight.index):
        raise ValueError('inconsistent index between variable and weight')

    _data = _get_value_col(df)
    col1 = _list_diff(_data.index.names, ['region'])
    col2 = META_IDX + [i for i in ['year', 'time'] if i in _weight.index.names]
    return (_data * _weight).groupby(col1).sum() / _weight.groupby(col2).sum()


def _list_diff(lst, exclude):
    """Return the list minus those elements in `exclude`"""
    return [i for i in lst if i not in exclude]


def _get_value_col(df, cols=None):
    """Return the value column as `pd.Series sorted by index"""
    cols = cols or [i for i in df.columns if i != 'value']
    return df.set_index(cols)['value'].sort_index()


def _get_method_func(method):
    """Translate a string to a known method"""
    if not isstr(method):
        return method

    if method in KNOWN_FUNCS:
        return KNOWN_FUNCS[method]

    # raise error if `method` is a string but not in dict of known methods
    raise ValueError('method `{}` is not a known aggregator'.format(method))
