import pandas as pd
import numpy as np
import logging

from pyam.logging import adjust_log_level
from pyam.utils import (
    islistable,
    isstr,
    find_depth,
    reduce_hierarchy,
    KNOWN_FUNCS
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


def _aggregate_recursive(df, variable, method=np.sum):
    """Recursive aggregation along the variable tree"""
    _df_aggregated = None
    _df = df.copy()

    # iterate over variables to find all subcategories to be aggregated
    sub_variables = []
    for d in reversed(range(1, max(find_depth(df.data.variable)) + 1)):
        depth = find_depth(df.data.variable)
        var_list = (
            df.data.variable[[i == d for i in depth]]
            .unique()
        )
        vars_up = pd.Series(
            [reduce_hierarchy(i, -1) for i in var_list]).unique()

        if [i for i, entr in enumerate(vars_up) if entr.startswith(variable)]:
            for v in vars_up:
                sub_variables.append(v)

    sub_variables = reversed(sorted(set(sub_variables)))

    # iterate over subcategories (bottom-up) and perform aggregation
    for entry in sub_variables:
        _df.aggregate(variable=entry, append=True)
        _df_temp = _df.aggregate(variable=entry, append=False)

        if _df_aggregated is None:
            _df_aggregated = _df_temp.copy()
        else:
            _df_aggregated.append(_df_temp, inplace=True)

    return _df_aggregated.data


def _aggregate_region(df, variable, region, subregions=None, components=False,
                      method='sum', weight=None):
    """Internal implementation for aggregating data over subregions"""
    if not isstr(variable) and components is not False:
        msg = 'aggregating by list of variables with components ' \
              'is not supported'
        raise ValueError(msg)

    if weight is not None and components is not False:
        msg = 'using weights and components in one operation not supported'
        raise ValueError(msg)

    # default subregions to all regions other than `region`
    subregions = subregions or df._all_other_regions(region, variable)

    if not len(subregions):
        msg = 'cannot aggregate variable `{}` to `{}` because it does not'\
              ' exist in any subregion'
        logger.info(msg.format(variable, region))

        return

    # compute aggregate over all subregions
    subregion_df = df.filter(region=subregions)
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
            region_df = df.filter(region=region)

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


def _aggregate_time(df, variable, column, value, components, method=np.sum):
    """Internal implementation for aggregating data over subannual time"""
    # default `components` to all entries in `column` other than `value`
    if components is None:
        components = list(set(df.data.subannual.unique()) - set([value]))

    # compute aggregate over time
    filter_args = dict(variable=variable)
    filter_args[column] = components
    index = _list_diff(df.data.columns, [column, 'value'])

    _data = pd.concat(
        [
            df.filter(**filter_args).data
            .pivot_table(index=index, columns=column)
            .value
            .rename_axis(None, axis=1)
            .apply(_get_method_func(method), axis=1)
        ], names=[column] + index, keys=[value])

    # reset index-level order to original IamDataFrame
    _data.index = _data.index.reorder_levels(df._LONG_IDX)

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

    w_cols = _list_diff(df.columns, ['variable', 'unit', 'value'])
    _weight = _get_value_col(weight, w_cols)

    if not _get_value_col(df, w_cols).index.equals(_weight.index):
        raise ValueError('inconsistent index between variable and weight')

    _data = _get_value_col(df)
    col1 = _list_diff(_data.index.names, ['region'])
    col2 = _list_diff(w_cols, ['region'])
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
