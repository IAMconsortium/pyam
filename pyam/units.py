import logging

import pandas as pd
import pint

from pathlib import Path

logger = logging.getLogger(__name__)

# get application pint.UnitRegistry and load energy-units
_REGISTRY = pint.get_application_registry()
file = Path(__file__).parents[1] / 'units' / 'definitions.txt'
_REGISTRY.load_definitions(str(file))


def convert_unit(df, current, to, factor=None, registry=None, context=None,
                 inplace=False):
    """Internal implementation of unit conversion with explicit kwargs"""
    ret = df.copy() if not inplace else df

    # check that (only) either factor or registry/context is provided
    if factor is not None and \
            any([i is not None for i in [registry, context]]):
        raise ValueError('use either `factor` or `pint.UnitRegistry`')

    # check that custom registry is valid
    if registry is not None and not isinstance(registry, pint.UnitRegistry):
        raise ValueError(f'registry` is not a valid UnitRegistry: {registry}')

    # if factor is not given, get it from custom or application registry
    if factor is None:
        _reg = registry or _REGISTRY
        args = [_reg[to]] if context is None else [_reg[to], context]
        factor = _reg[current].to(*args).magnitude

    # do the conversion
    where = ret.data['unit'] == current
    ret.data.loc[where, 'value'] *= factor
    ret.data.loc[where, 'unit'] = to

    if not inplace:
        return ret


def convert_unit_with_mapping(df, conversion_mapping, inplace=False):
    """Internal implementation of unit conversion by mapping (deprecated)"""
    # TODO: deprecate in next release (>=0.6.0)
    ret = df.copy() if not inplace else df
    for current_unit, (target_unit, factor) in conversion_mapping.items():
        factor = pd.to_numeric(factor)
        where = ret.data['unit'] == current_unit
        ret.data.loc[where, 'value'] *= factor
        ret.data.loc[where, 'unit'] = target_unit
    if not inplace:
        return ret
