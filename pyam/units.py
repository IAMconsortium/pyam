import logging

import pandas as pd
import pint

from pathlib import Path

logger = logging.getLogger(__name__)

# get application pint.UnitRegistry and load energy-units
_REGISTRY = pint.get_application_registry()
file = Path(__file__).parents[1] / 'units' / 'definitions.txt'
_REGISTRY.load_definitions(str(file))


def convert_unit(df, current, to, factor=None, inplace=False):
    """Internal implementation of unit conversion with explicit kwargs"""
    ret = df.copy() if not inplace else df

    if factor is None:
        factor = _REGISTRY[current].to(_REGISTRY[to]).magnitude
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
