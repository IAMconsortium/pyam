import logging

import iam_units
import pandas as pd
import pint


logger = logging.getLogger(__name__)


# Thin wrapper around pint.UndefinedUnitError to provide a usage hint
class UndefinedUnitError(pint.UndefinedUnitError):
    def __str__(self):
        return super().__str__() + (
            "\nMust provide IamDataFrame.convert_unit(..., context=...) to "
            "convert GHG species")


def convert_unit(df, current, to, factor=None, registry=None, context=None,
                 inplace=False):
    """Internal implementation of unit conversion with explicit kwargs"""
    ret = df.copy() if not inplace else df

    # Mask for rows having *current* units, to be converted
    where = ret.data['unit'] == current

    if factor:
        # Short code path: use an explicit conversion factor, don't use pint
        ret.data.loc[where, 'value'] *= factor
        ret.data.loc[where, 'unit'] = to
        return None if inplace else ret

    # Convert using a pint.UnitRegistry; default the one from iam_units
    registry = registry or iam_units.registry

    # Tuple of (magnitude, unit)
    qty = (ret.data.loc[where, 'value'].values, current)

    try:
        # Create a vector pint.Quantity
        qty = registry.Quantity(*qty)
    except pint.UndefinedUnitError as exc:
        # *current* might include a GHG species
        if not context:
            # Can't do anything without a context
            raise UndefinedUnitError(*exc.args) from None

        result, to = convert_gwp(context, qty, to)
    except AttributeError:
        # .Quantity() did not exist
        raise TypeError(f'{registry} must be `pint.UnitRegistry`') from None
    else:
        # Ordinary conversion, using an empty Context if none was provided
        result = qty.to(to, context or pint.Context())

    # Copy values from the result Quantity
    ret.data.loc[where, 'value'] = result.magnitude

    # Assign output units
    ret.data.loc[where, 'unit'] = to

    return None if inplace else ret


def convert_gwp(context, qty, to):
    """Helper for :meth:`convert_unit` to perform GWP conversions."""
    # Remove a leading 'gwp_' to produce the metric name
    metric = context.split('gwp_')[1]

    # Split *to* into a 1- or 3-tuple of str. This allows for *to* to be:
    _to = iam_units.emissions.pattern.split(to, maxsplit=1)
    if len(_to) == 1:
        # Only a species name ('CO2e') without any unit
        species_to = _to[0]
        units_to = None
    else:
        # An expression with both units and species name ('kg CO2e / year');
        # to[1] is the species
        species_to = _to[1]
        # Other elements are pre- and suffix, e.g. 'kg ' and ' / year'
        units_to = _to[0] + _to[2]

    # Convert GWP using the (magnitude, unit-and-species) tuple in *qty*
    result = iam_units.convert_gwp(metric, qty, species_to)

    if units_to:
        # Also convert the units
        result = result.to(units_to)
    else:
        # *to* was only a species name; provide units based on input and the
        # output species name
        to = iam_units.format_mass(result, species_to, spec=':~')

    return result, to


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
