import logging
import re

import iam_units
import pint

from pyam.logging import deprecation_warning
from pyam.index import replace_index_values

logger = logging.getLogger(__name__)


def convert_unit(
    df, current, to, factor=None, registry=None, context=None, inplace=False
):
    """Internal implementation of unit conversion with explicit kwargs"""
    ret = df.copy() if not inplace else df

    # Mask for rows having *current* units to be converted, args for replace
    try:
        where = ret._data.index.get_loc_level(current, "unit")[0]
    except KeyError:
        return None if inplace else ret

    index_args = [ret._data, "unit", {current: to}]

    if factor:
        # Short code path: use an explicit conversion factor, don't use pint
        ret._data.loc[where] *= factor
        ret._data.index = replace_index_values(*index_args)
        return None if inplace else ret

    # Convert using a pint.UnitRegistry; default the one from iam_units
    registry = registry or iam_units.registry

    # Make versions without -equiv
    _current, _to = [i.replace("-equiv", "") for i in [current, to]]
    # Pair of (magnitude, unit)
    qty = [ret._data.loc[where].values, _current]

    try:
        # Create a vector pint.Quantity and convert it ordinarily
        result = registry.Quantity(*qty).to(
            _to, context if context is not None else pint.Context()
        )
    except pint.UndefinedUnitError:
        # *qty* might include a GHG species; try GWP conversion
        result, _ = convert_gwp(context, qty, _to)
    except AttributeError:
        # .Quantity() did not exist
        raise TypeError(f"{registry} must be `pint.UnitRegistry`") from None

    # Copy values from the result Quantity and assign units
    ret._data.loc[where] = result.magnitude
    ret._data.index = replace_index_values(*index_args)

    return None if inplace else ret


# GWP conversion using iam_units

#: Supported lower-case aliases for chemical symbols of GHG species. See
#: :meth:`.convert_unit`.
# Keys and values can only differ in case; other items will have no effect.
SPECIES_ALIAS = {
    "ch4": "CH4",
    "co2": "CO2",
    "CO2-equiv": "CO2e",
    "co2_eq": "CO2_eq",
    "co2e": "CO2e",
    "co2eq": "CO2eq",
    "n2o": "N2O",
    "nh3": "NH3",
    "nox": "NOx",
}


# Thin wrapper around pint.UndefinedUnitError to provide a usage hint
class UndefinedUnitError(pint.UndefinedUnitError):
    def __str__(self):
        return super().__str__() + (
            "\nGWP conversion with IamDataFrame.convert_unit() requires a "
            "*context* and mass-based *to* units."
        )


def extract_species(expr):
    """Handle supported expressions for GHG species and units."""
    # Split *expr* into 1 or 3 strings. Unlike iam_units, re.IGNORECASE is used
    # to match e.g. lower-case 'ch4'.
    parts = re.split(
        iam_units.emissions.pattern.pattern, expr, maxsplit=1, flags=re.IGNORECASE
    )

    if len(parts) == 1:
        # No split occurred. *expr* is only a species ('CO2e') without units.
        species, units = parts[0], None
    else:
        # An expression with both units and species name ('kg CO2e / year').
        # parts[1] is the species, others are pre-/suffix ('kg ', ' / year').
        species, units = parts[1], (parts[0] + parts[2])

    # Convert allowed lower-case aliases to chemical symbols
    return SPECIES_ALIAS.get(species, species), units


def convert_gwp(context, qty, to):
    """Helper for :meth:`convert_unit` to perform GWP conversions."""
    # Remove a leading 'gwp_' to produce the metric name
    if context is not None and context.startswith("gwp_"):
        context = context[len("gwp_") :]
        deprecation_warning(
            f"Use context='{context}' instead",
            type='Prefixing a context with "gwp_"',
            stacklevel=5,
        )
    metric = context

    # Extract the species from *qty* and *to*, allowing supported aliases
    species_from, units_from = extract_species(qty[1])
    species_to, units_to = extract_species(to)

    try:
        # Convert using a (magnitude, unit) tuple with only units, and explicit
        # input and output units
        result = iam_units.convert_gwp(
            metric, (qty[0], units_from), species_from, species_to
        )
    except (AttributeError, ValueError):
        # Missing *metric*, or *species_to* contains invalid units. pyam
        # promises UndefinedUnitError in these cases. Use a subclass (above) to
        # add a usage hint.
        raise UndefinedUnitError(species_to) from None
    except pint.DimensionalityError:
        # Provide an exception with the user's inputs
        raise pint.DimensionalityError(qty[1], to) from None

    # Other exceptions are not caught and will pass up through convert_unit()

    if units_to:
        # Also convert the units
        result = result.to(units_to)
    else:
        # *to* was only a species name. Provide units based on input and the
        # output species name.
        to = iam_units.format_mass(result, species_to, spec=":~")

    return result, to
