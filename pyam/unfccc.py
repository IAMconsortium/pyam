import re
import numpy as np

try:
    import unfccc_di_api

    HAS_UNFCCC = True
except ImportError:  # pragma: no cover
    unfccc_di_api = None
    HAS_UNFCCC = False

from pyam import IamDataFrame
from pyam.utils import pattern_match, isstr, to_list

# columns from UNFCCC data that can be used for variable names
NAME_COLS = ["category", "classification", "measure", "gas"]

# UNFCCC-reader instance (instantiated at first use)
_READER = None


def read_unfccc(
    party_code,
    gases=None,
    tier=None,
    mapping=None,
    model="UNFCCC",
    scenario="Data Inventory",
):
    """Read data from the UNFCCC Data Inventory

    This function is a wrappter for the
    :meth:`unfccc_di_api.UNFCCCApiReader.query`.

    The data returned from the UNFCCC Data Inventory is transformed
    into a structure similar to the format used in IPCC reports and
    IAM model comparison projects. For compatibility with the
    `iam-units <https://github.com/IAMconsortium/units>`_ package
    and the :meth:`convert_unit <IamDataFrame.convert_unit>`,
    emissions species are formatted to standard text ('CO2')
    instead of subscripts ('CO₂') and the unit 'CO₂ equivalent'
    used by UNFCCC is replaced by 'CO2e'.

    Parameters
    ----------
    party_code : str
        ISO3-style code for UNFCCC party (country)
    gases : str or list of str, optional
        Emission species to be queried from the data inventory can be stated
        as subscript-format ('CO₂') or simple text ('CO2')
    tier : int or list of int
        Pre-specified groupings of UNFCCC data to a variable naming format
        used in IPCC reports and IAM model comparison projects
    mapping : dict, optional
        Mapping to cast UNFCCC-data columns into IAMC-style variables, e.g.

        .. code-block:: python

            {
                'Emissions|{gas}|Energy': ('1.  Energy', '*', '*', '*'),
            }

        where the tuple corresponds to filters for the columns
        `['category', 'classification', 'measure', 'gas']`
        and `{<col>}` tags in the key are replaced by the column value.
    model : str, optional
        Name to be used as model identifier
    scenario : str, optional
        Name to be used as scenario identifier

    Returns
    -------
    :class:`IamDataFrame`
    """
    if not HAS_UNFCCC:  # pragma: no cover
        raise ImportError("Required package `unfccc-di-api` not found!")

    # check that only one of `tier` or `mapping` is provided
    if (tier is None and mapping is None) or (tier is not None and mapping is not None):
        raise ValueError("Please specify either `tier` or `mapping`!")

    global _READER
    if _READER is None:
        _READER = unfccc_di_api.UNFCCCApiReader()

    # retrieve data, drop non-numeric data and base year
    data = _READER.query(party_code=party_code, gases=to_list(gases))
    data = data[~np.isnan(data.numberValue)]
    data = data[data.year != "Base year"]

    # create the mapping from the data if `tier` is given
    if tier is not None:
        _category = data.category.unique()
        mapping = {}

        for t in to_list(tier):
            # treatment of tear 1
            if t == 1:
                pattern = re.compile(".\\.  ")  # pattern of top-level category
                for i in [i for i in _category if pattern.match(i)]:
                    key = "Emissions|{gas}|" + i[4:]
                    mapping[key] = (
                        i,
                        "Total for category",
                        "Net emissions/removals",
                        "*",
                    )
            else:
                raise ValueError(f"Unknown value for `tier`: {t}")

    # add new `variable` column, iterate over mapping to determine variables
    data["variable"] = None
    for variable, value in mapping.items():
        matches = np.array([True] * len(data))
        for i, col in enumerate(NAME_COLS):
            matches &= pattern_match(data[col], value[i])

        data.loc[matches, "variable"] = data.loc[matches].apply(
            _compile_variable, variable=variable, axis=1
        )

    # drop unspecified rows and columns, rename value column
    cols = ["party", "variable", "unit", "year", "gas", "numberValue"]
    data = data.loc[[isstr(i) for i in data.variable], cols]
    data.rename(columns={"numberValue": "value"}, inplace=True)

    # append `gas` to unit, drop `gas` column
    data.loc[:, "unit"] = data.apply(_compile_unit, axis=1)
    data.drop(columns="gas", inplace=True)

    return IamDataFrame(data, model=model, scenario=scenario, region="party")


def _compile_variable(i, variable):
    """Translate UNFCCC columns into an IAMC-style variable"""
    if i["variable"]:
        raise ValueError("Conflict in variable mapping!")
    return variable.format(**dict((c, i[c]) for c in NAME_COLS))


def _compile_unit(i):
    """Append gas to unit and update CO2e for pint/iam-unit compatibility"""
    if " equivalent" in i["unit"]:
        return i["unit"].replace("CO2 equivalent", "CO2e")
    if i["unit"] in ["kt", "t"]:
        return " ".join([i["unit"], i["gas"]])
    else:
        return i["unit"]
