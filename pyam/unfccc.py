import re
import numpy as np

try:
    import unfccc_di_api
    HAS_UNFCCC = True
except ImportError:  # pragma: no cover
    unfccc_di_api = None
    HAS_UNFCCC = False

from pyam import IamDataFrame
from pyam.utils import pattern_match, isstr, islistable
from pyam.string_utils import unformat_subscript

# columns from UNFCCC data that can be used for variable names
NAME_COLS = ['category', 'classification', 'measure', 'gas']

# UNFCCC-reader instance (instantiated at first use)
_READER = None

# mapping from gas as simple string to subscript-format used by UNFCCC DI API
GAS_MAPPING = {
    'CH4': 'CH₄',
    'CO2': 'CO₂',
    'N2O': 'N₂O',
    'NF3': 'NF₃',
    'SF6': 'SF₆',
    'CF4': 'CF₄',
    'C2F6': 'C₂F₆',
    'c-C3F6': 'c-C₃F₆',
    'C3F8': 'C₃F₈',
    'c-C4F8': 'c-C₄F₈',
    'C4F10': 'C₄F₁₀',
    'C5F12': 'C5F₁₂',  # this seems to be a bug in the UNFCCC API
    'C6F14': 'C₆F₁₄',
    'C10F18': 'C₁₀F₁₈',
    'NH3': 'NH₃',
    'NOx': 'NOₓ',
    'SO2': 'SO₂'
}


def read_unfccc(party_code, gases=None, tier=None, mapping=None,
                model='UNFCCC', scenario='Data Inventory'):
    """Read data from the UNFCCC Data Inventory

    This function is a wrappter for the
    :meth:`unfccc_di_api.UNFCCCApiReader.query`.

    The data returned from the UNFCCC Data Inventory is transformed
    into a structure similar to the format used in IPCC reports and
    IAM model comparison projects. For compatibility with the
    `iam-units <https://github.com/IAMconsortium/units>`_ package
    and the :meth:`convert_unit <IamDataFrame.convert_unit>`,
    emissions species are formatted to standard text (:code:`CO2`)
    instead of subscripts (:code:`CO₂`) and the unit ':code:`CO₂ equivalent`'
    used by UNFCCC is replaced by ':code:`CO2e`'.

    Parameters
    ----------
    party_code : str
        ISO3-style code for UNFCCC party (country)
    gases : str or list of str, optional
        Emission species to be queried from the data inventory can be stated
        as subscript-format (:code:`CO₂`) or simple text (:code:`CO2`)
    tier : int or list of int
        Pre-specified groupings of UNFCCC data to a variable naming format
        used in IPCC reports and IAM model comparison projects
    mapping : dict, optional
        Mapping to cast UNFCCC-data columns into IAMC-style variables, e.g.
        ```
        {
            'Emissions|{gas}|Energy': ('1.  Energy', '*', '*', '*')
        }
        ```
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
        raise ImportError('Required package `unfccc-di-api` not found!')

    # check that only one of `tier` or `mapping` is provided
    if (tier is None and mapping is None) or \
            (tier is not None and mapping is not None):
        raise ValueError('Please specify either `tier` or `mapping`!')

    global _READER
    if _READER is None:
        _READER = unfccc_di_api.UNFCCCApiReader()

    # change `gases` kwarg to subscript-format used by UNFCCC
    if gases is not None:
        gases = [GAS_MAPPING.get(g, g)
                 for g in (gases if islistable(gases) else [gases])]

    # retrieve data, drop non-numeric data and base year
    data = _READER.query(party_code=party_code, gases=gases)
    data = data[~np.isnan(data.numberValue)]
    data = data[data.year != 'Base year']

    # create the mapping from the data if `tier` is given
    if tier is not None:
        _category = data.category.unique()
        mapping = {}

        for t in tier if islistable(tier) else [tier]:
            # treatment of tear 1
            if t == 1:
                pattern = re.compile('.\\.  ')  # pattern of top-level category
                for i in [i for i in _category if pattern.match(i)]:
                    key = 'Emissions|{gas}|' + i[4:]
                    mapping[key] = (i, 'Total for category',
                                    'Net emissions/removals', '*')
            else:
                raise ValueError(f'Unknown value for `tier`: {t}')

    # add new `variable` column, iterate over mapping to determine variables
    data['variable'] = None
    for variable, value in mapping.items():
        matches = np.array([True] * len(data))
        for i, col in enumerate(NAME_COLS):
            matches &= pattern_match(data[col], value[i])

        data.loc[matches, 'variable'] = data.loc[matches]\
            .apply(_compile_variable, variable=variable, axis=1)

    # drop unspecified rows and columns, rename value column
    cols = ['party', 'variable', 'unit', 'year', 'gas', 'numberValue']
    data = data.loc[[isstr(i) for i in data.variable], cols]
    data.rename(columns={'numberValue': 'value'}, inplace=True)

    # append `gas` to unit, drop `gas` column
    data.loc[:, 'unit'] = data.apply(_compile_unit, axis=1)
    data.drop(columns='gas', inplace=True)

    # cast to IamDataFrame, unformat subscripts
    df = IamDataFrame(data, model=model, scenario=scenario, region='party')
    for col, values in [('variable', df.variable), ('unit', df.unit)]:
        df.rename({col: _rename_mapping(values)}, inplace=True)

    return df


def _compile_variable(i, variable):
    """Translate UNFCCC columns into an IAMC-style variable"""
    if i['variable']:
        raise ValueError('Conflict in variable mapping!')
    return variable.format(**dict((c, i[c]) for c in NAME_COLS))


def _compile_unit(i):
    """Append gas to unit and update CO2e for pint/iam-unit compatibility"""
    if 'CO₂ equivalent' in i['unit']:
        return i['unit'].replace('CO₂ equivalent', 'CO2e')
    if i['unit'] in ['kt', 't']:
        return ' '.join([i['unit'], i['gas']])
    else:
        return i['unit']


def _rename_mapping(lst):
    """Create a mapping to non-subscripted strings"""
    dct = {}
    for g in lst:
        _g = unformat_subscript(g)
        if _g != g:
            dct[g] = _g
    return dct
