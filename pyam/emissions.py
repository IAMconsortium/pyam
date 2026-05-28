from pyam.aggregation import aggregate_data
from pyam.exceptions import raise_data_error

REQUIRED_KYOTO_SPECIES = ["Emissions|CO2", "Emissions|CH4", "Emissions|N2O"]

# Variable names follow https://github.com/iamconsortium/common-definitions
# from https://github.com/iiasa/emissions_harmonization_historical/blob/190affcf0caf71daeac395a45dd7c39846acbaec/notebooks/5194_post-process-emissions.py#L81
ALL_KYOTO_SPECIES = [
    # required variables
    "Emissions|CO2",
    "Emissions|CH4",
    "Emissions|N2O",
    # other species defined in common-definitions
    "Emissions|SF6",
    "Emissions|C2F6",
    "Emissions|C6F14",
    "Emissions|CF4",
    "Emissions|HFC|HFC125",
    "Emissions|HFC|HFC134a",
    "Emissions|HFC|HFC143a",
    "Emissions|HFC|HFC227ea",
    "Emissions|HFC|HFC23",
    "Emissions|HFC|HFC245fa",
    "Emissions|HFC|HFC32",
    "Emissions|HFC|HFC43-10",
    # other species *not* defined in common-definitions
    "Emissions|C3F8",
    "Emissions|C4F10",
    "Emissions|C5F12",
    "Emissions|C7F16",
    "Emissions|C8F18",
    "Emissions|cC4F8",
    "Emissions|NF3",
    "Emissions|HFC|HFC152a",
    "Emissions|HFC|HFC236fa",
    "Emissions|HFC|HFC365mfc",
]

SYNOMYMS_KYOTO_SPECIES = {
    "Emissions|HFC|HFC4310mee": "Emissions|HFC|HFC43-10",
    "Emissions|HFC|HFC4310": "Emissions|HFC|HFC43-10",
}


def aggregate_kyoto_ghg(df, metric: str, target_variable: str, target_unit: str):
    """Internal implementation of the `aggregate_kyoto_ghg` function"""

    # Filter and rename the synonyms (this will raise an error if synonyms are given)
    _df = df.filter(variable=ALL_KYOTO_SPECIES + list(SYNOMYMS_KYOTO_SPECIES)).rename(
        variable=SYNOMYMS_KYOTO_SPECIES
    )

    # Check that all required variables are present
    missing = _df.require_data(variable=REQUIRED_KYOTO_SPECIES)
    if missing is not None:
        raise_data_error(
            "Missing emission species required for Kyoto GHG aggregation", missing
        )

    # Convert units
    for unit in _df.unit:
        _df.convert_unit(unit, target_unit, context=metric, inplace=True)

    return aggregate_data(_df, target_variable, components=_df.variable)
