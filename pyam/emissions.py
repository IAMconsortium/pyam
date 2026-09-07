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

SYNONYMS_GCAGES_SPECIES = {
    "Emissions|HFC125": "Emissions|HFC|HFC125",
    "Emissions|HFC134a": "Emissions|HFC|HFC134a",
    "Emissions|HFC143a": "Emissions|HFC|HFC143a",
    "Emissions|HFC227ea": "Emissions|HFC|HFC227ea",
    "Emissions|HFC23": "Emissions|HFC|HFC23",
    "Emissions|HFC245fa": "Emissions|HFC|HFC245fa",
    "Emissions|HFC32": "Emissions|HFC|HFC32",
    "Emissions|HFC4310mee": "Emissions|HFC|HFC43-10",
    "Emissions|HFC152a": "Emissions|HFC|HFC152a",
    "Emissions|HFC236fa": "Emissions|HFC|HFC236fa",
    "Emissions|HFC365mfc": "Emissions|HFC|HFC365mfc",
}


def aggregate_kyoto_ghg(df, metric: str, target_variable: str, target_unit: str):
    """Internal implementation of the `aggregate_kyoto_ghg` function"""

    # check for GCAGES variable names and rename if present
    if any([species in df.variable for species in SYNONYMS_GCAGES_SPECIES]):
        df = df.rename(variable=SYNONYMS_GCAGES_SPECIES)

    # Filter and rename the synonyms (this will raise an error if synonyms are given)
    df = df.filter(variable=ALL_KYOTO_SPECIES + list(SYNOMYMS_KYOTO_SPECIES)).rename(
        variable=SYNOMYMS_KYOTO_SPECIES
    )

    # Check that all required variables are present
    missing = df.require_data(variable=REQUIRED_KYOTO_SPECIES)
    if missing is not None:
        raise_data_error(
            "Missing emission species required for Kyoto GHG aggregation", missing
        )

    # Convert units
    for unit in df.unit:
        df.convert_unit(unit, target_unit, context=metric, inplace=True)

    return aggregate_data(df, target_variable, components=df.variable)
