from pyam.aggregation import aggregate_data
from pyam.exceptions import raise_data_error

REQUIRED_KYOTO_SPECIES = ["Emissions|CO2", "Emissions|CH4", "Emissions|N2O"]


ALL_KYOTO_SPECIES = {
    "Emissions|CO2",
    "Emissions|CH4",
    "Emissions|N2O",
    "Emissions|HFC|HFC125",
    "Emissions|HFC|HFC134a",
    "Emissions|HFC|HFC143a",
    "Emissions|HFC|HFC152a",
    "Emissions|HFC|HFC227ea",
    "Emissions|HFC|HFC23",
    "Emissions|HFC|HFC236fa",
    "Emissions|HFC|HFC245fa",
    "Emissions|HFC|HFC32",
    "Emissions|HFC|HFC365mfc",
    "Emissions|HFC|HFC4310mee",
    "Emissions|NF3",
    "Emissions|SF6",
    "Emissions|C2F6",
    "Emissions|C3F8",
    "Emissions|C4F10",
    "Emissions|C5F12",
    "Emissions|C6F14",
    "Emissions|C7F16",
    "Emissions|C8F18",
    "Emissions|CF4",
    "Emissions|cC4F8",
}


def aggregate_kyoto_ghg(df, metric: str, target_variable: str, target_unit: str):
    """Internal implementation of the `aggregate_kyoto_ghg` function"""

    _df = df.filter(variable=ALL_KYOTO_SPECIES)

    missing = _df.require_data(variable=REQUIRED_KYOTO_SPECIES)
    if missing is not None:
        raise_data_error(
            "Missing emission species required for Kyoto GHG aggregation", missing
        )

    for unit in _df.unit:
        _df.convert_unit(unit, target_unit, context=metric, inplace=True)

    return aggregate_data(_df, target_variable, components=_df.variable)
