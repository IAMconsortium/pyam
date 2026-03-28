from pyam.exceptions import raise_data_error

REQUIRED_SPECIES = ["Emissions|CO2", "Emissions|CH4", "Emissions|N2O"]

ALL_KYOTO_SPECIES = {
    "Emissions|CO2",
    "Emissions|CH4",
    "Emissions|N2O",
    "Emissions|HFC125",
    "Emissions|HFC134a",
    "Emissions|HFC143a",
    "Emissions|HFC152a",
    "Emissions|HFC227ea",
    "Emissions|HFC23",
    "Emissions|HFC236fa",
    "Emissions|HFC245fa",
    "Emissions|HFC32",
    "Emissions|HFC365mfc",
    "Emissions|HFC4310mee",
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


def aggregate_kyoto_gases(df, metric: str):
    """Internal implementation of the `aggregate_kyoto_gases` function"""

    missing = df.require_data(variable=REQUIRED_SPECIES)
    if missing is not None:
        raise_data_error("Missing species for aggregation", missing)

    df_list = list()
    for species, unit in SPECIES_UNIT_MAPPING.items():
        if species in df.variable:
            df_list.append(
                df.filter(variable=species).convert_unit(
                    unit, "Mt CO2-equiv/yr", context=metric
                )
            )

    return df_list
