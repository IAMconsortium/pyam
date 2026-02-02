from pyam.exceptions import raise_data_error

REQUIRED_SPECIES = ["Emissions|CO2", "Emissions|CH4", "Emissions|N2O"]

SPECIES_UNIT_MAPPING = {
    "Emissions|CO2": "Mt CO2/yr",
    "Emissions|CH4": "Mt CH4/yr",
    "Emissions|N2O": "kt N2O/yr",
    "Emissions|SF6": "kt SF6/yr",
    "Emissions|C2F6": "kt C2F6/yr",
    "Emissions|C6F14": "kt C6F14/yr",
    "Emissions|CF4": "kt CF4/yr",
    "Emissions|HFC|HFC125": "kt HFC125/yr",
    "Emissions|HFC|HFC134a": "kt HFC134a/yr",
    "Emissions|HFC|HFC143a": "kt HFC143a/yr",
    "Emissions|HFC|HFC227ea": "kt HFC227ea/yr",
    "Emissions|HFC|HFC23": "kt HFC23/yr",
    "Emissions|HFC|HFC32": "kt HFC32/yr",
    # inconsistent notation between iam-units and common-definitions
    #    "Emissions|HFC|HFC43-10": "kt HFC43-10/yr",
}


def aggregate_kyoto_gases(df: "IamDataFrame", metric: str):
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
