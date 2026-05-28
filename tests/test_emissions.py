import pandas as pd
import pytest

from pyam import IamDataFrame
from pyam.testing import assert_iamframe_equal

EMISSIONS_SPECIES_DATA = pd.DataFrame(
    [
        ["Emissions|CO2", "Mt CO2/yr", 42885.41, 33011.87, 24642.81],
        # the following line must be ignored by the aggregation
        ["Emissions|CO2|Energy", "Mt CO2/yr", 32885.41, 23011.87, 14642.81],
        ["Emissions|CH4", "Mt CH4/yr", 413.63, 287.42, 233.97],
        ["Emissions|N2O", "kt N2O/yr", 11623.95, 9005.23, 8177.40],
        ["Emissions|SF6", "kt SF6/yr", 8.01, 5.26, 2.60],
        ["Emissions|HFC|HFC125", "kt HFC125/yr", 98.76, 57.44, 16.71],
        ["Emissions|HFC|HFC134a", "kt HFC134a/yr", 248.84, 144.53, 42.41],
        ["Emissions|HFC|HFC143a", "kt HFC143a/yr", 40.59, 23.61, 6.87],
        ["Emissions|HFC|HFC23", "kt HFC23/yr", 7.13, 4.24, 1.55],
        ["Emissions|HFC|HFC32", "kt HFC32/yr", 61.18, 35.55, 10.29],
        ["Emissions|HFC|HFC43-10", "kt HFC43-10/yr", 6.41, 7.32, 8.23],
    ],
    columns=["variable", "unit", 2020, 2025, 2030],
)


EXP_GHG_DATA = pd.DataFrame(
    [
        [
            "Emissions|Kyoto Gases [AR6GWP100]",
            "Mt CO2-equiv/yr",
            58948.32,
            44296.02,
            33679.55,
        ]
    ],
    columns=["variable", "unit", 2020, 2025, 2030],
)


# test for different notation of HFC4310
@pytest.mark.parametrize(
    "hfc4310",
    (
        None,
        dict(variable={"Emissions|HFC|HFC43-10": "Emissions|HFC|HFC4310"}),
        dict(unit={"kt HFC43-10/yr": "kt HFC4310/yr"}),
    ),
)
@pytest.mark.parametrize("append", (False, True))
def test_kyoto_ghg(hfc4310, append):
    df_args = dict(model="model_a", scenario="scenario_a", region="World")
    df = IamDataFrame(EMISSIONS_SPECIES_DATA, **df_args)
    df.rename(hfc4310, inplace=True)
    exp = IamDataFrame(EXP_GHG_DATA, **df_args)

    if append:
        obs = df.copy()
        obs.aggregate_kyoto_ghg(metric="AR6GWP100", append=append)
        exp = df.append(exp)
    else:
        obs = df.aggregate_kyoto_ghg(metric="AR6GWP100")

    assert_iamframe_equal(exp, obs)


def test_kyoto_ghg_missing_species_raises():
    df_args = dict(model="model_a", scenario="scenario_a", region="World")
    df = IamDataFrame(EMISSIONS_SPECIES_DATA, **df_args)
    df.filter(variable="Emissions|CH4", keep=False, inplace=True)

    match = "Missing species for aggregation:.* scenario_a  Emissions|CH4"
    with pytest.raises(ValueError, match=match):
        df.aggregate_kyoto_ghg(metric="AR6GWP100")


def test_kyoto_duplicate_hfc4310_raises():
    df_args = dict(model="model_a", scenario="scenario_a", region="World")
    df = IamDataFrame(EMISSIONS_SPECIES_DATA, **df_args)

    rename_mapping = {"Emissions|HFC|HFC43-10": "Emissions|HFC|HFC4310mee"}
    df.append(
        df.filter(variable=list(rename_mapping)).rename(variable=rename_mapping),
        inplace=True,
    )

    match = "Conflicting data rows after renaming:.* Emissions|HFC|HFC43-10"
    with pytest.raises(ValueError, match=match):
        df.aggregate_kyoto_ghg(metric="AR6GWP100")
