import pandas as pd
import pytest

from pyam import IamDataFrame
from pyam.testing import assert_iamframe_equal

EMISSIONS_SPECIES_DATA = pd.DataFrame(
    [
        ["Emissions|CO2", "Mt CO2/yr", 42885.41, 33011.87, 24642.81],
        ["Emissions|CH4", "Mt CH4/yr", 413.63, 287.42, 233.97],
        ["Emissions|N2O", "kt N2O/yr", 11623.95, 9005.23, 8177.40],
        ["Emissions|SF6", "kt SF6/yr", 8.01, 5.26, 2.60],
        ["Emissions|HFC|HFC125", "kt HFC125/yr", 98.76, 57.44, 16.71],
        ["Emissions|HFC|HFC134a", "kt HFC134a/yr", 248.84, 144.53, 42.41],
        ["Emissions|HFC|HFC143a", "kt HFC143a/yr", 40.59, 23.61, 6.87],
        ["Emissions|HFC|HFC23", "kt HFC23/yr", 7.13, 4.24, 1.55],
        ["Emissions|HFC|HFC32", "kt HFC32/yr", 61.18, 35.55, 10.29],
    ],
    columns=["variable", "unit", 2020, 2025, 2030],
)


EXP_GHG_DATA = pd.DataFrame(
    [
        [
            "Emissions|Kyoto Gases [AR6GWP100]",
            "Mt CO2-equiv/yr",
            58938.34,
            44284.49,
            33666.64,
        ]
    ],
    columns=["variable", "unit", 2020, 2025, 2030],
)


@pytest.mark.parametrize("append", ((False, True)))
def test_kyoto_ghg(append):
    df_args = dict(model="model_a", scenario="scenario_a", region="World")
    df = IamDataFrame(EMISSIONS_SPECIES_DATA, **df_args)
    exp = IamDataFrame(EXP_GHG_DATA, **df_args)

    if append:
        obs = df.copy()
        obs.aggregate_kyoto_gases(metric="AR6GWP100", append=append)
        exp = df.append(exp)
    else:
        obs = df.aggregate_kyoto_gases(metric="AR6GWP100")

    assert_iamframe_equal(exp, obs)


def test_kyoto_ghg_raises():
    df_args = dict(model="model_a", scenario="scenario_a", region="World")
    df = IamDataFrame(EMISSIONS_SPECIES_DATA, **df_args)
    df.filter(variable="Emissions|CH4", keep=False, inplace=True)

    match = "Missing species for aggregation:.* scenario_a  Emissions|CH4"
    with pytest.raises(ValueError, match=match):
        df.aggregate_kyoto_gases(metric="AR6GWP100")
