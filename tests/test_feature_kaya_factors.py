import pandas as pd
import pytest

from pyam import IamDataFrame
from pyam.kaya import input_variable_names, kaya_factor_names, kaya_variable_names
from pyam.testing import assert_iamframe_equal

TEST_DF = IamDataFrame(
    pd.DataFrame(
        [
            [input_variable_names.POPULATION, "million", 1000],
            [input_variable_names.GDP_PPP, "billion USD_2005/yr", 6],
            [input_variable_names.GDP_MER, "billion USD_2005/yr", 5],
            [input_variable_names.FINAL_ENERGY, "EJ/yr", 8],
            [input_variable_names.PRIMARY_ENERGY, "EJ/yr", 10],
            [input_variable_names.PRIMARY_ENERGY_COAL, "EJ/yr", 5],
            [input_variable_names.PRIMARY_ENERGY_GAS, "EJ/yr", 2],
            [input_variable_names.PRIMARY_ENERGY_OIL, "EJ/yr", 2],
            [
                input_variable_names.EMISSIONS_CO2_FOSSIL_FUELS_AND_INDUSTRY,
                "Mt CO2/yr",
                10,
            ],
            [input_variable_names.EMISSIONS_CO2_INDUSTRIAL_PROCESSES, "Mt CO2/yr", 1],
            [input_variable_names.EMISSIONS_CO2_AFOLU, "Mt CO2/yr", 1],
            [input_variable_names.EMISSIONS_CO2_CCS, "Mt CO2/yr", 4],
            [input_variable_names.EMISSIONS_CO2_CCS_BIOMASS, "Mt CO2/yr", 1],
            [input_variable_names.CCS_FOSSIL_ENERGY, "Mt CO2/yr", 2],
            [input_variable_names.CCS_FOSSIL_INDUSTRY, "Mt CO2/yr", 1],
            [input_variable_names.CCS_BIOMASS_ENERGY, "Mt CO2/yr", 0.5],
            [input_variable_names.CCS_BIOMASS_INDUSTRY, "Mt CO2/yr", 0.5],
        ],
        columns=["variable", "unit", 2010],
    ),
    model="model_a",
    scenario="scen_a",
    region="World",
)

EXP_DF = IamDataFrame(
    pd.DataFrame(
        [
            [kaya_factor_names.FE_per_GNP, "EJ / USD / billion", 1.33333],
            [kaya_factor_names.GNP_per_P, "USD * billion / million / a", 0.006000],
            [kaya_factor_names.NFC_per_TFC, "", 0.833333],
            [kaya_factor_names.PEdeq_per_FE, "", 1.250000],
            [kaya_factor_names.PEFF_per_PEDEq, "", 0.900000],
            [kaya_factor_names.TFC_per_PEFF, "Mt CO2/EJ", 1.333333],
            [input_variable_names.POPULATION, "million", 1000],
            [kaya_variable_names.TFC, "Mt CO2/yr", 12.0],
        ],
        columns=["variable", "unit", 2010],
    ),
    model="model_a",
    scenario="scen_a",
    region="World",
)

# can't append EXP_DF to TEST_DF because of overlapping values
# append this dataframe to create full results for comparison
EXP_DF_FOR_APPEND = IamDataFrame(
    pd.DataFrame(
        [
            [kaya_factor_names.FE_per_GNP, "EJ / USD / billion", 1.33333],
            [kaya_factor_names.GNP_per_P, "USD * billion / million / a", 0.006000],
            [kaya_factor_names.NFC_per_TFC, "", 0.833333],
            [kaya_factor_names.PEdeq_per_FE, "", 1.250000],
            [kaya_factor_names.PEFF_per_PEDEq, "", 0.900000],
            [kaya_factor_names.TFC_per_PEFF, "Mt CO2/EJ", 1.333333],
            [kaya_variable_names.TFC, "Mt CO2/yr", 12.0],
        ],
        columns=["variable", "unit", 2010],
    ),
    model="model_a",
    scenario="scen_a",
    region="World",
)


@pytest.mark.parametrize("append", (False, True))
def test_kaya_factors(append):
    """Test computing kaya factors"""

    if append:
        obs = TEST_DF.copy()
        obs.compute.kaya_factors(
            scenarios=[("model_a", "scen_a", "World")], append=True
        )
        assert_iamframe_equal(TEST_DF.append(EXP_DF_FOR_APPEND), obs)
    else:
        obs = TEST_DF.compute.kaya_factors(scenarios=[("model_a", "scen_a", "World")])
        assert_iamframe_equal(EXP_DF, obs)


@pytest.mark.parametrize("append", (False, True))
def test_kaya_variables_none_when_input_variables_missing(append):
    """Assert that computing kaya variables with
    missing input variables returns None
    """

    if append:
        obs = TEST_DF.copy()
        # select subset of required input variables
        (
            obs.filter(variable=input_variable_names.POPULATION).compute.kaya_factors(
                scenarios=[("model_a", "scen_a", "World")], append=True
            )
        )
        # assert that no data was added
        assert_iamframe_equal(TEST_DF, obs)
    else:
        obs = TEST_DF.filter(
            variable=input_variable_names.POPULATION
        ).compute.kaya_factors(scenarios=[("model_a", "scen_a", "World")])
        assert obs is None


def test_calling_kaya_factors_multiple_times():
    """Test calling the method a second time has no effect"""

    obs = TEST_DF.copy()
    obs.compute.kaya_factors(scenarios=[("model_a", "scen_a", "World")], append=True)
    obs.compute.kaya_factors(scenarios=[("model_a", "scen_a", "World")], append=True)
    assert_iamframe_equal(TEST_DF.append(EXP_DF_FOR_APPEND), obs)
