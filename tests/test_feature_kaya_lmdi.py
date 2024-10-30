import pandas as pd
import pytest

from pyam import IamDataFrame
from pyam.kaya import input_variable_names, lmdi_names
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
).append(
    IamDataFrame(
        pd.DataFrame(
            [
                [input_variable_names.POPULATION, "million", 1001],
                [input_variable_names.GDP_PPP, "billion USD_2005/yr", 7],
                [input_variable_names.GDP_MER, "billion USD_2005/yr", 6],
                [input_variable_names.FINAL_ENERGY, "EJ/yr", 9],
                [input_variable_names.PRIMARY_ENERGY, "EJ/yr", 11],
                [input_variable_names.PRIMARY_ENERGY_COAL, "EJ/yr", 6],
                [input_variable_names.PRIMARY_ENERGY_GAS, "EJ/yr", 3],
                [input_variable_names.PRIMARY_ENERGY_OIL, "EJ/yr", 3],
                [
                    input_variable_names.EMISSIONS_CO2_FOSSIL_FUELS_AND_INDUSTRY,
                    "Mt CO2/yr",
                    13,
                ],
                [
                    input_variable_names.EMISSIONS_CO2_INDUSTRIAL_PROCESSES,
                    "Mt CO2/yr",
                    2,
                ],
                [input_variable_names.EMISSIONS_CO2_AFOLU, "Mt CO2/yr", 2],
                [input_variable_names.EMISSIONS_CO2_CCS, "Mt CO2/yr", 5],
                [input_variable_names.EMISSIONS_CO2_CCS_BIOMASS, "Mt CO2/yr", 2],
                [input_variable_names.CCS_FOSSIL_ENERGY, "Mt CO2/yr", 3],
                [input_variable_names.CCS_FOSSIL_INDUSTRY, "Mt CO2/yr", 2],
                [input_variable_names.CCS_BIOMASS_ENERGY, "Mt CO2/yr", 1.5],
                [input_variable_names.CCS_BIOMASS_INDUSTRY, "Mt CO2/yr", 1.5],
            ],
            columns=["variable", "unit", 2010],
        ),
        model="model_a",
        scenario="scen_b",
        region="World",
    )
)


EXP_DF = IamDataFrame(
    pd.DataFrame(
        [
            [lmdi_names.FE_per_GNP_LMDI, "unknown", 1.321788],
            [lmdi_names.GNP_per_P_LMDI, "unknown", 0],
            [lmdi_names.PEdeq_per_FE_LMDI, "unknown", 0.816780],
            [lmdi_names.PEFF_per_PEDEq_LMDI, "unknown", 0],
            [lmdi_names.Pop_LMDI, "unknown", 0],
            [lmdi_names.TFC_per_PEFF_LMDI, "unknown", 4.853221],
        ],
        columns=["variable", "unit", 2010],
    ),
    model="model_a::model_a",
    scenario="scen_a::scen_b",
    region="World::World",
)


@pytest.mark.parametrize("append", (False, True))
def test_kaya_lmdi(append):
    """Test computing kaya LMDI"""

    if append:
        obs = TEST_DF.copy()
        obs.compute.kaya_lmdi(
            ref_scenario=("model_a", "scen_a", "World"),
            int_scenario=("model_a", "scen_b", "World"),
            append=True,
        )
        assert_iamframe_equal(TEST_DF.append(EXP_DF), obs)
    else:
        obs = TEST_DF.compute.kaya_lmdi(
            ref_scenario=("model_a", "scen_a", "World"),
            int_scenario=("model_a", "scen_b", "World"),
        )
        assert_iamframe_equal(EXP_DF, obs)


@pytest.mark.parametrize("append", (False, True))
def test_kaya_lmdi_none_when_input_variables_missing(append):
    """Assert that computing kaya LMDI with
    missing input variables returns None
    """

    if append:
        obs = TEST_DF.copy()
        # select subset of required input variables
        (
            obs.filter(variable=input_variable_names.POPULATION).compute.kaya_lmdi(
                ref_scenario=("model_a", "scen_a", "World"),
                int_scenario=("model_a", "scen_b", "World"),
                append=True,
            )
        )
        # assert that no data was added
        assert_iamframe_equal(TEST_DF, obs)
    else:
        obs = TEST_DF.filter(
            variable=input_variable_names.POPULATION
        ).compute.kaya_lmdi(
            ref_scenario=("model_a", "scen_a", "World"),
            int_scenario=("model_a", "scen_b", "World"),
        )
        assert obs is None


def test_calling_kaya_lmdi_multiple_times():
    """Test calling the method a second time has no effect"""

    obs = TEST_DF.copy()
    obs.compute.kaya_lmdi(
        ref_scenario=("model_a", "scen_a", "World"),
        int_scenario=("model_a", "scen_b", "World"),
        append=True,
    )
    obs.compute.kaya_lmdi(
        ref_scenario=("model_a", "scen_a", "World"),
        int_scenario=("model_a", "scen_b", "World"),
        append=True,
    )
    assert_iamframe_equal(TEST_DF.append(EXP_DF), obs)
