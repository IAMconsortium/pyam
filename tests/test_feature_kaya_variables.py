import pandas as pd
import pytest
import logging

from pyam import IamDataFrame
from pyam.kaya import input_variable_names, kaya_variable_names
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
            [input_variable_names.POPULATION, "million", 1000],
            [input_variable_names.GDP_PPP, "billion USD_2005/yr", 6],
            [input_variable_names.FINAL_ENERGY, "EJ/yr", 8.0],
            [input_variable_names.PRIMARY_ENERGY, "EJ/yr", 10.0],
            [kaya_variable_names.PRIMARY_ENERGY_FF, "EJ/yr", 9.0],
            [kaya_variable_names.TFC, "Mt CO2/yr", 12.0],
            [kaya_variable_names.NFC, "Mt CO2/yr", 10.0],
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
            [kaya_variable_names.PRIMARY_ENERGY_FF, "EJ/yr", 9.0],
            [kaya_variable_names.TFC, "Mt CO2/yr", 12.0],
            [kaya_variable_names.NFC, "Mt CO2/yr", 10.0],
        ],
        columns=["variable", "unit", 2010],
    ),
    model="model_a",
    scenario="scen_a",
    region="World",
)


@pytest.mark.parametrize("append", (False, True))
def test_kaya_variables(append):
    """Test computing kaya variables"""

    if append:
        obs = TEST_DF.copy()
        obs.compute.kaya_variables(append=True)
        assert_iamframe_equal(TEST_DF.append(EXP_DF_FOR_APPEND), obs)
    else:
        obs = TEST_DF.compute.kaya_variables()
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
            obs.filter(variable=input_variable_names.POPULATION).compute.kaya_variables(
                append=True
            )
        )
        # assert that no data was added
        assert_iamframe_equal(TEST_DF, obs)
    else:
        obs = TEST_DF.filter(
            variable=input_variable_names.POPULATION
        ).compute.kaya_variables()
        assert obs is None


def test_calling_kaya_variables_multiple_times():
    """Test calling the method a second time has no effect"""

    obs = TEST_DF.copy()
    obs.compute.kaya_variables(append=True)
    obs.compute.kaya_variables(append=True)
    assert_iamframe_equal(TEST_DF.append(EXP_DF_FOR_APPEND), obs)


def test_kaya_variables_logs_missing_variables(caplog):
    """Test that missing variables are correctly logged"""
    # Create test data with only population
    df_no_pop = TEST_DF.filter(variable=input_variable_names.POPULATION, keep=False)

    with caplog.at_level(logging.INFO):
        df_no_pop.compute.kaya_variables()

    # Check that the log message contains expected information
    assert (
        "Variables missing for model: model_a, scenario: scen_a, region: World"
        in caplog.text
    )

    assert input_variable_names.POPULATION in caplog.text


def test_kaya_variables_uses_gdp_mer_fallback():
    """Test that kaya_variables uses GDP_MER when GDP_PPP is not available"""
    # Create test data without GDP_PPP
    df_no_gdp_ppp = TEST_DF.filter(variable=input_variable_names.GDP_PPP, keep=False)

    # Create expected result without GDP_MER instead of GDP_PPP
    exp_no_gdp_ppp = EXP_DF.filter(
        variable=input_variable_names.GDP_PPP, keep=False
    ).append(TEST_DF.filter(variable=input_variable_names.GDP_MER))

    # Compute kaya variables
    obs = df_no_gdp_ppp.compute.kaya_variables()

    # Verify results match expected
    assert_iamframe_equal(exp_no_gdp_ppp, obs)


def test_kaya_variables_returns_none_when_no_gdp_available():
    """Test that kaya_variables returns None both
    GDP_MER and GDP_PPP are unavailable"""
    # Create test data without GDP_PPP
    df_no_gdp = TEST_DF.filter(
        variable=[input_variable_names.GDP_PPP, input_variable_names.GDP_MER],
        keep=False,
    )

    # Compute kaya variables
    obs = df_no_gdp.compute.kaya_variables()

    # Verify results match expected
    assert obs is None
