import logging

import pandas as pd
import pytest

from pyam import IamDataFrame
from pyam.testing import assert_iamframe_equal

TEST_DF = IamDataFrame(
    pd.DataFrame(
        [
            ["Population", "million", 1000],
            ["GDP|PPP", "billion USD_2005/yr", 6],
            ["GDP|MER", "billion USD_2005/yr", 5],
            ["Final Energy", "EJ/yr", 8],
            ["Primary Energy", "EJ/yr", 10],
            ["Primary Energy|Coal", "EJ/yr", 5],
            ["Primary Energy|Gas", "EJ/yr", 2],
            ["Primary Energy|Oil", "EJ/yr", 2],
            [
                "Emissions|CO2|Fossil Fuels and Industry",
                "Mt CO2/yr",
                10,
            ],
            ["Emissions|CO2|Industrial Processes", "Mt CO2/yr", 1],
            ["Emissions|CO2|AFOLU", "Mt CO2/yr", 1],
            ["Emissions|CO2|Carbon Capture and Storage", "Mt CO2/yr", 4],
            ["Emissions|CO2|Carbon Capture and Storage|Biomass", "Mt CO2/yr", 1],
            ["Carbon Sequestration|CCS|Fossil|Energy", "Mt CO2/yr", 2],
            ["Carbon Sequestration|CCS|Fossil|Industrial Processes", "Mt CO2/yr", 1],
            ["Carbon Sequestration|CCS|Biomass|Energy", "Mt CO2/yr", 0.5],
            ["Carbon Sequestration|CCS|Biomass|Industrial Processes", "Mt CO2/yr", 0.5],
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
            ["Population", "million", 1000],
            ["GDP|PPP", "billion USD_2005/yr", 6],
            ["Final Energy", "EJ/yr", 8.0],
            ["Primary Energy", "EJ/yr", 10.0],
            ["Primary Energy|Fossil", "EJ/yr", 9.0],
            ["Total Fossil Carbon", "Mt CO2/yr", 12.0],
            ["Net Fossil Carbon", "Mt CO2/yr", 10.0],
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
            ["Primary Energy|Fossil", "EJ/yr", 9.0],
            ["Total Fossil Carbon", "Mt CO2/yr", 12.0],
            ["Net Fossil Carbon", "Mt CO2/yr", 10.0],
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
        (obs.filter(variable="Population").compute.kaya_variables(append=True))
        # assert that no data was added
        assert_iamframe_equal(TEST_DF, obs)
    else:
        obs = TEST_DF.filter(variable="Population").compute.kaya_variables()
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
    df_no_pop = TEST_DF.filter(variable="Population", keep=False)

    with caplog.at_level(logging.INFO):
        df_no_pop.compute.kaya_variables()

    # Check that the log message contains expected information
    assert "model: model_a" in caplog.text
    assert "scenario: scen_a" in caplog.text
    assert "region: World" in caplog.text

    assert "Population" in caplog.text


def test_kaya_variables_uses_gdp_mer_fallback():
    """Test that kaya_variables uses GDP_MER when GDP_PPP is not available"""
    # Create test data without GDP_PPP
    df_no_gdp_ppp = TEST_DF.filter(variable="GDP|PPP", keep=False)

    # Create expected result without GDP_MER instead of GDP_PPP
    exp_no_gdp_ppp = EXP_DF.filter(variable="GDP|PPP", keep=False).append(
        TEST_DF.filter(variable="GDP|MER")
    )

    # Compute kaya variables
    obs = df_no_gdp_ppp.compute.kaya_variables()

    # Verify results match expected
    assert_iamframe_equal(exp_no_gdp_ppp, obs)


def test_kaya_variables_returns_none_when_no_gdp_available():
    """Test that kaya_variables returns None both
    GDP_MER and GDP_PPP are unavailable"""
    # Create test data without GDP_PPP
    df_no_gdp = TEST_DF.filter(
        variable=["GDP|PPP", "GDP|MER"],
        keep=False,
    )

    # Compute kaya variables
    obs = df_no_gdp.compute.kaya_variables()

    # Verify results match expected
    assert obs is None
