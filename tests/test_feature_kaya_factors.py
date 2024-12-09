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
            ["FE/GNP", "EJ / USD / billion", 1.33333],
            ["GNP/P", "USD * billion / million / a", 0.006000],
            ["NFC/TFC", "", 0.833333],
            ["PEDEq/FE", "", 1.250000],
            ["PEFF/PEDEq", "", 0.900000],
            ["TFC/PEFF", "Mt CO2/EJ", 1.333333],
            ["Population", "million", 1000],
            ["Total Fossil Carbon", "Mt CO2/yr", 12.0],
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
            ["FE/GNP", "EJ / USD / billion", 1.33333],
            ["GNP/P", "USD * billion / million / a", 0.006000],
            ["NFC/TFC", "", 0.833333],
            ["PEDEq/FE", "", 1.250000],
            ["PEFF/PEDEq", "", 0.900000],
            ["TFC/PEFF", "Mt CO2/EJ", 1.333333],
            ["Total Fossil Carbon", "Mt CO2/yr", 12.0],
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
        obs.compute.kaya_factors(append=True)
        assert_iamframe_equal(TEST_DF.append(EXP_DF_FOR_APPEND), obs)
    else:
        obs = TEST_DF.compute.kaya_factors()
        assert_iamframe_equal(EXP_DF, obs)


@pytest.mark.parametrize("append", (False, True))
def test_kaya_variables_none_when_input_variables_missing(append):
    """Assert that computing kaya variables with
    missing input variables returns None
    """

    if append:
        obs = TEST_DF.copy()
        # select subset of required input variables
        (obs.filter(variable="Population").compute.kaya_factors(append=True))
        # assert that no data was added
        assert_iamframe_equal(TEST_DF, obs)
    else:
        obs = TEST_DF.filter(variable="Population").compute.kaya_factors()
        assert obs is None


def test_calling_kaya_factors_multiple_times():
    """Test calling the method a second time has no effect"""

    obs = TEST_DF.copy()
    obs.compute.kaya_factors(append=True)
    obs.compute.kaya_factors(append=True)
    assert_iamframe_equal(TEST_DF.append(EXP_DF_FOR_APPEND), obs)
