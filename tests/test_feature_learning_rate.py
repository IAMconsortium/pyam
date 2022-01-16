import pandas as pd
from pyam import IamDataFrame, IAMC_IDX
from pyam.testing import assert_iamframe_equal
import pytest


TEST_DF = IamDataFrame(
    pd.DataFrame(
        [
            ["model_a", "scen_a", "World", "Cap", "GW", 1, 2],
            ["model_a", "scen_a", "World", "Cost", "US$2010/kW", 1, 0.5],
            ["model_a", "scen_b", "World", "Cap", "GW", 0.1, 0.2],
            ["model_a", "scen_b", "World", "Cost", "US$2010/kW", 1, 0.5],
            ["model_a", "scen_c", "World", "Cap", "GW", 10, 20],
            ["model_a", "scen_c", "World", "Cost", "US$2010/kW", 1, 0.5],
            ["model_a", "scen_d", "World", "Cap", "GW", 1, 2],
            ["model_a", "scen_d", "World", "Cost", "US$2010/kW", 1, 0.75],
            ["model_a", "scen_e", "World", "Cap", "GW", 1, 2],
            ["model_a", "scen_e", "World", "Cost", "US$2010/kW", 1, 0.25],
        ],
        columns=IAMC_IDX + [2005, 2010],
    )
)

EXP_DF = IamDataFrame(
    pd.DataFrame(
        [
            ["model_a", "scen_a", "World", "Learning Rate", "", 0.5],
            ["model_a", "scen_b", "World", "Learning Rate", "", 0.5],
            ["model_a", "scen_c", "World", "Learning Rate", "", 0.5],
            ["model_a", "scen_d", "World", "Learning Rate", "", 0.25],
            ["model_a", "scen_e", "World", "Learning Rate", "", 0.75],
        ],
        columns=IAMC_IDX + [2005],
    )
)


@pytest.mark.parametrize("append", (False, True))
def test_learning_rate(append):
    """Check computing the learning rate"""

    if append:
        obs = TEST_DF.copy()
        obs.compute.learning_rate("Learning Rate", "Cost", "Cap", append=True)
        assert_iamframe_equal(TEST_DF.append(EXP_DF), obs)
    else:
        obs = TEST_DF.compute.learning_rate("Learning Rate", "Cost", "Cap")
        assert_iamframe_equal(EXP_DF, obs)


@pytest.mark.parametrize("append", (False, True))
def test_learning_rate_empty(append):
    """Assert that computing the learning rate with invalid variables returns empty"""

    if append:
        obs = TEST_DF.copy()
        obs.compute.learning_rate("Learning Rate", "foo", "Cap", append=True)
        assert_iamframe_equal(TEST_DF, obs)  # assert that no data was added
    else:
        obs = TEST_DF.compute.learning_rate("Learning Rate", "foo", "Cap")
        assert obs.empty
