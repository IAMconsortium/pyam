import pandas as pd
from pyam import IamDataFrame, IAMC_IDX
from pyam.testing import assert_iamframe_equal
import pytest

from conftest import META_DF


EXP_DF = IamDataFrame(
    pd.DataFrame(
        [
            ["model_a", "scen_a", "World", "Growth Rate", "", 0.430969],
            ["model_a", "scen_b", "World", "Growth Rate", "", 0.284735],
        ],
        columns=IAMC_IDX + [2005],
    ),
    meta=META_DF,
)


@pytest.mark.parametrize("append", (False, True))
def test_learning_rate(test_df_year, append):
    """Check computing the growth rate from an IamDataFrame"""

    if append:
        obs = test_df_year.copy()
        obs.compute.growth_rate({"Primary Energy": "Growth Rate"}, append=True)
        assert_iamframe_equal(test_df_year.append(EXP_DF), obs)
    else:
        obs = test_df_year.compute.growth_rate({"Primary Energy": "Growth Rate"})
        assert_iamframe_equal(EXP_DF, obs)


@pytest.mark.parametrize("append", (False, True))
def test_learning_rate_empty(test_df_year, append):
    """Assert that computing the growth rate with invalid variables returns empty"""

    if append:
        obs = test_df_year.copy()
        obs.compute.growth_rate({"foo": "bar"}, append=True)
        assert_iamframe_equal(test_df_year, obs)  # assert that no data was added
    else:
        obs = test_df_year.compute.growth_rate({"foo": "bar"})
        assert obs.empty
