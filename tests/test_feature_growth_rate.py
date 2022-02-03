import math
import pandas as pd
import pandas.testing as pdt
from pyam import IamDataFrame, IAMC_IDX
from pyam.testing import assert_iamframe_equal
from pyam.timeseries import growth_rate
import pytest

from .conftest import META_DF


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
def test_growth_rate(test_df_year, append):
    """Check computing the growth rate from an IamDataFrame"""

    if append:
        obs = test_df_year.copy()
        obs.compute.growth_rate({"Primary Energy": "Growth Rate"}, append=True)
        assert_iamframe_equal(test_df_year.append(EXP_DF), obs)
    else:
        obs = test_df_year.compute.growth_rate({"Primary Energy": "Growth Rate"})
        assert_iamframe_equal(EXP_DF, obs)


@pytest.mark.parametrize("append", (False, True))
def test_growth_rate_empty(test_df_year, append):
    """Assert that computing the growth rate with invalid variables returns empty"""

    if append:
        obs = test_df_year.copy()
        obs.compute.growth_rate({"foo": "bar"}, append=True)
        assert_iamframe_equal(test_df_year, obs)  # assert that no data was added
    else:
        obs = test_df_year.compute.growth_rate({"foo": "bar"})
        assert obs.empty


@pytest.mark.parametrize("x2010", (1, 27, -3))
@pytest.mark.parametrize("rates", ([0.05, 1.25], [0.5, -0.5]))
def test_growth_rate_timeseries(x2010, rates):
    """Check several combinations of growth rates directly on the timeseries"""

    x2013 = x2010 * math.pow(1 + rates[0], 3)  # 3 years: 2010 - 2013
    x2017 = x2013 * math.pow(1 + rates[1], 4)  # 4 years: 2013 - 2017

    pdt.assert_series_equal(
        growth_rate(pd.Series([x2010, x2013, x2017], index=[2010, 2013, 2017])),
        pd.Series(rates, index=[2010, 2013]),
    )


@pytest.mark.parametrize("value", (0, -1))
def test_growth_rate_timeseries_fails(value):
    """Check that a timeseries reaching/crossing 0 raises"""

    with pytest.raises(ValueError, match="Cannot compute growth rate when*."):
        growth_rate(pd.Series([1.0, value]))
