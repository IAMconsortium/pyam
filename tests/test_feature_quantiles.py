from pyam import IamDataFrame
import pytest
from pyam.testing import assert_iamframe_equal
import pandas as pd


def test_quantile_one_variable(test_pd_df):
    """Tests interquartile range of standard test df

    Because it is only two datapoints, the only 'new' computation
    is the median
    """
    df = IamDataFrame(test_pd_df)
    quantiles = (0.25, 0.5, 0.75)
    obs = df.filter(variable="Primary Energy").compute.quantiles(quantiles)
    exp = IamDataFrame(
        pd.DataFrame(
            {
                "scenario": [str(q) for q in quantiles],
                "2005": [1, (1.0 + 2) / 2, 2],
                "2010": [6, (6 + 7) / 2, 7],
            }
        ),
        model="Quantiles",
        region="World",
        variable="Primary Energy",
        unit="EJ/yr",
    )
    assert_iamframe_equal(exp, obs)


def test_quantile_missing_variable(test_pd_df):
    df = IamDataFrame(test_pd_df)
    with pytest.raises(ValueError):
        df.compute.quantiles((0.25, 0.5))
