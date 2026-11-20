import pandas as pd
import pytest

from pyam import IamDataFrame
from pyam.testing import assert_iamframe_equal


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


def test_quantile_custom_model_and_scenario_names(test_pd_df):
    df = IamDataFrame(test_pd_df)
    quantiles = (0.25, 0.5, 0.75)
    obs = df.filter(variable="Primary Energy").compute.quantiles(
        quantiles, model="My model", scenario=lambda q: f"q{int(q * 100)}"
    )
    exp = IamDataFrame(
        pd.DataFrame(
            {
                "scenario": ["q25", "q50", "q75"],
                "2005": [1, (1.0 + 2) / 2, 2],
                "2010": [6, (6 + 7) / 2, 7],
            }
        ),
        model="My model",
        region="World",
        variable="Primary Energy",
        unit="EJ/yr",
    )
    assert_iamframe_equal(exp, obs)


def test_quantile_custom_model_with_weights(test_pd_df):
    df = IamDataFrame(test_pd_df)
    weights = pd.Series(
        [1.0, 1.0],
        index=pd.MultiIndex.from_tuples(
            [("model_a", "scen_a"), ("model_a", "scen_b")], names=["model", "scenario"]
        ),
        name="weight",
    )
    obs = df.filter(variable="Primary Energy").compute.quantiles(
        (0.5,), weights=weights, model="Custom weighted"
    )
    assert obs.model == ["Custom weighted"]
    assert obs.scenario == ["0.5"]


def test_quantile_missing_variable(test_pd_df):
    df = IamDataFrame(test_pd_df)
    with pytest.raises(ValueError):
        df.compute.quantiles((0.25, 0.5))
