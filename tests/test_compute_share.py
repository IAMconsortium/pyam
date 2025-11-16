import pandas as pd
import pytest

from pyam import IAMC_IDX, IamDataFrame
from pyam.testing import assert_iamframe_equal


@pytest.mark.parametrize("append", (False, True))
def test_compute_share_variable(test_df_year, append):
    """Check that computing the share works as expected"""

    data = pd.DataFrame(
        [["World", "Primary Energy|Coal [Share]", "%", 50.0, 50.0]],
        columns=["region", "variable", "unit", 2005, 2010],
    )
    exp = IamDataFrame(data, model="model_a", scenario="scen_a", meta=test_df_year.meta)

    args = ("Primary Energy|Coal", "Primary Energy", "Primary Energy|Coal [Share]")
    if append:
        obs = test_df_year.copy()
        obs.compute.share(*args, append=append)
        exp = test_df_year.append(exp)
    else:
        obs = test_df_year.compute.share(*args)

    assert_iamframe_equal(exp, obs)


@pytest.mark.parametrize("append", (False, True))
def test_compute_share_scenario(test_df_year, append):
    """Check that computing the share works as expected"""

    data = pd.DataFrame(
        [["World", "Primary Energy|Coal [Share]", "%", 50.0, 50.0]],
        columns=["region", "variable", "unit", 2005, 2010],
    )
    exp = IamDataFrame(data, model="model_a", scenario="scen_a", meta=test_df_year.meta)

    args = ("Primary Energy|Coal", "Primary Energy", "Primary Energy|Coal [Share]")
    if append:
        obs = test_df_year.copy()
        obs.compute.share(*args, append=append)
        exp = test_df_year.append(exp)
    else:
        obs = test_df_year.compute.share(*args)

    assert_iamframe_equal(exp, obs)


@pytest.mark.parametrize("append", (False, True))
def test_compute_share_scenario(test_df_year, append):
    """Check that computing the share works on a custom axis works as expected"""

    data = pd.DataFrame(
        [["scenario [Share]", "Primary Energy", "%", 50.0, 6 / 7 * 100]],
        columns=["scenario", "variable", "unit", 2005, 2010],
    )
    exp = IamDataFrame(data, model="model_a", region="World", meta=test_df_year.meta)

    args = ("scen_a", "scen_b", "scenario [Share]")
    if append:
        obs = test_df_year.copy()
        obs.compute.share(*args, axis="scenario", append=append)
        exp = test_df_year.append(exp)
    else:
        obs = test_df_year.compute.share(*args, axis="scenario")

    assert_iamframe_equal(exp, obs)


def test_compute_share_fail_mismatching_unit(test_df_year):
    # rename the unit for the variable "Primary Energy|Coal"
    test_df_year.rename(
        variable={"Primary Energy|Coal": "Primary Energy|Coal"},
        unit={"EJ/yr": "EJ / a"},
        inplace=True,
    )

    match = "Mismatching units: 'EJ / a' != 'EJ/yr'"
    with pytest.raises(ValueError, match=match):
        test_df_year.compute.share(
            "Primary Energy|Coal", "Primary Energy", "Primary Energy|Coal [Share]"
        )


def test_compute_share_fail_nonunique_unit(test_df_year):
    """Add a row so that the units of "Primary Energy|Coal"""
    test_df_year.append(
        pd.DataFrame(
            [["model_a", "scen_b", "World", "Primary Energy|Coal", "EJ / a", 1, 2]],
            columns=IAMC_IDX + [2005, 2010],
        ),
        inplace=True,
    )

    match = "Units of `a` not unique: EJ / a, EJ/yr"
    with pytest.raises(ValueError, match=match):
        test_df_year.compute.share(
            "Primary Energy|Coal", "Primary Energy", "Primary Energy|Coal [Share]"
        )
