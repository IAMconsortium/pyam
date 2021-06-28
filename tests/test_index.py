import pytest
import pandas as pd
import pandas.testing as pdt

from pyam.index import get_index_levels, replace_index_values, append_index_level
from pyam import IAMC_IDX


def test_get_index_levels(test_df_index):
    """Assert that get_index_levels returns the correct values"""
    assert get_index_levels(test_df_index, "scenario") == ["scen_a", "scen_b"]


def test_get_index_levels_raises(test_df_index):
    """Assert that get_index_levels raises with non-existing level"""
    with pytest.raises(KeyError):
        get_index_levels(test_df_index, "foo")


@pytest.mark.parametrize(
    "exp_scen, mapping",
    [
        (["scen_c", "scen_c", "scen_b"], {"scen_a": "scen_c"}),
        (["scen_c", "scen_c", "scen_c"], {"scen_a": "scen_c", "scen_b": "scen_c"}),
        (["scen_b", "scen_b", "scen_c"], {"scen_a": "scen_b", "scen_b": "scen_c"})
        # this test ensures that no transitive replacing occurs
    ],
)
@pytest.mark.parametrize("rows", (None, [False, True, True]))
def test_replace_index_level(test_pd_df, test_df_index, exp_scen, mapping, rows):
    """Assert that replace_index_value works as expected"""

    test_pd_df["scenario"] = exp_scen if rows is None else ["scen_a"] + exp_scen[1:]
    exp = test_pd_df.set_index(IAMC_IDX)

    test_df_index.index = replace_index_values(test_df_index, "scenario", mapping, rows)
    pdt.assert_frame_equal(exp, test_df_index)


def test_replace_index_level_raises(test_df_index):
    """Assert that replace_index_value raises with non-existing level"""
    with pytest.raises(KeyError):
        replace_index_values(test_df_index, "foo", {"scen_a": "scen_c"})


def test_append_index():
    """Assert that appending and re-ordering to an index works as expected"""

    index = pd.MultiIndex(
        codes=[[0, 1]],
        levels=[["scen_a", "scen_b"]],
        names=["scenario"],
    )

    obs = append_index_level(index, 0, "World", "region", order=["region", "scenario"])

    exp = pd.MultiIndex(
        codes=[[0, 0], [0, 1]],
        levels=[["World"], ["scen_a", "scen_b"]],
        names=["region", "scenario"],
    )
    pdt.assert_index_equal(obs, exp)
