import pandas as pd
import pytest


def test_slice_len(test_df_year):
    """Check the length of a slice"""

    assert len(test_df_year.slice(scenario="scen_a")) == 4


def test_slice_index_attributes(test_df):
    # assert that the index and data column attributes are set correctly in an IamSlice

    s = test_df.slice()

    assert s.model == ["model_a"]
    assert s.scenario == ["scen_a", "scen_b"]
    assert s.region == ["World"]
    assert s.variable == ["Primary Energy", "Primary Energy|Coal"]
    assert s.unit == ["EJ/yr"]
    if test_df.time_col == "year":
        assert s.year == [2005, 2010]
    else:
        match = "'IamSlice' object has no attribute 'year'"
        with pytest.raises(AttributeError, match=match):
            s.year
    assert s.time.equals(pd.Index(test_df.data[test_df.time_col].unique()))


def test_filtered_slice_index_attributes(test_df_year):
    # assert that the attributes are set correctly in a filtered IamSlice

    s = test_df_year.slice(scenario="scen_b")
    assert s.scenario == ["scen_b"]


def test_print(test_df_year):
    """Assert that `print(IamSlice)` (and `info()`) returns as expected"""
    exp = "\n".join(
        [
            "<class 'pyam.slice.IamSlice'>",
            "Index dimensions and data coordinates:",
            "   model    : model_a (1)",
            "   scenario : scen_a, scen_b (2)",
            "   region   : World (1)",
            "   variable : Primary Energy, Primary Energy|Coal (2)",
            "   unit     : EJ/yr (1)",
            "   year     : 2005, 2010 (2)",
        ]
    )
    obs = test_df_year.slice().info()
    assert obs == exp
