def test_slice_len(test_df_year):
    """Check the length of a slice"""

    assert len(test_df_year.slice(scenario="scen_a")) == 4
