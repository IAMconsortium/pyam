import pytest
from datetime import datetime
from pyam import IamDataFrame, compare


@pytest.mark.parametrize("inplace", [True, False])
def test_swap_time_to_year(test_df, inplace):
    if test_df.time_col == "year":
        return  # year df not relevant for this test

    exp = test_df.data
    exp["year"] = exp["time"].apply(lambda x: x.year)
    exp = exp.drop("time", axis="columns")
    exp = IamDataFrame(exp)

    obs = test_df.swap_time_for_year(inplace=inplace)

    if inplace:
        assert obs is None
        obs = test_df

    assert compare(obs, exp).empty
    assert obs.year == [2005, 2010]
    with pytest.raises(AttributeError):
        obs.time


@pytest.mark.parametrize("inplace", [True, False])
def test_swap_time_to_year_errors(test_df, inplace):
    if test_df.time_col == "year":
        with pytest.raises(ValueError):
            test_df.swap_time_for_year(inplace=inplace)
        return

    tdf = test_df.data
    tdf["time"] = tdf["time"].apply(lambda x: datetime(2005, x.month, x.day))

    with pytest.raises(ValueError):
        IamDataFrame(tdf).swap_time_for_year(inplace=inplace)