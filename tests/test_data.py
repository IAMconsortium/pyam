from datetime import datetime

import pytest
from pandas import testing as pdt

from pyam import IamDataFrame


@pytest.mark.parametrize("inplace", [True, False])
def test_data_sort(test_df, inplace):
    """Assert that data can be sorted as expected"""

    # revert order of _data
    df = IamDataFrame(test_df.data.iloc[[5, 4, 3, 2, 1, 0]])

    # assert that data is not sorted as expected
    with pytest.raises(AssertionError):
        pdt.assert_frame_equal(df.data, test_df.data)

    # assert that data is sorted as expected
    if inplace:
        obs = df.copy()
        obs.sort_data(inplace=True)
    else:
        obs = df.sort_data()
    pdt.assert_frame_equal(obs.data, test_df.data)


@pytest.mark.parametrize("inplace", [True, False])
def test_data_sort_mixed_time_domain(test_df_year, inplace):
    """Assert that timeseries with mixed time domain can be sorted as expected"""

    # TODO implement mixed df in conftest.py
    mixed_data = test_df_year.data
    mixed_data.year.replace({2005: datetime(2005, 1, 1, 0, 0)}, inplace=True)
    mixed_data.rename(columns={"time": "year"}, inplace=True)

    # revert order of _data
    df = IamDataFrame(mixed_data.iloc[[5, 4, 3, 2, 1, 0]])

    # assert that data is not sorted as expected
    with pytest.raises(AssertionError):
        pdt.assert_frame_equal(df.data, mixed_data)

    # assert that data is sorted as expected
    if inplace:
        obs = df.copy()
        obs.sort_data(inplace=True)
    else:
        obs = df.sort_data()
    pdt.assert_frame_equal(obs.data, mixed_data)
