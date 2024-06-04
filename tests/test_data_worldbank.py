import logging

import pandas as pd
import pytest
from requests.exceptions import ReadTimeout

from pyam import IamDataFrame, read_worldbank
from pyam.testing import assert_iamframe_equal
from pyam.utils import IAMC_IDX

logger = logging.getLogger(__name__)

try:
    import wbdata  # noqa: F401

    WB_UNAVAILABLE = False
except ImportError:
    WB_UNAVAILABLE = True

WB_REASON = "World Bank API unavailable"

WB_DF = pd.DataFrame(
    [
        ["foo", "WDI", "Canada", "GDP", "n/a", 49231.9, 50283.0, 51409.4],
        ["foo", "WDI", "Mexico", "GDP", "n/a", 20065.3, 20477.6, 19144.0],
        ["foo", "WDI", "United States", "GDP", "n/a", 51569.8, 53035.7, 54395.4],
    ],
    columns=IAMC_IDX + [2003, 2004, 2005],
)


@pytest.mark.skipif(WB_UNAVAILABLE, reason=WB_REASON)
def test_worldbank():
    try:
        # Find the country codes via wbdata.get_countries(query="Canada") etc
        obs = read_worldbank(
            model="foo",
            indicators={"NY.GDP.PCAP.PP.KD": "GDP"},
            country=["CAN", "MEX", "USA"],
            date=("2003", "2005"),
        )
        exp = IamDataFrame(WB_DF)
        # test data with 5% relative tolerance to guard against minor data changes
        assert_iamframe_equal(obs, exp, rtol=5.0e-2)
    except ReadTimeout:
        logger.error("Timeout when reading from WorldBank API.")
