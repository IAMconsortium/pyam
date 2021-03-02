from requests.exceptions import ConnectionError, ReadTimeout
import pytest
import logging
import pandas as pd

from pyam import IamDataFrame, IAMC_IDX, read_worldbank
from pyam.testing import assert_iamframe_equal
from pandas_datareader import wb

logger = logging.getLogger(__name__)

try:
    wb.get_indicators()
    WB_UNAVAILABLE = False
except (ReadTimeout, ConnectionError):
    WB_UNAVAILABLE = True

WB_REASON = "World Bank API unavailable"

WB_DF = pd.DataFrame(
    [
        ["foo", "WDI", "Canada", "GDP", "n/a", 39356.9, 40517.5, 42141.8],
        ["foo", "WDI", "Mexico", "GDP", "n/a", 17262.3, 17693.0, 17846.0],
        ["foo", "WDI", "United States", "GDP", "n/a", 51569.8, 53035.7, 54395.4],
    ],
    columns=IAMC_IDX + [2003, 2004, 2005],
)


@pytest.mark.skipif(WB_UNAVAILABLE, reason=WB_REASON)
def test_worldbank():
    try:
        obs = read_worldbank(model="foo", indicator={"NY.GDP.PCAP.PP.KD": "GDP"})
        exp = IamDataFrame(WB_DF)
        # test data with 5% relative tolerance to guard against minor data changes
        assert_iamframe_equal(obs, exp, rtol=5.0e-2)
    except ReadTimeout:
        logger.error("Timeout when reading from WorldBank API!")
