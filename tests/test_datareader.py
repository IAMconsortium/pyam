from requests.exceptions import ConnectionError
import pytest
import pandas as pd

from pyam import IamDataFrame, IAMC_IDX, read_worldbank
from pyam.testing import assert_iamframe_equal
from pandas_datareader import wb

try:
    wb.get_indicators()
    WB_UNAVAILABLE = False
except ConnectionError:
    WB_UNAVAILABLE = True

WB_REASON = 'World Bank API unavailable'

WB_DF = pd.DataFrame([
    ['foo', 'WDI', 'Canada', 'GDP', 'n/a', 39473.4, 40637.4, 42266.5],
    ['foo', 'WDI', 'Mexico', 'GDP', 'n/a', 17231.7, 17661.6, 17815.2],
    ['foo', 'WDI', 'United States', 'GDP', 'n/a', 51643.7, 53111.8, 54473.4]
], columns=IAMC_IDX + [2003, 2004, 2005])


@pytest.mark.skipif(WB_UNAVAILABLE, reason=WB_REASON)
def test_worldbank():
    obs = read_worldbank(model='foo', indicator={'NY.GDP.PCAP.PP.KD': 'GDP'})
    exp = IamDataFrame(WB_DF)
    assert_iamframe_equal(obs, exp)
